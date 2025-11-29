#!/usr/bin/env python3
"""Evaluate retrieval quality on BEIR-style datasets."""

from __future__ import annotations

import argparse
import json
import time
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List
from tqdm import tqdm

from weighted_rag import WeightedRAGPipeline
from weighted_rag.config import PipelineConfig, pipeline_config_from_dict
from weighted_rag.evaluation.loader import load_beir_split
from weighted_rag.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k
from weighted_rag.types import Query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on BEIR-style datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to BEIR split (contains corpus.jsonl, queries.jsonl, qrels.tsv).")
    parser.add_argument("--config", type=Path, help="Optional JSON config path.")
    parser.add_argument("--ks", type=str, default="1,3,5,10", help="Comma-separated list of cutoff values for metrics.")
    parser.add_argument("--max-queries", type=int, help="Limit evaluation to the first N queries.")
    parser.add_argument("--save-results", type=Path, help="Optional path to dump per-query results JSON.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/indexes"), help="Directory to cache vector indexes.")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing even if cache exists.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching entirely.")
    return parser.parse_args()


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        # Use a safe default configuration
        from weighted_rag.config import EmbeddingConfig, ChunkingConfig, RetrievalConfig, IndexStageConfig
        
        embedding_config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16,
            use_fp16=False,
            device="cpu",
            normalize=True,
            truncate_dims=None
        )
        
        chunking_config = ChunkingConfig(
            max_tokens=128,  # Very small chunks
            overlap_tokens=16,
            tokenizer_name="bert-base-uncased"
        )
        
        retrieval_config = RetrievalConfig(
            stages=[
                IndexStageConfig(
                    name="coarse",
                    dimension=384,
                    top_k=200,
                    weight=1.0,
                    normalize=True,
                    index_factory="HNSW32",
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            ]
        )
        
        return PipelineConfig(
            chunking=chunking_config,
            embedding=embedding_config,
            retrieval=retrieval_config
        )
    payload: Dict[str, Any] = json.loads(path.read_text())
    return pipeline_config_from_dict(payload)


def get_corpus_hash(dataset_root: Path) -> str:
    """Hash the corpus file to detect changes."""
    corpus_path = dataset_root / "corpus.jsonl"
    if not corpus_path.exists():
        return "no-corpus"
    
    # Hash file modification time and size for speed
    # For more reliability, could hash content but that's slower
    stat = corpus_path.stat()
    content = f"{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(content.encode()).hexdigest()[:8]


def compute_cache_key(dataset_root: Path, config: PipelineConfig) -> str:
    """Generate a unique cache key based on dataset and config."""
    # Include corpus hash to detect when chunks change
    corpus_hash = get_corpus_hash(dataset_root)
    
    # Serialize config
    config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
    
    # Combine everything
    combined = f"{dataset_root.name}:{corpus_hash}:{config_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def get_cache_path(cache_dir: Path, dataset_name: str, cache_key: str) -> Path:
    """Get the cache file path for a given cache key."""
    return cache_dir / dataset_name / f"index_{cache_key}.pkl"


def get_cache_metadata_path(cache_path: Path) -> Path:
    """Get metadata file path for a cache."""
    return cache_path.with_suffix('.json')


def save_pipeline(pipeline: WeightedRAGPipeline, cache_path: Path, metadata: Dict[str, Any]) -> None:
    """Save the pipeline index to disk with metadata."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving index to cache: {cache_path}")
    start = time.perf_counter()
    
    with open(cache_path, 'wb') as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save metadata
    metadata_path = get_cache_metadata_path(cache_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    
    elapsed = time.perf_counter() - start
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"✓ Index cached in {elapsed:.1f}s ({size_mb:.1f} MB)")


def load_pipeline(cache_path: Path, config: PipelineConfig) -> WeightedRAGPipeline | None:
    """Load a cached pipeline from disk."""
    if not cache_path.exists():
        return None
    
    try:
        print(f"Loading cached index from: {cache_path}")
        
        # Load and display metadata
        metadata_path = get_cache_metadata_path(cache_path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            print(f"  Cache created: {metadata.get('created_at', 'unknown')}")
            print(f"  Documents: {metadata.get('num_documents', 'unknown')}")
            print(f"  Chunks: {metadata.get('num_chunks', 'unknown')}")
        
        start = time.perf_counter()
        with open(cache_path, 'rb') as f:
            pipeline = pickle.load(f)
        
        elapsed = time.perf_counter() - start
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"✓ Index loaded in {elapsed:.1f}s ({size_mb:.1f} MB)")
        return pipeline
        
    except Exception as e:
        print(f"⚠ Failed to load cached index: {e}")
        print("  Will rebuild index from scratch...")
        return None


def unique_doc_sequence(chunk_ids: Iterable[str], pipeline: WeightedRAGPipeline) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for chunk_id in chunk_ids:
        doc_id = pipeline.index.get_chunk(chunk_id).doc_id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
    return ordered


def compute_metrics(ranked_docs: List[str], qrels: Dict[str, float], ks: List[int]) -> Dict[str, float]:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    if not relevant:
        return {}

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"precision@{k}"] = precision_at_k(relevant, ranked_docs, k)
        metrics[f"recall@{k}"] = recall_at_k(relevant, ranked_docs, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k([qrels.get(doc_id, 0.0) for doc_id in ranked_docs], k)
    metrics["mrr"] = mean_reciprocal_rank(relevant, ranked_docs)
    return metrics


def merge_metrics(accumulator: Dict[str, float], update: Dict[str, float]) -> None:
    for key, value in update.items():
        accumulator[key] = accumulator.get(key, 0.0) + value


def average_metrics(metrics: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(count, 1) for key, value in metrics.items()}


def build_index(split: Any, pipeline: WeightedRAGPipeline) -> int:
    """Build index from documents and return total chunks."""
    documents = list(split.corpus.values())
    print(f"Indexing {len(documents)} documents from corpus...")
    
    # Process documents in very small batches to show progress
    batch_size = 5
    total_chunks = 0
    
    with tqdm(total=len(documents), desc="Processing documents", unit="docs") as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                batch_chunks = pipeline.add_documents(batch)
                total_chunks += batch_chunks
                pbar.update(len(batch))
                pbar.set_postfix(total_chunks=total_chunks)
            except Exception as e:
                print(f"\n⚠ Error processing batch {i//batch_size + 1}: {e}")
                pbar.update(len(batch))
                continue
    
    print(f"✓ Indexed {total_chunks} chunks across {len(documents)} documents.")
    return total_chunks


def build_or_load_index(
    dataset_root: Path, 
    config: PipelineConfig, 
    cache_dir: Path, 
    force_reindex: bool,
    use_cache: bool
) -> WeightedRAGPipeline:
    """Build a new index or load from cache."""
    
    if not use_cache:
        print("Caching disabled, building new index...")
        split = load_beir_split(dataset_root)
        pipeline = WeightedRAGPipeline(config)
        build_index(split, pipeline)
        return pipeline
    
    cache_key = compute_cache_key(dataset_root, config)
    cache_path = get_cache_path(cache_dir, dataset_root.name, cache_key)
    
    # Try to load from cache
    if not force_reindex:
        pipeline = load_pipeline(cache_path, config)
        if pipeline is not None:
            return pipeline
        print("No valid cache found, building new index...")
    else:
        print("Force reindex enabled, building new index...")
    
    # Build new index
    split = load_beir_split(dataset_root)
    pipeline = WeightedRAGPipeline(config)
    total_chunks = build_index(split, pipeline)
    
    # Save to cache with metadata
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_root.name,
        "cache_key": cache_key,
        "corpus_hash": get_corpus_hash(dataset_root),
        "num_documents": len(split.corpus),
        "num_chunks": total_chunks,
        "config": json.loads(json.dumps(config.__dict__, default=str))
    }
    save_pipeline(pipeline, cache_path, metadata)
    
    return pipeline


def evaluate(
    dataset_root: Path, 
    config: PipelineConfig, 
    ks: List[int], 
    max_queries: int | None, 
    save_path: Path | None,
    cache_dir: Path,
    force_reindex: bool,
    use_cache: bool
) -> None:
    split = load_beir_split(dataset_root)
    
    # Build or load index with caching
    print("=" * 60)
    print(f"Dataset: {dataset_root.name}")
    print(f"Cache: {'Disabled' if not use_cache else ('Force rebuild' if force_reindex else 'Enabled')}")
    print("=" * 60)
    
    pipeline = build_or_load_index(dataset_root, config, cache_dir, force_reindex, use_cache)

    stage_names = [stage.name for stage in config.retrieval.stages]
    stage_metrics: Dict[str, Dict[str, float]] = {name: {} for name in stage_names}
    final_metrics: Dict[str, float] = {}
    evaluated = 0
    per_query_output: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    # Determine total queries to evaluate
    total_queries = len(split.queries)
    if max_queries is not None:
        total_queries = min(total_queries, max_queries)
    
    # Add progress bar for query evaluation
    queries_to_process = list(split.queries.items())[:total_queries] if max_queries else list(split.queries.items())
    
    print("\nEvaluating queries...")
    with tqdm(total=len(queries_to_process), desc="Evaluating queries", unit="queries") as pbar:
        for query_id, query in queries_to_process:
            if max_queries is not None and evaluated >= max_queries:
                break
            qrel = split.qrels.get(query_id)
            if not qrel:
                pbar.update(1)
                continue
            
            retrieval, stage_results = pipeline.retrieve_with_details(query)
            final_docs = unique_doc_sequence([item.chunk.chunk_id for item in retrieval.chunks], pipeline)
            query_metrics = compute_metrics(final_docs, qrel, ks)
            
            if not query_metrics:
                pbar.update(1)
                continue

            merge_metrics(final_metrics, query_metrics)

            stage_details: Dict[str, Any] = {}
            for name in stage_names:
                result = stage_results.get(name)
                if not result:
                    continue
                ranked_docs = unique_doc_sequence(result.ids, pipeline)
                metrics = compute_metrics(ranked_docs, qrel, ks)
                merge_metrics(stage_metrics.setdefault(name, {}), metrics)
                stage_details[name] = {
                    "retrieved": ranked_docs[: max(ks)],
                    "metrics": metrics,
                }

            per_query_output.append(
                {
                    "query_id": query_id,
                    "query": query.text,
                    "qrels": qrel,
                    "final_ranked_docs": final_docs[: max(ks)],
                    "metrics": query_metrics,
                    "stage_details": stage_details,
                }
            )
            evaluated += 1
            
            # Update progress bar with current metrics
            if evaluated % 5 == 0 and final_metrics:
                avg_metrics = average_metrics(final_metrics, evaluated)
                mrr = avg_metrics.get('mrr', 0)
                pbar.set_postfix(evaluated=evaluated, MRR=f"{mrr:.3f}")
            else:
                pbar.set_postfix(evaluated=evaluated)
            
            pbar.update(1)

    elapsed = time.perf_counter() - start_time
    print(f"\n✓ Evaluated {evaluated} queries in {elapsed:.2f}s ({evaluated / max(elapsed, 1e-6):.2f} qps)")

    if evaluated == 0:
        print("⚠ No queries evaluated (missing qrels?).")
        return

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nFinal stage metrics:")
    for key, value in sorted(average_metrics(final_metrics, evaluated).items()):
        print(f"  {key}: {value:.4f}")

    for name in stage_names:
        if not stage_metrics.get(name):
            continue
        print(f"\nStage '{name}' metrics:")
        for key, value in sorted(average_metrics(stage_metrics[name], evaluated).items()):
            print(f"  {key}: {value:.4f}")

    if save_path:
        payload = {
            "dataset_root": str(dataset_root),
            "evaluated_queries": evaluated,
            "metrics": average_metrics(final_metrics, evaluated),
            "stage_metrics": {name: average_metrics(metrics, evaluated) for name, metrics in stage_metrics.items()},
            "queries": per_query_output,
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"\n✓ Saved detailed results to {save_path}")


def main() -> None:
    args = parse_args()
    ks = sorted({int(value) for value in args.ks.split(",") if value.strip()})
    config = load_config(args.config)
    evaluate(
        args.dataset_root, 
        config, 
        ks, 
        args.max_queries, 
        args.save_results, 
        args.cache_dir, 
        args.force_reindex,
        not args.no_cache
    )


if __name__ == "__main__":
    main()