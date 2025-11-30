#!/usr/bin/env python3
"""Evaluate retrieval quality on BEIR-style datasets."""

from __future__ import annotations

# Set environment variables before any other imports to prevent semaphore leaks
import os

import argparse
import json
import time
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
    return parser.parse_args()


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        # Use a safe default configuration
        from weighted_rag.config import EmbeddingConfig, ChunkingConfig, RetrievalConfig, IndexStageConfig,CrossEncoderConfig
        
        # Use consistent model name for both embedding and retrieval stages
        model_name = "Qwen/Qwen3-Embedding-4B"
        
        embedding_config = EmbeddingConfig(
            model_name=model_name,
            batch_size=64,
            use_fp16=False,
            device="cuda",
            normalize=True,
            truncate_dims=None
        )
        
        chunking_config = ChunkingConfig(
            max_tokens=480,  # Reasonable chunk size
            overlap_tokens=32,
            tokenizer_name="Qwen/Qwen3-Embedding-4B"
        )
        
        retrieval_config = RetrievalConfig(
            stages=[
                IndexStageConfig(
                    name="coarse",
                    dimension=2560,
                    top_k=200,  # Smaller top-k to avoid memory issues
                    weight=1.0,
                    normalize=True,
                    index_factory="HNSW32",
                    model_name=model_name  # Use same model for consistency
                )
            ]
        )
                # Configure cross-encoder reranker
        cross_encoder_config = CrossEncoderConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L12-v2",
            device="cuda",
            batch_size=64,
            top_n=10  # Rerank top 10 results
        )
        
        config = PipelineConfig(
            chunking=chunking_config,
            embedding=embedding_config,
            retrieval=retrieval_config,
            cross_encoder= None
        )
        
        # Debug: Verify consistent model configuration
        print(f"Configuration verification:")
        print(f"  Embedding model: {config.embedding.model_name}")
        print(f"  Retrieval stage models: {[stage.model_name for stage in config.retrieval.stages]}")
        print(f"  Cross-encoder model: {config.cross_encoder.model_name if config.cross_encoder else 'None'}")
        print(f"  Cross-encoder reranking: {'Enabled' if config.cross_encoder else 'Disabled'}")
        print()
        
        return config
    else:
        payload: Dict[str, Any] = json.loads(path.read_text())
        config = pipeline_config_from_dict(payload)
        print(f"Loaded config from {path}")
        print(f"  Embedding model: {config.embedding.model_name}")
        print(f"  Retrieval stages: {[stage.model_name for stage in config.retrieval.stages]}")
        return config


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


def evaluate(dataset_root: Path, config: PipelineConfig, ks: List[int], max_queries: int | None, save_path: Path | None) -> None:
    split = load_beir_split(dataset_root)
    pipeline = WeightedRAGPipeline(config)

    print("Indexing corpus...")
    documents = list(split.corpus.values())
    print(f"Indexing {len(documents)} documents from corpus...")
    
    # Process documents in very small batches to show progress and avoid memory issues
    batch_size = 64  # Process only 5 documents at a time
    total_chunks = 0
    
    with tqdm(total=len(documents), desc="Processing documents", unit="docs") as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                batch_chunks = pipeline.add_documents(batch)
                total_chunks += batch_chunks
                
                pbar.update(len(batch))
                pbar.set_postfix(total_chunks=total_chunks, batch_chunks=batch_chunks)
                print(f"Batch completed: {batch_chunks} chunks created")
            except Exception as e:
                print(f"\nError processing batch {i//batch_size + 1}: {e}")
                print(f"Batch documents: {len(batch)}")
                if batch:
                    print(f"First doc length: {len(batch[0].text)}")
                # Skip this batch and continue
                pbar.update(len(batch))
                continue
    
    print(f"✓ Indexed {total_chunks} chunks across {len(documents)} documents.")

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
            if evaluated % 5 == 0 and final_metrics:  # Update every 5 queries
                avg_metrics = average_metrics(final_metrics, evaluated)
                mrr = avg_metrics.get('mrr', 0)
                pbar.set_postfix(evaluated=evaluated, MRR=f"{mrr:.3f}")
            else:
                pbar.set_postfix(evaluated=evaluated)
            
            pbar.update(1)

    elapsed = time.perf_counter() - start_time
    print(f"Evaluated {evaluated} queries in {elapsed:.2f}s ({evaluated / max(elapsed, 1e-6):.2f} qps)")

    if evaluated == 0:
        print("No queries evaluated (missing qrels?).")
        return

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
        print(f"\nSaved detailed results to {save_path}")


def main() -> None:
    args = parse_args()
    ks = sorted({int(value) for value in args.ks.split(",") if value.strip()})
    config = load_config(args.config)
    evaluate(args.dataset_root, config, ks, args.max_queries, args.save_results)


if __name__ == "__main__":
    main()
