#!/usr/bin/env python3
"""Evaluate retrieval quality on BEIR-style datasets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
        return PipelineConfig()
    payload: Dict[str, Any] = json.loads(path.read_text())
    return pipeline_config_from_dict(payload)


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
    total_chunks = pipeline.add_documents(documents)
    print(f"Indexed {total_chunks} chunks across {len(documents)} documents.")

    stage_names = [stage.name for stage in config.retrieval.stages]
    stage_metrics: Dict[str, Dict[str, float]] = {name: {} for name in stage_names}
    final_metrics: Dict[str, float] = {}
    evaluated = 0
    per_query_output: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for query_id, query in split.queries.items():
        if max_queries is not None and evaluated >= max_queries:
            break
        qrel = split.qrels.get(query_id)
        if not qrel:
            continue
        retrieval, stage_results = pipeline.retrieve_with_details(query)
        final_docs = unique_doc_sequence([item.chunk.chunk_id for item in retrieval.chunks], pipeline)
        query_metrics = compute_metrics(final_docs, qrel, ks)
        if not query_metrics:
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
