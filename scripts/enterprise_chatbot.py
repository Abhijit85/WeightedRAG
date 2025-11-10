#!/usr/bin/env python3
"""Enterprise-flavored CLI that mirrors the reference chatbot's workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from weighted_rag import WeightedRAGPipeline
from weighted_rag.config import PipelineConfig, pipeline_config_from_dict
from weighted_rag.data.pdf_loader import PDFIngestionConfig, load_pdf_documents
from weighted_rag.types import Query, RetrievedChunk
from weighted_rag.generation.textgrad_optimizer import (
    TextGradPromptOptimizer,
    TextGradSettings,
    load_textgrad_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enterprise-style PDF ingestion and querying.")
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory containing enterprise PDFs.")
    parser.add_argument("--question", type=str, required=True, help="User question to answer.")
    parser.add_argument("--config", type=Path, help="Optional JSON pipeline config.")
    parser.add_argument("--locations", type=str, help="Comma separated list of location filters.")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="FIELD=VALUE1,VALUE2",
        help="Additional metadata filters (matches chunk metadata fields).",
    )
    parser.add_argument("--show-chunks", type=int, default=5, help="How many retrieved chunks to display.")
    parser.add_argument("--ingest-chunk-size", type=int, default=2000, help="PDF splitter chunk size.")
    parser.add_argument("--ingest-overlap", type=int, default=500, help="PDF splitter overlap.")
    parser.add_argument("--save-chunks", type=Path, help="Optional path to dump retrieved chunk metadata as JSON.")
    parser.add_argument("--textgrad-examples", type=Path, help="JSON/JSONL file with TextGrad supervision examples.")
    parser.add_argument("--textgrad-history", type=Path, help="Optional path to save TextGrad optimization trace.")
    parser.add_argument("--textgrad-steps", type=int, default=3, help="Max optimization steps.")
    parser.add_argument("--textgrad-min-gain", type=float, default=0.01, help="Minimum mean F1 gain to accept mutation.")
    parser.add_argument("--textgrad-sample-size", type=int, help="Optional number of examples to subsample per step.")
    parser.add_argument("--textgrad-mutations", type=int, default=4, help="Candidate instruction mutations per step.")
    return parser.parse_args()


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    payload: Dict[str, Any] = json.loads(path.read_text())
    return pipeline_config_from_dict(payload)


def build_query_metadata(args: argparse.Namespace) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if args.locations:
        values = [value.strip() for value in args.locations.split(",") if value.strip()]
        if values:
            metadata["filter__locations"] = json.dumps(values)
    for clause in args.filter:
        key, _, value = clause.partition("=")
        key = key.strip()
        if not key or not value:
            continue
        values = [entry.strip() for entry in value.split(",") if entry.strip()]
        if values:
            metadata[f"filter__{key}"] = json.dumps(values)
    return metadata


def print_chunks(chunks: List[RetrievedChunk], limit: int) -> None:
    print("\n--- Retrieved Chunks ---")
    for idx, item in enumerate(chunks[:limit], start=1):
        meta = item.chunk.metadata
        title = meta.get("title", item.chunk.doc_id)
        score = item.similarity
        print(f"Rank {idx} | Title: {title} | Score: {score:.4f}")
        print(f"Type: {meta.get('document_type', 'n/a')} | Dept: {meta.get('department', 'n/a')} | Confidentiality: {meta.get('confidentiality_level', 'n/a')}")
        preview = item.chunk.text.replace("\n", " ")
        print(f"Content Preview: {preview[:200]}...\n")


def save_chunks(path: Path, chunks: List[RetrievedChunk]) -> None:
    serialized = [
        {
            "chunk_id": item.chunk.chunk_id,
            "doc_id": item.chunk.doc_id,
            "score": item.similarity,
            "metadata": item.chunk.metadata,
            "text": item.chunk.text,
        }
        for item in chunks
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialized, indent=2))
    print(f"Saved retrieved chunk details to {path}")


def maybe_run_textgrad(pipeline: WeightedRAGPipeline, args: argparse.Namespace) -> None:
    if not args.textgrad_examples:
        return
    examples = load_textgrad_examples(args.textgrad_examples)
    if not examples:
        print("No valid TextGrad examples found; skipping prompt optimization.")
        return
    settings = TextGradSettings(
        max_steps=args.textgrad_steps,
        min_gain=args.textgrad_min_gain,
        sample_size=args.textgrad_sample_size,
        mutations_per_step=args.textgrad_mutations,
    )
    optimizer = TextGradPromptOptimizer(pipeline, settings)
    print(f"Running TextGrad optimization over {len(examples)} examples...")
    new_prompt, history = optimizer.optimize(examples, pipeline.config.generation.system_prompt)
    pipeline.generator.update_system_prompt(new_prompt)
    pipeline.config.generation.system_prompt = new_prompt
    if args.textgrad_history:
        args.textgrad_history.parent.mkdir(parents=True, exist_ok=True)
        args.textgrad_history.write_text(json.dumps(history, indent=2))
        print(f"Saved TextGrad history to {args.textgrad_history}")
    final_score = history[-1]["score"] if history else 0.0
    print(f"TextGrad complete. Final prompt score: {final_score:.4f}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ingestion_config = PDFIngestionConfig(chunk_size=args.ingest_chunk_size, chunk_overlap=args.ingest_overlap)
    try:
        documents = load_pdf_documents(args.pdf_dir, ingestion_config)
    except ImportError as exc:
        raise SystemExit(str(exc))

    pipeline = WeightedRAGPipeline(config)
    total_chunks = pipeline.add_documents(documents)
    print(f"Indexed {total_chunks} chunks from {len(documents)} PDF segments.")

    maybe_run_textgrad(pipeline, args)

    metadata = build_query_metadata(args)
    query = Query(query_id="enterprise-query", text=args.question, metadata=metadata)
    retrieval, _ = pipeline.retrieve_with_details(query)

    print_chunks(list(retrieval.chunks), args.show_chunks)

    result = pipeline.generator.generate(retrieval)
    print("\nAnswer:\n")
    print(result.answer)
    print("\nReferences:", result.references)

    if args.save_chunks:
        save_chunks(args.save_chunks, list(retrieval.chunks))


if __name__ == "__main__":
    main()
