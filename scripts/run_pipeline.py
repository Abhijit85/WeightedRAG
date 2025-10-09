#!/usr/bin/env python3
"""Command-line entry point for the WeightedRAG pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from weighted_rag import WeightedRAGPipeline
from weighted_rag.config import PipelineConfig, pipeline_config_from_dict
from weighted_rag.types import Query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WeightedRAG pipeline end-to-end.")
    parser.add_argument("--source", type=Path, required=True, help="Directory or JSONL file with documents.")
    parser.add_argument("--question", type=str, required=True, help="Question to answer.")
    parser.add_argument("--config", type=Path, help="Optional JSON config path.")
    parser.add_argument("--metadata", type=str, help="JSON string with global metadata to attach to documents.")
    return parser.parse_args()


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    payload: Dict[str, Any] = json.loads(path.read_text())
    return pipeline_config_from_dict(payload)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline = WeightedRAGPipeline(config)
    metadata = json.loads(args.metadata) if args.metadata else {}
    documents = pipeline.ingest_path(args.source, metadata=metadata)
    count = pipeline.add_documents(documents)
    print(f"Indexed {count} chunks from {len(documents)} documents.")
    query = Query(query_id="query-0", text=args.question)
    result = pipeline.answer(query)
    print("\nAnswer:", result.answer)
    print("\nReferences:", result.references)


if __name__ == "__main__":
    main()
