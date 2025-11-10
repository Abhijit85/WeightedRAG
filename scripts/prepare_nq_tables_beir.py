#!/usr/bin/env python3
"""Utility to convert NQ-table chunk outputs into a BEIR-style split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NQ-table chunks into BEIR files.")
    parser.add_argument(
        "--tables",
        type=Path,
        required=True,
        help="Path to processed_tables.jsonl produced by chunking/core/create_retrieval_tables.py",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to retrieval_chunks.jsonl (or any chunk jsonl) produced by the same script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where corpus.jsonl, queries.jsonl, and qrels.tsv will be written.",
    )
    parser.add_argument(
        "--chunk-types",
        type=str,
        default="full_table,table_only,table_row,sliding_window,table_sample,pure_table",
        help="Comma-separated subset of chunk types to keep in the BEIR corpus.",
    )
    return parser.parse_args()


def normalize_text(chunk: Dict) -> str | None:
    """Return a plaintext representation for the chunk."""
    if "content" in chunk and chunk["content"]:
        return str(chunk["content"])
    if "text" in chunk and chunk["text"]:
        return str(chunk["text"])
    if "table" in chunk and chunk["table"]:
        return json.dumps(chunk["table"])
    return None


def load_chunks(path: Path, allowed_types: set[str]) -> Tuple[List[Dict], Dict[str, List[str]]]:
    corpus_entries: List[Dict] = []
    table_to_chunks: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            chunk_type = payload.get("chunk_type", "")
            if allowed_types and chunk_type not in allowed_types:
                continue
            text = normalize_text(payload)
            if not text:
                continue
            doc_id = str(payload.get("id") or payload.get("chunk_id"))
            table_id = str(payload.get("table_id") or doc_id.split("_")[0])
            corpus_entries.append(
                {
                    "_id": doc_id,
                    "title": chunk_type or "chunk",
                    "text": text,
                    "table_id": table_id,
                }
            )
            table_to_chunks[table_id].append(doc_id)
    return corpus_entries, table_to_chunks


def load_tables(path: Path) -> List[Dict]:
    entries: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            entries.append(payload)
    return entries


def write_beir_files(
    output_dir: Path,
    corpus: List[Dict],
    tables: List[Dict],
    table_to_chunks: Dict[str, List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    queries_path = output_dir / "queries.jsonl"
    qrels_path = output_dir / "qrels.tsv"

    with corpus_path.open("w", encoding="utf-8") as corpus_handle:
        for doc in corpus:
            corpus_handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with queries_path.open("w", encoding="utf-8") as queries_handle, qrels_path.open(
        "w", encoding="utf-8"
    ) as qrels_handle:
        for entry in tables:
            table_id = str(entry.get("table_id"))
            question = entry.get("original_question") or entry.get("question") or ""
            if not question.strip():
                continue
            queries_handle.write(json.dumps({"_id": table_id, "text": question.strip()}, ensure_ascii=False) + "\n")
            relevant_chunks = table_to_chunks.get(table_id, [])
            for chunk_id in relevant_chunks:
                qrels_handle.write(f"{table_id}\t0\t{chunk_id}\t1\n")


def main() -> None:
    args = parse_args()
    allowed_types = {chunk_type.strip() for chunk_type in args.chunk_types.split(",") if chunk_type.strip()}
    corpus, table_to_chunks = load_chunks(args.chunks, allowed_types)
    tables = load_tables(args.tables)
    if not corpus:
        raise SystemExit("No chunks collected. Check --chunk-types or chunk file path.")
    if not tables:
        raise SystemExit("No tables loaded from processed_tables.jsonl.")
    write_beir_files(args.output_dir, corpus, tables, table_to_chunks)
    print(f"Wrote {len(corpus)} documents, {len(tables)} queries to {args.output_dir} in BEIR format.")


if __name__ == "__main__":
    main()
