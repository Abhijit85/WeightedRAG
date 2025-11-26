#!/usr/bin/env python3
"""Utility to convert NQ-table chunk outputs into a BEIR-style split."""

from __future__ import annotations

import argparse
import json
import shutil
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


def filter_dataset_for_chunks(output_dir: Path, chunk_types: str) -> Dict[str, int]:
    """Filter BEIR dataset to remove queries that don't have corresponding chunks.
    
    Args:
        output_dir: Directory containing the BEIR files
        chunk_types: Comma-separated chunk types that were included
    
    Returns:
        Dictionary with filtering statistics
    """
    print("\n🔍 Filtering dataset to remove queries without table chunks...")
    
    # Paths
    queries_file = output_dir / "queries.jsonl"
    qrels_file = output_dir / "qrels.tsv"
    corpus_file = output_dir / "corpus.jsonl"
    
    # Get valid table IDs from corpus
    valid_table_ids = set()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            corpus_item = json.loads(line.strip())
            corpus_id = corpus_item['_id']
            # Extract table_id from chunk_id (format: table_id_chunk_type)
            if 'table_id' in corpus_item:
                table_id = corpus_item['table_id']
                valid_table_ids.add(table_id)
    
    print(f"✅ Found {len(valid_table_ids)} tables with chunks")
    
    # Filter queries
    valid_queries = []
    total_queries = 0
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_data = json.loads(line.strip())
            total_queries += 1
            query_id = query_data['_id']
            
            if query_id in valid_table_ids:
                valid_queries.append(query_data)
    
    # Filter qrels
    valid_qrels = []
    valid_query_ids = {q['_id'] for q in valid_queries}
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                query_id = parts[0]
                if query_id in valid_query_ids:
                    valid_qrels.append(line.strip())
    
    # Filter corpus
    valid_corpus = []
    expected_corpus_ids = set()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            corpus_item = json.loads(line.strip())
            corpus_id = corpus_item['_id']
            # Only keep chunks whose table_id has a corresponding query
            table_id = corpus_item.get('table_id', '')
            if table_id in valid_query_ids:
                valid_corpus.append(corpus_item)
                expected_corpus_ids.add(corpus_id)
    
    # Create filtered files
    filtered_dir = output_dir / "filtered"
    filtered_dir.mkdir(exist_ok=True)
    
    with open(filtered_dir / "queries.jsonl", 'w', encoding='utf-8') as f:
        for query in valid_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    with open(filtered_dir / "qrels.tsv", 'w', encoding='utf-8') as f:
        for qrel in valid_qrels:
            f.write(qrel + '\n')
    
    with open(filtered_dir / "corpus.jsonl", 'w', encoding='utf-8') as f:
        for item in valid_corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Replace original files with filtered ones
    shutil.move(str(filtered_dir / "queries.jsonl"), str(queries_file))
    shutil.move(str(filtered_dir / "qrels.tsv"), str(qrels_file))
    shutil.move(str(filtered_dir / "corpus.jsonl"), str(corpus_file))
    
    # Remove temporary directory
    filtered_dir.rmdir()
    
    stats = {
        'original_queries': total_queries,
        'filtered_queries': len(valid_queries),
        'removed_queries': total_queries - len(valid_queries),
        'retention_rate': len(valid_queries) / total_queries if total_queries > 0 else 0
    }
    
    print(f"📊 Filtering Results:")
    print(f"  Original queries: {stats['original_queries']}")
    print(f"  Filtered queries: {stats['filtered_queries']}")
    print(f"  Removed queries: {stats['removed_queries']}")
    print(f"  Retention rate: {stats['retention_rate']*100:.1f}%")
    
    return stats


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
    
    # Automatically filter the dataset to remove queries without table chunks
    filter_stats = filter_dataset_for_chunks(args.output_dir, args.chunk_types)
    
    print(f"\n✅ Final dataset: {filter_stats['filtered_queries']} queries with table chunks")
    if filter_stats['removed_queries'] > 0:
        print(f"🗑️  Removed {filter_stats['removed_queries']} queries without table chunks")


if __name__ == "__main__":
    main()
