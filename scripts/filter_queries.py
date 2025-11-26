#!/usr/bin/env python3
"""
Filter BEIR dataset to remove queries that don't have corresponding table chunks.
This script removes non-table entries (lists, simple text) from the evaluation dataset.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def main():
    print("🔍 Filtering BEIR dataset to match available table chunks")
    print("=" * 60)
    
    # Paths
    beir_dir = Path("datasets/nq-table/beir")
    chunks_dir = Path("retrieval_tables")
    
    queries_file = beir_dir / "queries.jsonl"
    qrels_file = beir_dir / "qrels.tsv"
    corpus_file = beir_dir / "corpus.jsonl"
    
    # New filtered files
    filtered_queries_file = beir_dir / "queries_filtered.jsonl"
    filtered_qrels_file = beir_dir / "qrels_filtered.tsv"
    filtered_corpus_file = beir_dir / "corpus_filtered.jsonl"
    
    # Get example IDs that have pure table chunks
    print("📊 Reading available table chunks...")
    valid_table_ids = set()
    
    # Check which chunks exist (use pure_table as reference since it requires structured data)
    pure_table_file = chunks_dir / "chunks_pure_table.jsonl"
    if pure_table_file.exists():
        with open(pure_table_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line.strip())
                table_id = chunk['table_id']  # This is the table hash ID
                valid_table_ids.add(table_id)
    
    print(f"✅ Found {len(valid_table_ids)} entries with valid table chunks")
    
    # Read original queries and map query_id to table_id
    print("📝 Reading original queries...")
    query_to_table = {}
    valid_queries = []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_data = json.loads(line.strip())
            query_id = query_data['_id']
            
            # The query _id is the table_id
            table_id = query_id
            query_to_table[query_id] = table_id
            
            if table_id in valid_table_ids:
                valid_queries.append(query_data)
    
    print(f"✅ Filtered queries: {len(valid_queries)} out of {len(query_to_table)} original")
    
    # Write filtered queries
    print("💾 Writing filtered queries...")
    with open(filtered_queries_file, 'w', encoding='utf-8') as f:
        for query in valid_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    # Read and filter qrels (relevance judgments)
    print("📊 Reading and filtering qrels...")
    valid_query_ids = {q['_id'] for q in valid_queries}
    filtered_qrels = []
    
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                query_id, _, corpus_id, relevance = parts[0], parts[1], parts[2], parts[3]
                if query_id in valid_query_ids:
                    filtered_qrels.append(line.strip())
    
    print(f"✅ Filtered qrels: {len(filtered_qrels)} relevance judgments")
    
    # Write filtered qrels
    with open(filtered_qrels_file, 'w', encoding='utf-8') as f:
        for qrel in filtered_qrels:
            f.write(qrel + '\n')
    
    # Read and filter corpus
    print("📚 Reading and filtering corpus...")
    valid_corpus_items = []
    
    # Create set of expected corpus IDs (table_id + _pure_table)
    expected_corpus_ids = {f"{table_id}_pure_table" for table_id in valid_table_ids}
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            corpus_item = json.loads(line.strip())
            corpus_id = corpus_item['_id']
            
            # Check if this corpus item corresponds to a valid table
            if corpus_id in expected_corpus_ids:
                valid_corpus_items.append(corpus_item)
    
    print(f"✅ Filtered corpus: {len(valid_corpus_items)} documents")
    
    # Write filtered corpus
    with open(filtered_corpus_file, 'w', encoding='utf-8') as f:
        for item in valid_corpus_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Summary
    print("\n" + "="*60)
    print("📊 FILTERING SUMMARY")
    print("="*60)
    
    original_queries = len(query_to_table)
    filtered_queries_count = len(valid_queries)
    removed_count = original_queries - filtered_queries_count
    
    print(f"Original queries: {original_queries}")
    print(f"Available table chunks: {len(valid_table_ids)}")
    print(f"Filtered queries: {filtered_queries_count}")
    print(f"Removed queries: {removed_count}")
    print(f"Retention rate: {filtered_queries_count/original_queries*100:.1f}%")
    
    print("\n📁 Output files:")
    print(f"  • {filtered_queries_file}")
    print(f"  • {filtered_qrels_file}")
    print(f"  • {filtered_corpus_file}")
    
    print("\n💡 To use filtered dataset:")
    print("  python scripts/evaluate_retrieval.py --dataset-root datasets/nq-table/beir_filtered")
    
    # Create symbolic links or copy files to a new beir_filtered directory
    beir_filtered_dir = beir_dir.parent / "beir_filtered"
    beir_filtered_dir.mkdir(exist_ok=True)
    
    # Copy filtered files to new directory
    import shutil
    shutil.copy2(filtered_queries_file, beir_filtered_dir / "queries.jsonl")
    shutil.copy2(filtered_qrels_file, beir_filtered_dir / "qrels.tsv")
    shutil.copy2(filtered_corpus_file, beir_filtered_dir / "corpus.jsonl")
    
    print(f"\n✅ Created filtered dataset directory: {beir_filtered_dir}")
    
    return {
        'original_queries': original_queries,
        'filtered_queries': filtered_queries_count,
        'removed_queries': removed_count,
        'retention_rate': filtered_queries_count/original_queries
    }

if __name__ == "__main__":
    stats = main()
    print(f"\n🎉 Dataset filtering completed successfully!")
    sys.exit(0)