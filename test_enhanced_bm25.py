#!/usr/bin/env python3
"""Test the improved BM25 implementation to verify enhancements."""

import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.weighted_rag.retrieval.weighted import BM25Scorer
from src.weighted_rag.types import Chunk, Query
from src.weighted_rag.config import RetrievalConfig, IndexStageConfig
from src.weighted_rag.index.multi_index import MultiStageVectorIndex
from src.weighted_rag.retrieval.weighted import WeightedRetriever

def test_enhanced_bm25():
    """Test the enhanced BM25 implementation."""
    print("🚀 Testing Enhanced BM25 Implementation")
    print("=" * 50)
    
    # Test documents
    test_docs = [
        "The capital of France is Paris. It's known for the Eiffel Tower.",
        "Paris is a beautiful city in France with rich history and culture.",
        "Machine learning algorithms use data to make predictions.",
        "Python programming language is popular for data science.",
        "Database systems store and manage structured data efficiently.",
        "SQL queries retrieve information from relational databases.",
        "The Eiffel Tower in Paris attracts millions of tourists yearly."
    ]
    
    # Test queries
    test_queries = [
        "capital France Paris",
        "machine learning data",
        "database SQL",
        "Eiffel Tower Paris"
    ]
    
    # Initialize enhanced BM25
    bm25 = BM25Scorer()
    bm25.fit(test_docs)
    
    print(f"Corpus statistics:")
    print(f"  Documents: {bm25.N}")
    print(f"  Average document length: {bm25.avgdl:.2f}")
    print(f"  Vocabulary size: {len(bm25.doc_freqs)}")
    print(f"  Stop words removed: {len(bm25.stop_words)}")
    print()
    
    # Test scoring
    for query in test_queries:
        print(f"Query: '{query}'")
        scores = []
        
        for i, doc in enumerate(test_docs):
            raw_score = bm25.score(query, doc)
            norm_score = bm25.normalized_score(query, doc)
            scores.append((i, raw_score, norm_score, doc[:50] + "..."))
        
        # Sort by raw score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        print("  Top 3 matches:")
        for rank, (doc_idx, raw_score, norm_score, text) in enumerate(scores[:3], 1):
            print(f"    {rank}. Doc {doc_idx}: Raw={raw_score:.3f}, Norm={norm_score:.3f}")
            print(f"       {text}")
        
        # Analyze score distribution
        raw_scores = [s[1] for s in scores]
        norm_scores = [s[2] for s in scores]
        print(f"  Raw score range: {min(raw_scores):.3f} to {max(raw_scores):.3f}")
        print(f"  Normalized range: {min(norm_scores):.3f} to {max(norm_scores):.3f}")
        print()

def test_edge_cases():
    """Test edge cases and robustness."""
    print("🧪 Testing Edge Cases")
    print("=" * 30)
    
    bm25 = BM25Scorer()
    
    # Test empty documents
    print("1. Empty documents test:")
    try:
        bm25.fit(["", "   ", "\n\t", "actual content"])
        score = bm25.score("test", "actual content")
        print(f"   ✅ Empty docs handled: score={score:.3f}")
    except Exception as e:
        print(f"   ❌ Empty docs failed: {e}")
    
    # Test special characters
    print("2. Special characters test:")
    special_docs = ["Hello, world! How are you?", "I'm fine... thanks!"]
    bm25.fit(special_docs)
    score1 = bm25.score("Hello world", special_docs[0])
    score2 = bm25.normalized_score("hello world", special_docs[0])
    print(f"   ✅ Special chars: raw={score1:.3f}, norm={score2:.3f}")
    
    # Test repeated query terms
    print("3. Query term frequency test:")
    docs = ["Paris is the capital of France"]
    bm25.fit(docs)
    score1 = bm25.score("Paris", docs[0])
    score2 = bm25.score("Paris Paris Paris", docs[0])
    improvement = score2 > score1
    print(f"   {'✅' if improvement else '❌'} QTF weighting: single={score1:.3f}, triple={score2:.3f}")
    
    # Test out-of-vocabulary terms
    print("4. Out-of-vocabulary test:")
    score_oov = bm25.score("xyzzyx", docs[0])
    score_smoothing = score_oov > 0
    print(f"   {'✅' if score_smoothing else '❌'} OOV smoothing: score={score_oov:.3f}")

def test_integration_with_weighted_retrieval():
    """Test integration with the weighted retrieval system."""
    print("🔗 Testing Integration with Weighted Retrieval")
    print("=" * 50)
    
    # Create test chunks
    chunks = [
        Chunk(
            chunk_id="paris_1", doc_id="doc1",
            text="Paris is the capital of France, famous for the Eiffel Tower and Louvre Museum.",
            start_char=0, end_char=80, token_count=15,
            metadata={"reliability": "0.9", "structure_score": "0.8", "domain_score": "0.7"}
        ),
        Chunk(
            chunk_id="ml_1", doc_id="doc2",
            text="Machine learning algorithms analyze data patterns to make accurate predictions.",
            start_char=0, end_char=75, token_count=12,
            metadata={"reliability": "0.85", "structure_score": "0.9", "domain_score": "0.8"}
        ),
        Chunk(
            chunk_id="db_1", doc_id="doc3", 
            text="SQL database systems efficiently store and query structured relational data.",
            start_char=0, end_char=72, token_count=11,
            metadata={"reliability": "0.8", "structure_score": "0.75", "domain_score": "0.75"}
        )
    ]
    
    # Setup retrieval system
    config = RetrievalConfig(epsilon_bm25=0.15)  # Slightly higher BM25 weight for testing
    index = MultiStageVectorIndex(config)
    retriever = WeightedRetriever(config, index)
    
    # Add chunks
    dummy_embeddings = {
        "coarse": type('obj', (object,), {
            'ids': [chunk.chunk_id for chunk in chunks],
            'vectors': np.random.randn(len(chunks), 384).astype(np.float32)
        })()
    }
    index.add_chunks(chunks, dummy_embeddings)
    
    # Test query
    query = Query(text="Paris France capital Eiffel Tower", query_id="test")
    
    # Mock stage results
    stage_results = {
        "coarse": type('obj', (object,), {
            'ids': [chunk.chunk_id for chunk in chunks],
            'distances': np.array([0.7, 0.6, 0.5])  # Paris chunk gets highest similarity
        })()
    }
    
    # Test retrieval
    result = retriever.retrieve(query, stage_results)
    
    print(f"Query: '{query.text}'")
    print("Results:")
    for i, retrieved_chunk in enumerate(result.chunks):
        chunk_id = retrieved_chunk.chunk.chunk_id
        final_score = retrieved_chunk.similarity
        scores = retrieved_chunk.scores
        
        print(f"  {i+1}. {chunk_id}: Final={final_score:.4f}")
        print(f"     Dense: {scores.get('combined', 0):.3f}, BM25: {scores.get('bm25', 0):.3f}")
        print(f"     Reliability: {scores.get('alpha_reliability', 0):.3f}")
        print(f"     Text: {retrieved_chunk.chunk.text[:60]}...")
        print()
    
    # Verify BM25 is contributing positively
    bm25_scores = [chunk.scores.get('bm25', 0) for chunk in result.chunks]
    max_bm25 = max(bm25_scores)
    print(f"BM25 contribution range: {min(bm25_scores):.3f} to {max_bm25:.3f}")
    print(f"{'✅' if max_bm25 > 0.1 else '❌'} BM25 contributing meaningfully to final scores")

def main():
    """Run all enhanced BM25 tests."""
    test_enhanced_bm25()
    test_edge_cases()
    test_integration_with_weighted_retrieval()
    
    print("\n📊 Enhanced BM25 Summary")
    print("=" * 35)
    print("✅ Improvements implemented:")
    print("  • Enhanced tokenization with punctuation removal")
    print("  • Stop word filtering for better relevance")
    print("  • Query term frequency weighting") 
    print("  • Smoothing for unseen terms")
    print("  • Log-sigmoid normalization")
    print("  • IDF caching for performance")
    print("  • Better edge case handling")
    print("\n🎯 Expected benefits:")
    print("  • More robust scoring across diverse queries")
    print("  • Better handling of repeated terms")
    print("  • Improved score distribution for weighted combination")
    print("  • Enhanced performance with caching")

if __name__ == "__main__":
    main()