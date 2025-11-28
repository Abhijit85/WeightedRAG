#!/usr/bin/env python3
"""
Test script to validate the BM25 fixes for negative scores and better smoothing.
"""

import sys
import os
sys.path.append('/Users/atharvaguprta/Desktop/WeightedRAGAbhi')

from src.weighted_rag.retrieval.weighted import BM25Scorer
from collections import Counter
import math

def test_bm25_fixes():
    """Test the BM25 fixes for edge cases"""
    print("Testing BM25 Fixes")
    print("=" * 50)
    
    # Test documents
    documents = [
        "data science machine learning artificial intelligence",
        "python programming web development",
        "natural language processing NLP text analysis",
        "database SQL queries optimization",
        "cloud computing AWS Azure"
    ]
    
    # Initialize BM25 scorer
    scorer = BM25Scorer()
    scorer.fit(documents)
    
    print(f"Total documents: {scorer.N}")
    print(f"Average document length: {scorer.avgdl:.2f}")
    print(f"Vocabulary size: {len(scorer.doc_freqs)}")
    print()
    
    # Test queries with various characteristics
    test_queries = [
        "machine learning learning learning",  # Repeated terms
        "data science",                        # Common terms
        "quantum computing blockchain",        # OOV terms
        "python",                             # Single term
        "artificial intelligence AI ML"       # Mix of seen/unseen terms
    ]
    
    print("Testing Query Scoring:")
    print("-" * 30)
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: '{query}'")
        
        # Tokenize query
        query_tokens = scorer.tokenize(query)
        query_term_freqs = dict(Counter(query_tokens))
        print(f"Query tokens: {query_tokens}")
        print(f"Query term frequencies: {query_term_freqs}")
        
        # Score each document
        for doc_idx, doc in enumerate(documents):
            raw_score = scorer.score(query, doc)
            normalized_score = scorer.normalized_score(query, doc)
            
            print(f"  Doc {doc_idx}: raw={raw_score:.4f}, norm={normalized_score:.4f} | '{doc[:30]}{'...' if len(doc) > 30 else ''}')")
    
    print("\n" + "="*50)
    print("Testing IDF Cache Values:")
    print("-" * 30)
    
    # Show IDF values for common terms
    test_terms = ["machine", "learning", "python", "data", "science"]
    for term in test_terms:
        if term in scorer.idf_cache:
            df = scorer.doc_freqs.get(term, 0)
            idf = scorer.idf_cache[term]
            print(f"'{term}': df={df}, idf={idf:.4f}")
        else:
            print(f"'{term}': not in vocabulary")
    
    print("\n" + "="*50)
    print("Testing Edge Cases:")
    print("-" * 30)
    
    # Test empty document
    try:
        empty_doc_score = scorer.score("test query", "")
        print(f"Empty document score: {empty_doc_score:.4f}")
    except Exception as e:
        print(f"Empty document error: {e}")
    
    # Test empty query
    try:
        empty_query_score = scorer.score("", "test document with words")
        print(f"Empty query score: {empty_query_score:.4f}")
    except Exception as e:
        print(f"Empty query error: {e}")
    
    # Test very repeated terms query
    repeated_score = scorer.score("machine learning machine learning machine learning", "machine learning artificial intelligence")
    print(f"Very repeated terms score: {repeated_score:.4f}")
    
    print("\n" + "="*50)
    print("Testing Score Normalization Range:")
    print("-" * 30)
    
    # Test various queries for score range
    test_score_queries = [
        ("completely unrelated terms", "data science machine learning"),
        ("exact match", "data science machine learning"),  
        ("partial match", "data artificial"),
        ("single word", "machine"),
        ("repeated query", "machine machine machine machine")
    ]
    
    test_doc = "data science machine learning artificial intelligence"
    for desc, query in test_score_queries:
        raw_score = scorer.score(query, test_doc)
        norm_score = scorer.normalized_score(query, test_doc)
        print(f"{desc:20}: Raw={raw_score:6.4f}, Norm={norm_score:.4f}")

if __name__ == "__main__":
    test_bm25_fixes()