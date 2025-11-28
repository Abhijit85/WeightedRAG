#!/usr/bin/env python3
"""Enhanced test to demonstrate BM25 effectiveness in weighted reranking."""

import numpy as np
from src.weighted_rag.config import RetrievalConfig, IndexStageConfig
from src.weighted_rag.index.multi_index import MultiStageVectorIndex
from src.weighted_rag.retrieval.weighted import WeightedRetriever
from src.weighted_rag.types import Chunk, Query

def create_test_chunks():
    """Create test chunks with overlapping terms to better test BM25."""
    chunks = [
        Chunk(
            chunk_id="1",
            doc_id="geography",
            text="Paris is the capital city of France. The city is famous for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
            start_char=0, end_char=130, token_count=25,
            metadata={"reliability": "0.9", "structure_score": "0.8", "domain_score": "0.7"}
        ),
        Chunk(
            chunk_id="2", 
            doc_id="programming",
            text="Python programming language is widely used for machine learning and data science applications. Many developers choose Python for its simplicity.",
            start_char=0, end_char=140, token_count=22,
            metadata={"reliability": "0.8", "structure_score": "0.7", "domain_score": "0.9"}
        ),
        Chunk(
            chunk_id="3",
            doc_id="ai",
            text="Machine learning algorithms can be implemented in Python. Popular libraries include TensorFlow, PyTorch, and scikit-learn for various ML tasks.",
            start_char=0, end_char=142, token_count=21,
            metadata={"reliability": "0.95", "structure_score": "0.9", "domain_score": "0.8"}
        ),
        Chunk(
            chunk_id="4",
            doc_id="history",
            text="The Eiffel Tower in Paris was constructed for the 1889 World's Fair. It has become an iconic symbol of France and attracts millions of visitors.",
            start_char=0, end_char=140, token_count=26,
            metadata={"reliability": "0.85", "structure_score": "0.6", "domain_score": "0.7"}
        ),
        Chunk(
            chunk_id="5",
            doc_id="travel",
            text="France is a popular tourist destination. Paris offers many attractions including museums, cafes, and historic architecture throughout the city.",
            start_char=0, end_char=135, token_count=21,
            metadata={"reliability": "0.7", "structure_score": "0.5", "domain_score": "0.6"}
        )
    ]
    return chunks

def test_queries():
    """Test multiple queries to show BM25 effectiveness."""
    
    # Create configuration with balanced weights for BM25
    config = RetrievalConfig(
        lambda_similarity=0.3,   
        alpha_reliability=0.2,
        beta_temporal=0.1,
        gamma_domain=0.1,
        delta_structure=0.1,
        epsilon_bm25=0.2         # Significant BM25 contribution
    )
    
    # Create index and retriever
    index = MultiStageVectorIndex(config)
    retriever = WeightedRetriever(config, index)
    
    # Create test chunks
    chunks = create_test_chunks()
    
    # Create dummy embeddings for chunks
    dummy_embeddings = {
        "coarse": type('obj', (object,), {
            'ids': [chunk.chunk_id for chunk in chunks],
            'vectors': np.random.randn(len(chunks), 384).astype(np.float32)
        })()
    }
    
    # Add chunks to index
    index.add_chunks(chunks, dummy_embeddings)
    
    print(f"Loaded {len(chunks)} chunks for testing")
    print("=" * 80)
    
    # Test queries
    test_cases = [
        {
            "query": "What is the capital of France?",
            "expected_top": ["1", "4", "5"],  # Paris-related chunks
            "description": "Geographic query - should favor Paris chunks via BM25"
        },
        {
            "query": "Python machine learning programming",
            "expected_top": ["2", "3"],  # Python/ML chunks
            "description": "Technical query - should favor Python/ML chunks"
        },
        {
            "query": "Eiffel Tower Paris attractions",
            "expected_top": ["4", "1", "5"],  # Tourism/Paris chunks
            "description": "Tourism query - should favor Eiffel Tower and Paris"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 60)
        
        query = Query(text=test_case['query'], query_id=f"test_{i}")
        
        # Create mock stage results with random similarities
        stage_results = {
            "coarse": type('obj', (object,), {
                'ids': [chunk.chunk_id for chunk in chunks],
                'distances': np.random.uniform(0.4, 0.9, len(chunks))
            })()
        }
        
        # Test individual BM25 scores
        print("Individual BM25 Scores:")
        bm25_scores = []
        for chunk in chunks:
            score = retriever._compute_bm25_score(query, chunk)
            bm25_scores.append((chunk.chunk_id, score))
            print(f"  ID {chunk.chunk_id}: {score:.3f} - {chunk.text[:60]}...")
        
        print()
        
        # Test weighted retrieval
        result = retriever.retrieve(query, stage_results)
        
        print("Final Weighted Rankings:")
        for rank, retrieved_chunk in enumerate(result.chunks, 1):
            chunk_id = retrieved_chunk.chunk.chunk_id
            final_score = retrieved_chunk.similarity
            bm25_score = retrieved_chunk.scores.get('bm25', 0)
            base_score = retrieved_chunk.scores.get('combined', 0)
            
            # Check if this chunk was expected to be top-ranked
            expected = "⭐" if chunk_id in test_case.get('expected_top', [])[:3] else "  "
            
            print(f"  {expected} Rank {rank}: ID {chunk_id} (Final: {final_score:.3f})")
            print(f"      BM25: {bm25_score:.3f}, Base: {base_score:.3f}")
            print(f"      Text: {retrieved_chunk.chunk.text[:70]}...")
            
            # Show score breakdown
            scores = retrieved_chunk.scores
            print(f"      Breakdown - Reliability: {scores.get('alpha_reliability', 0):.3f}, "
                  f"Domain: {scores.get('gamma_domain', 0):.3f}, "
                  f"Structure: {scores.get('delta_structure', 0):.3f}")
            print()
    
    print("=" * 80)
    print("✅ BM25 Enhanced Weighted Retrieval Test Complete!")
    print(f"Configuration: λ_sim={config.lambda_similarity}, ε_bm25={config.epsilon_bm25}")

if __name__ == "__main__":
    test_queries()