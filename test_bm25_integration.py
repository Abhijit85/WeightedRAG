#!/usr/bin/env python3
"""Test script to verify BM25 integration with weighted reranking."""

import numpy as np
from src.weighted_rag.config import RetrievalConfig, IndexStageConfig
from src.weighted_rag.index.multi_index import MultiStageVectorIndex
from src.weighted_rag.retrieval.weighted import WeightedRetriever
from src.weighted_rag.types import Chunk, Query

def create_test_chunks():
    """Create test chunks with different content types."""
    chunks = [
        Chunk(
            chunk_id="1",
            doc_id="doc1",
            text="The capital of France is Paris. It is known for the Eiffel Tower.",
            start_char=0,
            end_char=65,
            token_count=15,
            metadata={"source": "geography", "reliability": "0.9", "structure_score": "0.8"}
        ),
        Chunk(
            chunk_id="2", 
            doc_id="doc2",
            text="Python is a programming language. It is widely used for data science.",
            start_char=0,
            end_char=70,
            token_count=14,
            metadata={"source": "programming", "reliability": "0.8", "structure_score": "0.7"}
        ),
        Chunk(
            chunk_id="3",
            doc_id="doc3", 
            text="Machine learning involves training algorithms on data to make predictions.",
            start_char=0,
            end_char=75,
            token_count=12,
            metadata={"source": "ai", "reliability": "0.95", "structure_score": "0.9"}
        ),
        Chunk(
            chunk_id="4",
            doc_id="doc4",
            text="The Eiffel Tower was built in Paris for the 1889 World's Fair.",
            start_char=0,
            end_char=64,
            token_count=13,
            metadata={"source": "history", "reliability": "0.85", "structure_score": "0.6"}
        )
    ]
    return chunks

def test_bm25_scoring():
    """Test BM25 scoring integration."""
    print("Testing BM25 Integration with Weighted Reranking...")
    
    # Create configuration with BM25 weight
    config = RetrievalConfig(
        stages=[IndexStageConfig(
            name="test_stage",
            dimension=384,
            top_k=10,
            weight=1.0,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )],
        lambda_similarity=0.4,   # Reduced to make room for BM25
        alpha_reliability=0.2,
        beta_temporal=0.1,
        gamma_domain=0.1,
        delta_structure=0.1,
        epsilon_bm25=0.1         # BM25 contribution
    )
    
    # Create index and retriever
    index = MultiStageVectorIndex(config)
    retriever = WeightedRetriever(config, index)
    
    # Create test chunks
    chunks = create_test_chunks()
    
    # Create dummy embeddings for chunks
    dummy_embeddings = {
        "test_stage": type('obj', (object,), {
            'ids': [chunk.chunk_id for chunk in chunks],
            'vectors': np.random.randn(len(chunks), 384).astype(np.float32)
        })()
    }
    
    # Add chunks to index
    index.add_chunks(chunks, dummy_embeddings)
    
    print(f"Added {len(chunks)} chunks to index")
    
    # Test BM25 scorer directly
    print("\nTesting BM25 scorer...")
    bm25 = retriever.bm25_scorer
    
    # Fit BM25 on corpus
    documents = [chunk.text for chunk in chunks]
    bm25.fit(documents)
    print(f"Fitted BM25 on {len(documents)} documents")
    
    # Test query
    query_text = "What is the capital of France?"
    query = Query(text=query_text, query_id="test_query")
    
    # Test BM25 scores for each chunk
    print(f"\nQuery: '{query_text}'")
    print("BM25 Scores:")
    for chunk in chunks:
        score = bm25.score(query_text, chunk.text)
        print(f"  {chunk.chunk_id}: {score:.4f} - {chunk.text[:50]}...")
    
    # Test weighted retrieval (with dummy stage results)
    dummy_stage_results = {
        "test_stage": type('obj', (object,), {
            'ids': [chunk.chunk_id for chunk in chunks],
            'distances': np.array([0.8, 0.6, 0.7, 0.9])  # Mock similarity scores
        })()
    }
    
    print("\nTesting weighted retrieval with BM25...")
    try:
        result = retriever.retrieve(query, dummy_stage_results)
        
        print(f"Retrieved {len(result.chunks)} chunks:")
        for i, retrieved_chunk in enumerate(result.chunks):
            print(f"  Rank {i+1}: ID={retrieved_chunk.chunk.chunk_id}, "
                  f"Score={retrieved_chunk.similarity:.4f}")
            print(f"    BM25: {retrieved_chunk.scores.get('bm25', 0):.4f}, "
                  f"Combined: {retrieved_chunk.scores.get('combined', 0):.4f}")
            print(f"    Text: {retrieved_chunk.chunk.text[:80]}...")
            print()
        
        print("✅ BM25 integration test passed!")
        
    except Exception as e:
        print(f"❌ Error in weighted retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bm25_scoring()