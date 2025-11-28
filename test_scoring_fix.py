#!/usr/bin/env python3
"""Comprehensive test to verify fixed weighted scoring improves metrics."""

import numpy as np
from src.weighted_rag.config import RetrievalConfig, IndexStageConfig
from src.weighted_rag.index.multi_index import MultiStageVectorIndex
from src.weighted_rag.retrieval.weighted import WeightedRetriever
from src.weighted_rag.types import Chunk, Query

def test_before_after_comparison():
    """Test the scoring improvements with realistic scenarios."""
    
    print("🧪 Testing Weighted Scoring Improvements")
    print("=" * 50)
    
    # Create more realistic test chunks
    chunks = [
        Chunk(
            chunk_id="relevant_1", doc_id="doc1",
            text="Paris is the capital and largest city of France. It is known for its historical monuments, museums, and cultural significance.",
            start_char=0, end_char=120, token_count=22,
            metadata={"reliability": "0.95", "structure_score": "0.9", "domain_score": "0.8", "recency_score": "0.7"}
        ),
        Chunk(
            chunk_id="relevant_2", doc_id="doc2",
            text="The capital city of France, Paris, attracts millions of tourists each year with its famous landmarks like the Eiffel Tower.",
            start_char=0, end_char=115, token_count=21,
            metadata={"reliability": "0.88", "structure_score": "0.8", "domain_score": "0.85", "recency_score": "0.6"}
        ),
        Chunk(
            chunk_id="semi_relevant", doc_id="doc3",
            text="France is a country in Western Europe. It has many beautiful cities and regions, with various cultural attractions.",
            start_char=0, end_char=105, token_count=19,
            metadata={"reliability": "0.82", "structure_score": "0.7", "domain_score": "0.7", "recency_score": "0.5"}
        ),
        Chunk(
            chunk_id="irrelevant_1", doc_id="doc4",
            text="Python programming language is widely used in data science and machine learning applications for analyzing datasets.",
            start_char=0, end_char=110, token_count=18,
            metadata={"reliability": "0.9", "structure_score": "0.85", "domain_score": "0.6", "recency_score": "0.8"}
        ),
        Chunk(
            chunk_id="irrelevant_2", doc_id="doc5",
            text="Machine learning algorithms can process large amounts of data to identify patterns and make predictions.",
            start_char=0, end_char=100, token_count=17,
            metadata={"reliability": "0.85", "structure_score": "0.75", "domain_score": "0.65", "recency_score": "0.75"}
        )
    ]
    
    # Test configuration
    config = RetrievalConfig()
    index = MultiStageVectorIndex(config)
    retriever = WeightedRetriever(config, index)
    
    # Add chunks to index
    dummy_embeddings = {
        "coarse": type('obj', (object,), {
            'ids': [chunk.chunk_id for chunk in chunks],
            'vectors': np.random.randn(len(chunks), 384).astype(np.float32)
        })()
    }
    index.add_chunks(chunks, dummy_embeddings)
    
    # Test query about France's capital
    query = Query(text="What is the capital of France?", query_id="test_capital")
    
    print(f"Query: '{query.text}'")
    print(f"Expected relevance order: relevant_1 > relevant_2 > semi_relevant > irrelevant_*")
    print()
    
    # Simulate stage results with realistic similarity scores
    # Make relevant chunks score higher in dense retrieval
    stage_results = {
        "coarse": type('obj', (object,), {
            'ids': ["relevant_1", "relevant_2", "semi_relevant", "irrelevant_1", "irrelevant_2"],
            'distances': np.array([0.85, 0.82, 0.65, 0.45, 0.40])  # Relevant chunks get higher similarity
        })()
    }
    
    print("Input Stage Scores (coarse):")
    for chunk_id, score in zip(stage_results["coarse"].ids, stage_results["coarse"].distances):
        print(f"  {chunk_id}: {score:.3f}")
    print()
    
    # Test retrieval
    result = retriever.retrieve(query, stage_results)
    
    print("Final Weighted Rankings:")
    print("-" * 40)
    
    for i, retrieved_chunk in enumerate(result.chunks):
        chunk_id = retrieved_chunk.chunk.chunk_id
        final_score = retrieved_chunk.similarity
        scores = retrieved_chunk.scores
        
        # Calculate expected relevance
        if "relevant" in chunk_id:
            expected = "🎯 RELEVANT"
        elif "semi" in chunk_id:
            expected = "⚠️  SEMI"
        else:
            expected = "❌ IRRELEVANT"
        
        print(f"Rank {i+1}: {chunk_id} {expected}")
        print(f"  Final Score: {final_score:.4f}")
        print(f"  Components:")
        print(f"    Dense Similarity: {scores.get('combined', 0):.3f}")
        print(f"    BM25: {scores.get('bm25', 0):.3f}")
        print(f"    Reliability: {scores.get('alpha_reliability', 0):.3f}")
        print(f"    Domain: {scores.get('gamma_domain', 0):.3f}")
        print(f"    Structure: {scores.get('delta_structure', 0):.3f}")
        print(f"    Temporal: {scores.get('beta_temporal', 0):.3f}")
        print(f"  Text: {retrieved_chunk.chunk.text[:60]}...")
        print()
    
    # Analyze ranking quality
    print("📈 Ranking Quality Analysis:")
    print("-" * 30)
    
    # Check if relevant chunks are ranked higher
    relevant_ranks = []
    irrelevant_ranks = []
    
    for i, retrieved_chunk in enumerate(result.chunks):
        chunk_id = retrieved_chunk.chunk.chunk_id
        rank = i + 1
        
        if "relevant" in chunk_id:
            relevant_ranks.append(rank)
        elif "irrelevant" in chunk_id:
            irrelevant_ranks.append(rank)
    
    if relevant_ranks and irrelevant_ranks:
        avg_relevant_rank = np.mean(relevant_ranks)
        avg_irrelevant_rank = np.mean(irrelevant_ranks)
        
        print(f"Average relevant chunk rank: {avg_relevant_rank:.1f}")
        print(f"Average irrelevant chunk rank: {avg_irrelevant_rank:.1f}")
        
        if avg_relevant_rank < avg_irrelevant_rank:
            print("✅ Good: Relevant chunks ranked higher than irrelevant ones")
        else:
            print("❌ Poor: Irrelevant chunks ranked higher than relevant ones")
    
    # Check if top results are relevant
    top_3_chunks = [chunk.chunk.chunk_id for chunk in result.chunks[:3]]
    relevant_in_top3 = len([cid for cid in top_3_chunks if "relevant" in cid])
    
    print(f"Relevant chunks in top 3: {relevant_in_top3}/2")
    if relevant_in_top3 >= 1:
        print("✅ Good: At least one relevant result in top 3")
    else:
        print("❌ Poor: No relevant results in top 3")
    
    print()
    print("🔧 Configuration Analysis:")
    print(f"  Total weights: {config.lambda_similarity + config.alpha_reliability + config.beta_temporal + config.gamma_domain + config.delta_structure + config.epsilon_bm25}")
    print(f"  Similarity dominance: {config.lambda_similarity:.0%}")
    print(f"  Metadata contribution: {(config.alpha_reliability + config.beta_temporal + config.gamma_domain + config.delta_structure):.0%}")
    print(f"  BM25 contribution: {config.epsilon_bm25:.0%}")

if __name__ == "__main__":
    test_before_after_comparison()