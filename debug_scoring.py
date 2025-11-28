#!/usr/bin/env python3
"""Debug script to analyze score distributions and identify scaling issues."""

import numpy as np
from src.weighted_rag.config import RetrievalConfig, IndexStageConfig
from src.weighted_rag.index.multi_index import MultiStageVectorIndex
from src.weighted_rag.retrieval.weighted import WeightedRetriever
from src.weighted_rag.types import Chunk, Query

def create_debug_chunks():
    """Create chunks that represent typical data for score analysis."""
    chunks = [
        Chunk(
            chunk_id="1", doc_id="doc1",
            text="The capital of France is Paris. The city has many museums and historical landmarks including the Eiffel Tower.",
            start_char=0, end_char=100, token_count=20,
            metadata={"reliability": "0.95", "structure_score": "0.8", "domain_score": "0.7", "recency_score": "0.6"}
        ),
        Chunk(
            chunk_id="2", doc_id="doc2", 
            text="Python programming language is used for machine learning. It has libraries like scikit-learn and TensorFlow.",
            start_char=0, end_char=95, token_count=18,
            metadata={"reliability": "0.88", "structure_score": "0.9", "domain_score": "0.85", "recency_score": "0.7"}
        ),
        Chunk(
            chunk_id="3", doc_id="doc3",
            text="Machine learning algorithms process data to make predictions. Deep learning is a subset of machine learning.",
            start_char=0, end_char=98, token_count=19,
            metadata={"reliability": "0.9", "structure_score": "0.75", "domain_score": "0.9", "recency_score": "0.8"}
        ),
        Chunk(
            chunk_id="4", doc_id="doc4",
            text="Database systems store and manage data efficiently. SQL is a language for querying relational databases.",
            start_char=0, end_char=97, token_count=17,
            metadata={"reliability": "0.82", "structure_score": "0.7", "domain_score": "0.6", "recency_score": "0.5"}
        ),
        Chunk(
            chunk_id="5", doc_id="doc5",
            text="Web development involves creating applications for the internet. HTML, CSS, and JavaScript are core technologies.",
            start_char=0, end_char=105, token_count=19,
            metadata={"reliability": "0.78", "structure_score": "0.65", "domain_score": "0.8", "recency_score": "0.9"}
        )
    ]
    return chunks

def analyze_score_distributions():
    """Analyze the distribution and scale of different scoring components."""
    print("🔍 Analyzing Score Distributions for WeightedRAG")
    print("=" * 60)
    
    # Create test setup
    config = RetrievalConfig()
    index = MultiStageVectorIndex(config)
    retriever = WeightedRetriever(config, index)
    
    chunks = create_debug_chunks()
    
    # Create dummy embeddings
    dummy_embeddings = {
        "coarse": type('obj', (object,), {
            'ids': [chunk.chunk_id for chunk in chunks],
            'vectors': np.random.randn(len(chunks), 384).astype(np.float32)
        })()
    }
    
    # Add chunks to index
    index.add_chunks(chunks, dummy_embeddings)
    
    # Test queries with different characteristics
    test_queries = [
        "What is the capital of France?",
        "Python machine learning libraries",
        "database SQL query language",
        "web development technologies"
    ]
    
    print(f"Current Configuration:")
    print(f"  λ_similarity: {config.lambda_similarity}")
    print(f"  α_reliability: {config.alpha_reliability}")  
    print(f"  β_temporal: {config.beta_temporal}")
    print(f"  γ_domain: {config.gamma_domain}")
    print(f"  δ_structure: {config.delta_structure}")
    print(f"  ε_bm25: {config.epsilon_bm25}")
    print(f"  Total weight: {config.lambda_similarity + config.alpha_reliability + config.beta_temporal + config.gamma_domain + config.delta_structure + config.epsilon_bm25}")
    print()
    
    all_similarity_scores = []
    all_bm25_scores = []
    all_metadata_scores = {
        'reliability': [],
        'temporal': [], 
        'domain': [],
        'structure': []
    }
    
    for query_text in test_queries:
        print(f"Query: '{query_text}'")
        print("-" * 40)
        
        query = Query(text=query_text, query_id=f"test_{query_text[:10]}")
        
        # Mock similarity scores (what dense retrieval would return)
        mock_similarities = np.random.uniform(0.3, 0.9, len(chunks))
        
        print("Score Components:")
        for i, chunk in enumerate(chunks):
            similarity = mock_similarities[i]
            bm25_score = retriever._compute_bm25_score(query, chunk)
            meta_scores = retriever._metadata_scores(chunk)
            
            all_similarity_scores.append(similarity)
            all_bm25_scores.append(bm25_score)
            all_metadata_scores['reliability'].append(meta_scores['alpha_reliability'])
            all_metadata_scores['temporal'].append(meta_scores['beta_temporal'])
            all_metadata_scores['domain'].append(meta_scores['gamma_domain'])
            all_metadata_scores['structure'].append(meta_scores['delta_structure'])
            
            # Calculate component contributions
            sim_contrib = config.lambda_similarity * similarity
            rel_contrib = config.alpha_reliability * meta_scores['alpha_reliability']
            temp_contrib = config.beta_temporal * meta_scores['beta_temporal']
            domain_contrib = config.gamma_domain * meta_scores['gamma_domain']
            struct_contrib = config.delta_structure * meta_scores['delta_structure']
            bm25_contrib = config.epsilon_bm25 * bm25_score
            
            total_score = sim_contrib + rel_contrib + temp_contrib + domain_contrib + struct_contrib + bm25_contrib
            
            print(f"  Chunk {chunk.chunk_id}:")
            print(f"    Similarity: {similarity:.3f} → {sim_contrib:.3f}")
            print(f"    BM25: {bm25_score:.3f} → {bm25_contrib:.3f}")
            print(f"    Reliability: {meta_scores['alpha_reliability']:.3f} → {rel_contrib:.3f}")
            print(f"    Temporal: {meta_scores['beta_temporal']:.3f} → {temp_contrib:.3f}")
            print(f"    Domain: {meta_scores['gamma_domain']:.3f} → {domain_contrib:.3f}")
            print(f"    Structure: {meta_scores['delta_structure']:.3f} → {struct_contrib:.3f}")
            print(f"    TOTAL: {total_score:.3f}")
            print()
        print()
    
    # Statistical analysis
    print("📊 Statistical Summary:")
    print("=" * 40)
    print(f"Similarity Scores: min={np.min(all_similarity_scores):.3f}, max={np.max(all_similarity_scores):.3f}, mean={np.mean(all_similarity_scores):.3f}, std={np.std(all_similarity_scores):.3f}")
    print(f"BM25 Scores: min={np.min(all_bm25_scores):.3f}, max={np.max(all_bm25_scores):.3f}, mean={np.mean(all_bm25_scores):.3f}, std={np.std(all_bm25_scores):.3f}")
    
    for name, scores in all_metadata_scores.items():
        print(f"{name.capitalize()} Scores: min={np.min(scores):.3f}, max={np.max(scores):.3f}, mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
    
    # Component contribution analysis
    print("\n🎯 Component Contribution Analysis:")
    print("=" * 50)
    sim_contribs = [config.lambda_similarity * s for s in all_similarity_scores]
    bm25_contribs = [config.epsilon_bm25 * s for s in all_bm25_scores]
    
    print(f"Similarity contributions: min={np.min(sim_contribs):.3f}, max={np.max(sim_contribs):.3f}, mean={np.mean(sim_contribs):.3f}")
    print(f"BM25 contributions: min={np.min(bm25_contribs):.3f}, max={np.max(bm25_contribs):.3f}, mean={np.mean(bm25_contribs):.3f}")
    
    # Identify potential issues
    print("\n⚠️  Potential Issues:")
    print("=" * 30)
    
    total_weight = config.lambda_similarity + config.alpha_reliability + config.beta_temporal + config.gamma_domain + config.delta_structure + config.epsilon_bm25
    if abs(total_weight - 1.0) > 0.01:
        print(f"❌ Weight sum is {total_weight:.3f} (should be close to 1.0)")
    
    if np.mean(bm25_contribs) > np.mean(sim_contribs):
        print(f"❌ BM25 contributions ({np.mean(bm25_contribs):.3f}) dominating similarity ({np.mean(sim_contribs):.3f})")
    
    if np.max(all_bm25_scores) > 10 * np.max(all_similarity_scores):
        print(f"❌ BM25 score range ({np.max(all_bm25_scores):.3f}) much larger than similarity range ({np.max(all_similarity_scores):.3f})")
    
    print("\n💡 Recommendations:")
    print("=" * 25)
    
    # Suggest normalized weights
    suggested_weights = {
        'lambda_similarity': 0.70,
        'alpha_reliability': 0.10,
        'beta_temporal': 0.05,
        'gamma_domain': 0.05,
        'delta_structure': 0.05,
        'epsilon_bm25': 0.05
    }
    
    print("Suggested weight configuration:")
    for weight_name, value in suggested_weights.items():
        print(f"  {weight_name}: {value}")
    
    # BM25 normalization recommendation
    bm25_max = np.max(all_bm25_scores)
    if bm25_max > 1:
        print(f"\nConsider normalizing BM25 scores to [0,1] range")
        print(f"Current BM25 max: {bm25_max:.3f}")
        print(f"Suggested normalization: score / {bm25_max:.1f}")

if __name__ == "__main__":
    analyze_score_distributions()