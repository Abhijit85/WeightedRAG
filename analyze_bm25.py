#!/usr/bin/env python3
"""Comprehensive analysis and testing of BM25 implementation for robustness improvements."""

import math
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import re

class BM25AnalysisSuite:
    """Test suite to analyze BM25 implementation and suggest improvements."""
    
    def __init__(self):
        self.test_documents = [
            "The capital of France is Paris. It is known for the Eiffel Tower and Louvre Museum.",
            "Paris is a beautiful city in France with rich history and culture.",
            "France has many cities but Paris is the most famous capital city.",
            "Machine learning algorithms use data to make predictions and classifications.",
            "Python programming language is popular for data science and machine learning.",
            "The Eiffel Tower in Paris attracts millions of tourists every year.",
            "Database systems store and manage large amounts of structured data efficiently.",
            "SQL queries are used to retrieve information from relational databases.",
            "Web development involves HTML, CSS, JavaScript for frontend applications.",
            "Natural language processing helps computers understand human language."
        ]
        
        self.test_queries = [
            "capital of France",
            "Paris Eiffel Tower", 
            "machine learning data",
            "database SQL queries",
            "Python programming"
        ]
    
    def test_current_implementation(self):
        """Test the current BM25 implementation."""
        print("🔍 Testing Current BM25 Implementation")
        print("=" * 50)
        
        # Import current implementation
        import sys
        import os
        sys.path.append(os.getcwd())
        from src.weighted_rag.retrieval.weighted import BM25Scorer
        
        scorer = BM25Scorer()
        scorer.fit(self.test_documents)
        
        print(f"Corpus size: {scorer.N}")
        print(f"Average document length: {scorer.avgdl:.2f}")
        print(f"Vocabulary size: {len(scorer.doc_freqs)}")
        print()
        
        # Test scoring behavior
        for query in self.test_queries:
            print(f"Query: '{query}'")
            scores = []
            for i, doc in enumerate(self.test_documents):
                score = scorer.score(query, doc)
                scores.append((i, score, doc[:60] + "..."))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            print("  Top 3 matches:")
            for rank, (doc_idx, score, text) in enumerate(scores[:3], 1):
                print(f"    {rank}. Doc {doc_idx}: {score:.4f} - {text}")
            
            # Analyze score distribution
            score_values = [s[1] for s in scores]
            print(f"  Score range: {min(score_values):.4f} to {max(score_values):.4f}")
            print(f"  Score std dev: {np.std(score_values):.4f}")
            print()
    
    def analyze_potential_issues(self):
        """Identify potential issues with current implementation."""
        print("⚠️  Potential Issues Analysis")
        print("=" * 40)
        
        import sys
        import os
        sys.path.append(os.getcwd())
        from src.weighted_rag.retrieval.weighted import BM25Scorer
        
        issues = []
        
        # 1. Test with edge cases
        scorer = BM25Scorer()
        
        # Empty documents
        try:
            scorer.fit(["", "  ", "\n\t"])
            if scorer.N > 0:
                score = scorer.score("test", "")
                issues.append("❌ Empty documents not handled properly")
        except:
            issues.append("❌ Empty documents cause crashes")
        
        # Very short documents
        scorer.fit(["a", "b", "c"])
        score = scorer.score("a", "a")
        if score <= 0:
            issues.append("❌ Single-character documents give poor scores")
        
        # Very long documents
        long_doc = " ".join(["word"] * 1000)
        scorer.fit([long_doc])
        score = scorer.score("word", long_doc)
        if not (0 <= score <= 50):  # Reasonable range check
            issues.append("❌ Very long documents give unreasonable scores")
        
        # Special characters and punctuation
        special_doc = "Hello, world! How are you? I'm fine."
        scorer.fit([special_doc])
        score1 = scorer.score("Hello world", special_doc)
        score2 = scorer.query("hello world", special_doc) if hasattr(scorer, 'query') else scorer.score("hello world", special_doc)
        # Note: Current implementation might not handle case sensitivity well
        
        # Out-of-vocabulary terms
        scorer.fit(self.test_documents)
        oov_score = scorer.score("xyzzyx", self.test_documents[0])
        if oov_score != 0:
            issues.append("❌ Out-of-vocabulary terms not handled correctly")
        
        # Repeated terms in query
        normal_score = scorer.score("Paris", self.test_documents[0])
        repeated_score = scorer.score("Paris Paris Paris", self.test_documents[0])
        if repeated_score <= normal_score:
            issues.append("❌ Query term frequency not properly weighted")
        
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        
        if not issues:
            print("  ✅ No major issues detected")
        
        return issues
    
    def suggest_improvements(self):
        """Suggest improvements for more robust BM25 implementation."""
        print("\n💡 Suggested Improvements")
        print("=" * 35)
        
        improvements = {
            "1. Enhanced Tokenization": [
                "- Use more sophisticated tokenization (remove punctuation, handle contractions)",
                "- Add stemming/lemmatization for better term matching", 
                "- Handle case normalization consistently",
                "- Filter stop words for better relevance"
            ],
            "2. Query Processing": [
                "- Add query term frequency (multiple occurrences should increase relevance)",
                "- Implement query expansion with synonyms",
                "- Add phrase detection and proximity scoring",
                "- Handle boolean operators (AND, OR, NOT)"
            ],
            "3. Score Normalization": [
                "- Use log-sigmoid normalization: 1 / (1 + exp(-score/scale))",
                "- Adaptive normalization based on corpus statistics",
                "- Min-max normalization with corpus-specific bounds",
                "- Z-score normalization for better distribution"
            ],
            "4. Performance Optimizations": [
                "- Cache tokenized documents to avoid re-tokenization", 
                "- Precompute IDF values for faster scoring",
                "- Use sparse vectors for large vocabularies",
                "- Implement incremental fitting for new documents"
            ],
            "5. Robustness Enhancements": [
                "- Better handling of empty/very short documents",
                "- Smoothing for unseen terms (add-one smoothing)",
                "- Document length normalization improvements",
                "- Handling of special characters and Unicode"
            ]
        }
        
        for category, items in improvements.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  {item}")
    
    def demonstrate_improved_bm25(self):
        """Demonstrate an improved BM25 implementation."""
        print("\n🚀 Enhanced BM25 Implementation Demo")
        print("=" * 45)
        
        class EnhancedBM25:
            def __init__(self, k1=1.2, b=0.75, epsilon=0.25):
                self.k1 = k1
                self.b = b
                self.epsilon = epsilon  # Smoothing parameter
                self.doc_freqs = Counter()
                self.doc_lens = []
                self.avgdl = 0.0
                self.N = 0
                self.idf_cache = {}
                
            def tokenize(self, text: str) -> List[str]:
                """Enhanced tokenization with better preprocessing."""
                if not text or not text.strip():
                    return []
                
                # Convert to lowercase and handle basic punctuation
                text = text.lower()
                # Remove punctuation but keep alphanumeric and spaces
                text = re.sub(r'[^\w\s]', ' ', text)
                # Split and filter empty tokens
                tokens = [token.strip() for token in text.split() if token.strip()]
                
                # Basic stop word removal (minimal set)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
                tokens = [token for token in tokens if token not in stop_words]
                
                return tokens
            
            def fit(self, documents: List[str]):
                """Fit with enhanced preprocessing."""
                self.doc_freqs = Counter()
                self.doc_lens = []
                tokenized_docs = []
                
                for doc in documents:
                    tokens = self.tokenize(doc)
                    tokenized_docs.append(tokens)
                    self.doc_lens.append(len(tokens))
                    
                    # Count unique terms
                    unique_tokens = set(tokens)
                    for token in unique_tokens:
                        self.doc_freqs[token] += 1
                
                self.N = len(documents)
                self.avgdl = sum(self.doc_lens) / max(self.N, 1)
                
                # Precompute IDF values
                for term, df in self.doc_freqs.items():
                    self.idf_cache[term] = math.log((self.N - df + 0.5) / (df + 0.5))
            
            def score(self, query: str, document: str) -> float:
                """Enhanced BM25 scoring with improvements."""
                if self.N == 0:
                    return 0.0
                
                query_tokens = self.tokenize(query)
                doc_tokens = self.tokenize(document)
                
                if not query_tokens or not doc_tokens:
                    return 0.0
                
                doc_len = len(doc_tokens)
                doc_term_freqs = Counter(doc_tokens)
                query_term_freqs = Counter(query_tokens)  # Consider query term frequency
                
                score = 0.0
                for term, qtf in query_term_freqs.items():
                    if term in doc_term_freqs:
                        tf = doc_term_freqs[term]
                        df = self.doc_freqs.get(term, 0)
                        
                        if df > 0:
                            # Use cached IDF
                            idf = self.idf_cache[term]
                            
                            # Enhanced BM25 with query term frequency
                            numerator = tf * (self.k1 + 1)
                            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                            
                            # Add query term frequency weighting
                            qtf_weight = (qtf * (self.k1 + 1)) / (qtf + self.k1)
                            
                            score += idf * (numerator / denominator) * qtf_weight
                        else:
                            # Add smoothing for unseen terms
                            score += self.epsilon * math.log(self.N)
                
                return score
            
            def normalized_score(self, query: str, document: str) -> float:
                """Return normalized score in [0,1] range."""
                raw_score = self.score(query, document)
                # Log-sigmoid normalization
                normalized = 1.0 / (1.0 + math.exp(-raw_score / 5.0))
                return normalized
        
        # Test enhanced implementation
        enhanced_bm25 = EnhancedBM25()
        enhanced_bm25.fit(self.test_documents)
        
        print("Enhanced BM25 Results:")
        for query in self.test_queries[:2]:  # Test first 2 queries
            print(f"\nQuery: '{query}'")
            
            scores = []
            for i, doc in enumerate(self.test_documents):
                raw_score = enhanced_bm25.score(query, doc)
                norm_score = enhanced_bm25.normalized_score(query, doc)
                scores.append((i, raw_score, norm_score, doc[:50] + "..."))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            print("  Top 3 matches:")
            for rank, (doc_idx, raw_score, norm_score, text) in enumerate(scores[:3], 1):
                print(f"    {rank}. Doc {doc_idx}: Raw={raw_score:.3f}, Norm={norm_score:.3f}")
                print(f"        {text}")
    
    def run_full_analysis(self):
        """Run complete BM25 analysis."""
        self.test_current_implementation()
        issues = self.analyze_potential_issues()
        self.suggest_improvements()
        self.demonstrate_improved_bm25()
        
        print("\n📊 Summary")
        print("=" * 20)
        print(f"Total issues found: {len(issues)}")
        print("Key recommendations:")
        print("  1. Implement enhanced tokenization with stop word removal")
        print("  2. Add query term frequency weighting")
        print("  3. Use log-sigmoid normalization for better score distribution")
        print("  4. Add smoothing for unseen terms")
        print("  5. Cache precomputed values for better performance")

if __name__ == "__main__":
    suite = BM25AnalysisSuite()
    suite.run_full_analysis()