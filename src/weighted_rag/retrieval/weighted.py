"""Weighted multi-stage retrieval with metadata-aware scoring."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List

try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from ..config import RetrievalConfig
from ..index.multi_index import MultiStageVectorIndex, StageResult
from ..types import Chunk, Query, RetrievedChunk, RetrievalResult
from .structural import StructuralSimilarityScorer


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class BM25Scorer:
    """Enhanced BM25 scorer with improved tokenization, query term frequency weighting, and robust normalization."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon  # Smoothing parameter for unseen terms
        self.corpus_tokens = []
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.avgdl = 0.0
        self.N = 0
        self.idf_cache = {}  # Cache IDF values for performance
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Initialize spaCy tokenizer
        if SPACY_AVAILABLE:
            try:
                self.nlp = English()
                # Add only tokenizer for better performance
                self.nlp.add_pipe('tokenizer')
            except Exception:
                self.nlp = None
        else:
            self.nlp = None
    
    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with punctuation handling, case normalization, and stop word removal."""
        if not text or not text.strip():
            return []
        
        # Use spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text.lower())
                tokens = [token.text for token in doc if not token.is_space and not token.is_punct]
            except Exception:
                # Fallback to regex-based tokenization
                tokens = self._regex_tokenize(text)
        else:
            tokens = self._regex_tokenize(text)
        
        # Filter stop words and very short tokens
        filtered_tokens = [
            token for token in tokens 
            if len(token) >= 2 and token not in self.stop_words
        ]
        
        return filtered_tokens
    
    def _regex_tokenize(self, text: str) -> List[str]:
        """Fallback regex-based tokenization."""
        text = text.lower()
        # Remove punctuation but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split and filter empty tokens
        return [token.strip() for token in text.split() if token.strip()]
    
    def fit(self, documents: List[str]):
        """Fit BM25 parameters on a corpus with enhanced preprocessing."""
        self.corpus_tokens = []
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.idf_cache = {}
        
        valid_documents = [doc for doc in documents if doc and doc.strip()]
        if not valid_documents:
            self.N = 0
            self.avgdl = 0.0
            return
        
        for doc in valid_documents:
            tokens = self.tokenize(doc)
            self.corpus_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            
            # Count unique terms in document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        self.N = len(valid_documents)
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)
        
        # Precompute IDF values for performance
        for term, df in self.doc_freqs.items():
            if df > 0:
                # Ensure IDF is always positive by using max(0.1, ...)
                idf_value = math.log(max(1.1, (self.N - df + 0.5) / (df + 0.5)))
                self.idf_cache[term] = max(0.1, idf_value)  # Ensure minimum positive IDF
    
    def score(self, query: str, document: str) -> float:
        """Calculate enhanced BM25 score with query term frequency weighting."""
        if self.N == 0 or not query.strip() or not document.strip():
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
            tf = doc_term_freqs.get(term, 0)  # Get term frequency, 0 if not found
            
            if tf > 0:  # Term found in document
                df = self.doc_freqs.get(term, 0)
                
                if df > 0 and term in self.idf_cache:
                    # Use cached IDF
                    idf = self.idf_cache[term]
                    
                    # BM25 with query term frequency weighting
                    tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    qtf_component = (qtf * (self.k1 + 1)) / (qtf + self.k1)  # Query term frequency weighting
                    
                    score += idf * tf_component * qtf_component
                else:
                    # Smoothing for terms in document but not in training vocabulary
                    score += self.epsilon * math.log(self.N + 1) * qtf
            else:
                # Term not in document - no contribution to score
                # This is correct BM25 behavior - only matching terms contribute
                pass
        
        return score
    
    def normalized_score(self, query: str, document: str) -> float:
        """Return BM25 score normalized to [0,1] range using sigmoid."""
        raw_score = self.score(query, document)
        # Log-sigmoid normalization with adaptive scaling
        scale = 5.0 + math.log(self.avgdl + 1)  # Adaptive scaling based on corpus
        normalized = 1.0 / (1.0 + math.exp(-raw_score / scale))
        return normalized


class WeightedRetriever:
    """Implements the Matryoshka-aware weighted scoring strategy."""

    def __init__(self, config: RetrievalConfig, index: MultiStageVectorIndex):
        self.config = config
        self.index = index
        self.stage_weights = {stage.name: stage.weight for stage in config.stages}
        self.stage_dims = {stage.name: stage.dimension for stage in config.stages}
        
        # Initialize BM25 scorer
        self.bm25_scorer = BM25Scorer()
        self._bm25_fitted = False
        
        # Initialize structural similarity scorer if enabled
        self.structural_scorer = None
        if (config.enable_structural_similarity and 
            hasattr(config, 'structural_chunks_path') and 
            config.structural_chunks_path):
            try:
                # We'll pass the embedder from the pipeline when it becomes available
                # For now, initialize with None and set it later
                self.structural_scorer = None
                self._structural_enabled = True
            except Exception as e:
                print(f"Warning: Failed to initialize structural similarity scorer: {e}")
                self._structural_enabled = False
        else:
            self._structural_enabled = False

    def set_embedder(self, embedder):
        """Set the embedder for structural similarity scoring."""
        if (self._structural_enabled and 
            hasattr(self.config, 'structural_chunks_path') and 
            self.config.structural_chunks_path):
            try:
                self.structural_scorer = StructuralSimilarityScorer(
                    embedding_model=embedder,
                    structural_chunks_path=self.config.structural_chunks_path,
                    cache_dir=getattr(self.config, 'structural_cache_dir', None)
                )
            except Exception as e:
                print(f"Warning: Failed to initialize structural similarity scorer: {e}")
                self.structural_scorer = None

    def _metadata_scores(self, chunk: Chunk) -> Dict[str, float]:
        meta = chunk.metadata
        reliability = _parse_float(meta.get("reliability", "0"), 0.0)
        recency = _parse_float(meta.get("recency_score", "0"), 0.0)
        domain = _parse_float(meta.get("domain_score", "0"), 0.0)
        return {
            "alpha_reliability": reliability,
            "beta_temporal": recency,
            "gamma_domain": domain,
        }
    
    def _ensure_bm25_fitted(self):
        """Ensure BM25 scorer is fitted on the corpus."""
        if not self._bm25_fitted:
            # Get all chunk texts from index
            all_chunks = self.index.get_all_chunks()
            documents = [chunk.text for chunk in all_chunks]
            
            if documents:
                self.bm25_scorer.fit(documents)
                self._bm25_fitted = True
    
    def _compute_bm25_score(self, query: Query, chunk: Chunk) -> float:
        """Compute normalized BM25 score for a query-chunk pair."""
        self._ensure_bm25_fitted()
        # Use the new normalized scoring method for better distribution
        return self.bm25_scorer.normalized_score(query.text, chunk.text)

    def _combine_scores(self, stage_results: Dict[str, StageResult]) -> Dict[str, float]:
        combined: Dict[str, float] = defaultdict(float)
        for stage_name, result in stage_results.items():
            weight = self.stage_weights[stage_name]
            for chunk_id, score in zip(result.ids, result.distances):
                combined[chunk_id] += weight * float(score)
        return combined

    def _query_filters(self, query: Query) -> Dict[str, List[str]]:
        filters: Dict[str, List[str]] = {}
        for key, raw in query.metadata.items():
            if not key.startswith("filter__"):
                continue
            field = key[len("filter__") :]
            terms = self._parse_filter_terms(raw)
            if terms:
                filters[field] = terms
        return filters

    def _parse_filter_terms(self, raw: str) -> List[str]:
        raw = (raw or "").strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                return [str(item).lower() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [part.strip().lower() for part in re.split(r"[;,]", raw) if part.strip()]

    def _normalize_metadata_entry(self, value: str) -> List[str]:
        value = (value or "").strip()
        if not value:
            return []
        try:
            payload = json.loads(value)
            if isinstance(payload, list):
                return [str(item).lower() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [segment.strip().lower() for segment in re.split(r"[;,]", value) if segment.strip()] or [value.lower()]

    def _matches_filters(self, chunk: Chunk, filters: Dict[str, List[str]]) -> bool:
        for field, terms in filters.items():
            if not terms:
                continue
            value = chunk.metadata.get(field)
            if value is None:
                return False
            normalized_values = self._normalize_metadata_entry(value)
            if not any(term in candidate for term in terms for candidate in normalized_values):
                return False
        return True

    def retrieve(self, query: Query, stage_results: Dict[str, StageResult]) -> RetrievalResult:
        combined = self._combine_scores(stage_results)

        # Calculate total weight for normalization
        total_weight = (
            self.config.lambda_similarity 
            + self.config.alpha_reliability 
            + self.config.beta_temporal 
            + self.config.gamma_domain 
            + getattr(self.config, 'zeta_structural_similarity', 0.0)
            + self.config.epsilon_bm25
        )
        
        items: List[RetrievedChunk] = []
        for chunk_id, similarity in combined.items():
            chunk = self.index.get_chunk(chunk_id)
            meta_scores = self._metadata_scores(chunk)
            
            # Compute BM25 score (already normalized to [0,1])
            bm25_score = self._compute_bm25_score(query, chunk)
            
            # Create initial retrieved chunk for structural similarity computation
            temp_retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                similarity=similarity,
                rank=0,
                scores={}
            )
            
            items.append(temp_retrieved_chunk)
        
        # Compute structural similarities for all chunks at once
        structural_similarities = []
        if (self.structural_scorer and 
            getattr(self.config, 'zeta_structural_similarity', 0.0) > 0):
            try:
                structural_similarities = self.structural_scorer.compute_structural_similarity(query, items)
            except Exception as e:
                print(f"Warning: Structural similarity computation failed: {e}")
                structural_similarities = [0.0] * len(items)
        else:
            structural_similarities = [0.0] * len(items)
        
        # Update items with final weighted scores
        for i, item in enumerate(items):
            chunk = item.chunk
            similarity = item.similarity
            meta_scores = self._metadata_scores(chunk)
            bm25_score = self._compute_bm25_score(query, chunk)
            structural_sim = structural_similarities[i] if i < len(structural_similarities) else 0.0
            
            # Compute weighted similarity with normalized weights
            weighted_similarity = (
                (self.config.lambda_similarity / total_weight) * similarity
                + (self.config.alpha_reliability / total_weight) * meta_scores["alpha_reliability"]
                + (self.config.beta_temporal / total_weight) * meta_scores["beta_temporal"]
                + (self.config.gamma_domain / total_weight) * meta_scores["gamma_domain"]
                + (getattr(self.config, 'zeta_structural_similarity', 0.0) / total_weight) * structural_sim
                + (self.config.epsilon_bm25 / total_weight) * bm25_score
            )
            
            # Add all scores to the scores dictionary
            all_scores = {
                "combined": similarity, 
                "bm25": bm25_score, 
                "structural_similarity": structural_sim,
                "weighted": weighted_similarity, 
                **meta_scores
            }
            
            # Update the item
            item.similarity = weighted_similarity
            item.scores = all_scores

        items.sort(key=lambda item: item.similarity, reverse=True)
        for rank, item in enumerate(items, start=1):
            item.rank = rank

        filters = self._query_filters(query)
        if filters:
            filtered = [item for item in items if self._matches_filters(item.chunk, filters)]
            if filtered:
                items = filtered

        limit = self.config.stages[-1].top_k if self.config.stages else len(items)
        return RetrievalResult(query=query, chunks=items[:limit])
