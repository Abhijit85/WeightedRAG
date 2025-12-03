"""Structural embedding utilities for table-aware retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np

from ..config import EmbeddingConfig


class StructuralEmbedder:
    """Handles embedding of table structural information."""
    
    def __init__(self, config: EmbeddingConfig, cache_dir: Optional[str] = None):
        """
        Initialize structural embedder.
        
        Args:
            config: Embedding configuration
            cache_dir: Directory for caching embeddings
        """
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_structure(self, structure_data: Dict[str, Any]) -> str:
        """
        Convert structured table data to normalized text representation.
        
        Args:
            structure_data: Dictionary containing table structure information
            
        Returns:
            Normalized text representation suitable for embedding
        """
        components = []
        
        # Table description
        if 'table_description' in structure_data:
            desc = structure_data['table_description']
            # Clean up description
            if desc and isinstance(desc, str):
                desc = desc.strip()
                if len(desc) > 200:  # Truncate very long descriptions
                    desc = desc[:200] + "..."
                components.append(f"Description: {desc}")
        
        # Column information
        if 'columns' in structure_data and structure_data['columns']:
            columns = structure_data['columns']
            column_info = []
            
            for col_name, col_data in columns.items():
                if not col_name or not isinstance(col_data, dict):
                    continue
                
                # Column name and type
                col_type = col_data.get('type', 'unknown')
                col_info = f"{col_name} ({col_type})"
                
                # Add sample values if available
                samples = col_data.get('samples', [])
                if samples and isinstance(samples, list):
                    # Take first few non-empty samples
                    valid_samples = [s for s in samples[:3] if s and str(s).strip()]
                    if valid_samples:
                        sample_text = ', '.join(str(s)[:50] for s in valid_samples)  # Limit sample length
                        col_info += f": {sample_text}"
                
                column_info.append(col_info)
            
            if column_info:
                components.append(f"Columns: {' | '.join(column_info)}")
        
        # Table patterns/characteristics
        patterns = []
        if 'patterns' in structure_data and structure_data['patterns']:
            patterns.extend(structure_data['patterns'])
        
        # Infer additional patterns from structure
        if 'rows' in structure_data:
            row_count = structure_data['rows']
            if isinstance(row_count, (int, str)):
                try:
                    rows = int(row_count)
                    if rows > 100:
                        patterns.append('large')
                    elif rows < 5:
                        patterns.append('small')
                except (ValueError, TypeError):
                    pass
        
        if 'columns' in structure_data:
            col_count = len(structure_data['columns'])
            if col_count > 8:
                patterns.append('wide')
            elif col_count <= 2:
                patterns.append('narrow')
        
        if patterns:
            components.append(f"Patterns: {', '.join(set(patterns))}")
        
        # Row count information
        if 'rows' in structure_data:
            components.append(f"Rows: {structure_data['rows']}")
        
        return ' | '.join(components)
    
    def extract_query_structure_hints(self, query_text: str) -> str:
        """
        Extract structural hints from a query text.
        
        Args:
            query_text: The query text to analyze
            
        Returns:
            Structural description based on query hints
        """
        query_lower = query_text.lower()
        hints = []
        
        # Column-related hints
        column_keywords = ['column', 'columns', 'field', 'fields', 'attribute', 'attributes']
        if any(word in query_lower for word in column_keywords):
            hints.append("column-focused")
        
        # Row/record hints
        row_keywords = ['row', 'rows', 'record', 'records', 'entry', 'entries']
        if any(word in query_lower for word in row_keywords):
            hints.append("row-focused")
        
        # Data type hints
        if any(word in query_lower for word in ['name', 'title', 'label', 'id', 'identifier']):
            hints.append("has-identifiers")
        
        if any(word in query_lower for word in ['number', 'count', 'amount', 'value', 'price', 'cost']):
            hints.append("has-numeric")
        
        if any(word in query_lower for word in ['date', 'time', 'year', 'month', 'day', 'when']):
            hints.append("has-temporal")
        
        if any(word in query_lower for word in ['category', 'type', 'kind', 'class', 'group']):
            hints.append("has-categorical")
        
        # Size/scope hints
        if any(word in query_lower for word in ['all', 'every', 'total', 'complete', 'full']):
            hints.append("comprehensive")
        
        if any(word in query_lower for word in ['first', 'last', 'top', 'bottom', 'few', 'some']):
            hints.append("selective")
        
        # Relationship hints
        if any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs', 'between']):
            hints.append("comparative")
        
        if any(word in query_lower for word in ['list', 'table', 'chart', 'data']):
            hints.append("tabular")
        
        # Construct structural description
        if hints:
            return f"Query structure: {', '.join(hints)}"
        else:
            return "General table query"
    
    def create_embedding_cache_key(self, content: str) -> str:
        """Create a cache key for embedding content."""
        import hashlib
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings."""
        if not self.cache_dir:
            return {}
        
        cache_file = self.cache_dir / "structural_embeddings_cache.npz"
        if not cache_file.exists():
            return {}
        
        try:
            data = np.load(cache_file, allow_pickle=True)
            return data['embeddings'].item()
        except Exception:
            return {}
    
    def save_embedding_cache(self, cache: Dict[str, np.ndarray]):
        """Save embeddings to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "structural_embeddings_cache.npz"
        np.savez(cache_file, embeddings=cache)
    
    def batch_normalize_structures(self, structure_data_list: List[Dict[str, Any]]) -> List[str]:
        """
        Batch normalize multiple structure data objects.
        
        Args:
            structure_data_list: List of structure data dictionaries
            
        Returns:
            List of normalized text representations
        """
        return [self.normalize_structure(data) for data in structure_data_list]
    
    def compute_structure_similarity(self, 
                                   query_structure: str, 
                                   candidate_structures: List[str],
                                   embedder_func) -> List[float]:
        """
        Compute similarity between query structure and candidate structures.
        
        Args:
            query_structure: Normalized query structure text
            candidate_structures: List of normalized candidate structure texts
            embedder_func: Function to compute embeddings
            
        Returns:
            List of similarity scores
        """
        if not candidate_structures:
            return []
        
        # Embed query structure
        all_texts = [query_structure] + candidate_structures
        embeddings = embedder_func(all_texts)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Compute cosine similarities
        similarities = []
        for candidate_embedding in candidate_embeddings:
            similarity = self._cosine_similarity(query_embedding, candidate_embedding)
            # Normalize to [0, 1]
            normalized_sim = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            similarities.append(normalized_sim)
        
        return similarities
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))