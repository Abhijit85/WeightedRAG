"""Structural similarity scoring for table-aware retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from ..types import Query, RetrievedChunk, Chunk


class StructuralSimilarityScorer:
    """Computes structural similarity between queries and retrieved table chunks."""
    
    def __init__(self, embedding_model, structural_chunks_path: str, cache_dir: Optional[str] = None):
        """
        Initialize with embedding model and path to structural chunks.
        
        Args:
            embedding_model: The same embedding model used for content embeddings
            structural_chunks_path: Path to chunks_table_structure.jsonl file
            cache_dir: Directory for caching structural embeddings
        """
        self.embedding_model = embedding_model
        self.structural_chunks_path = Path(structural_chunks_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Internal storage
        self._structural_documents: Dict[str, Dict[str, Any]] = {}
        self._structural_embeddings: Dict[str, np.ndarray] = {}
        self._loaded = False
        
        # Initialize cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _ensure_loaded(self):
        """Ensure structural documents are loaded and embeddings computed."""
        if not self._loaded:
            self.load_structural_documents()
            self._compute_embeddings()
            self._loaded = True
    
    def load_structural_documents(self) -> Dict[str, Dict[str, Any]]:
        """Load all structural documents indexed by table_id."""
        if not self.structural_chunks_path.exists():
            print(f"Warning: Structural chunks file not found: {self.structural_chunks_path}")
            return {}
        
        structural_docs = {}
        
        with open(self.structural_chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk_data = json.loads(line.strip())
                    if chunk_data.get('chunk_type') == 'table_structure':
                        table_id = chunk_data.get('table_id')
                        if table_id:
                            # Parse the JSON content which contains the actual structure
                            content = chunk_data.get('content', '{}')
                            if isinstance(content, str):
                                structure_data = json.loads(content)
                            else:
                                structure_data = content
                            
                            structural_docs[table_id] = {
                                'id': chunk_data['id'],
                                'table_id': table_id,
                                'structure_data': structure_data,
                                'raw_content': content if isinstance(content, str) else json.dumps(content)
                            }
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse structural chunk: {e}")
                    continue
        
        self._structural_documents = structural_docs
        print(f"Loaded {len(structural_docs)} structural documents")
        return structural_docs
    
    def get_structural_document(self, table_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve exact structural document for a table_id."""
        self._ensure_loaded()
        return self._structural_documents.get(table_id)
    
    def _compute_embeddings(self):
        """Pre-compute embeddings for all structural documents."""
        if not self._structural_documents:
            return
        
        # Check cache first
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / "structural_embeddings.npz"
            if cache_file.exists():
                try:
                    cache_data = np.load(cache_file, allow_pickle=True)
                    cached_embeddings = cache_data['embeddings'].item()
                    cached_table_ids = cache_data['table_ids'].tolist()
                    
                    # Verify cache is complete and up-to-date
                    if set(cached_table_ids) == set(self._structural_documents.keys()):
                        self._structural_embeddings = cached_embeddings
                        print(f"Loaded {len(cached_embeddings)} structural embeddings from cache")
                        return
                except Exception as e:
                    print(f"Warning: Failed to load cached embeddings: {e}")
        
        # Compute embeddings
        print("Computing structural embeddings...")
        texts_to_embed = []
        table_ids_ordered = []
        
        for table_id, doc_data in self._structural_documents.items():
            # Create a normalized text representation of the structure for embedding
            structure_text = self._structure_to_text(doc_data['structure_data'])
            texts_to_embed.append(structure_text)
            table_ids_ordered.append(table_id)
        
        if texts_to_embed:
            # Use the same embedding model as content
            embeddings = self._embed_texts(texts_to_embed)
            
            # Store embeddings indexed by table_id
            for table_id, embedding in zip(table_ids_ordered, embeddings):
                self._structural_embeddings[table_id] = embedding
            
            # Cache results
            if cache_file:
                np.savez(
                    cache_file,
                    embeddings=self._structural_embeddings,
                    table_ids=table_ids_ordered
                )
            
            print(f"Computed {len(self._structural_embeddings)} structural embeddings")
    
    def _structure_to_text(self, structure_data: Dict[str, Any]) -> str:
        """Convert structure data to normalized text for embedding."""
        parts = []
        
        # Add table description
        if 'table_description' in structure_data:
            parts.append(f"Table: {structure_data['table_description']}")
        
        # Add column information
        if 'columns' in structure_data:
            columns = structure_data['columns']
            column_parts = []
            for col_name, col_info in columns.items():
                col_type = col_info.get('type', 'unknown')
                samples = col_info.get('samples', [])
                sample_text = ', '.join(samples[:3]) if samples else ''
                column_parts.append(f"{col_name} ({col_type}): {sample_text}")
            
            if column_parts:
                parts.append(f"Columns: {' | '.join(column_parts)}")
        
        # Add patterns
        if 'patterns' in structure_data and structure_data['patterns']:
            patterns = ', '.join(structure_data['patterns'])
            parts.append(f"Patterns: {patterns}")
        
        # Add row count
        if 'rows' in structure_data:
            parts.append(f"Rows: {structure_data['rows']}")
        
        return ' '.join(parts)
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the same model as content embeddings."""
        # Try to use the first available model in the embedder
        if hasattr(self.embedding_model, '_models') and self.embedding_model._models:
            # Get the first available model
            available_models = {k: v for k, v in self.embedding_model._models.items() if v is not None}
            if available_models:
                model_name = list(available_models.keys())[0]
                model = available_models[model_name]
                embeddings = model.encode(
                    texts,
                    batch_size=64,
                    normalize_embeddings=False,
                    convert_to_numpy=True
                )
                return embeddings.astype(np.float32)
        
        # Fallback: try using _model_encode with available model
        if hasattr(self.embedding_model, '_model_encode') and hasattr(self.embedding_model, '_models'):
            available_models = {k: v for k, v in self.embedding_model._models.items() if v is not None}
            if available_models:
                model_name = list(available_models.keys())[0]
                return self.embedding_model._model_encode(texts, model_name)
        
        # Last resort: use embed_by_stage if available
        if hasattr(self.embedding_model, 'embed_by_stage'):
            try:
                # Create dummy IDs for the texts
                ids = [f"structural_{i}" for i in range(len(texts))]
                result = self.embedding_model.embed_by_stage(ids, texts)
                # Get embeddings from the first stage
                if result:
                    stage_name = list(result.keys())[0]
                    return result[stage_name].vectors
            except Exception:
                pass
        
        # Final fallback to hash encoding (this is what was causing the issue)
        print("Warning: Using hash encoding fallback for structural embeddings - this may produce poor results")
        return self.embedding_model._hash_encode(texts)
    
    def _infer_query_structure(self, query: Query) -> str:
        """Infer expected table structure from query text."""
        query_text = query.text.lower()
        
        # Simple heuristics for structure inference
        structure_hints = []
        
        # Common table structure indicators
        if any(word in query_text for word in ['column', 'columns', 'field', 'fields']):
            structure_hints.append("has_columns")
        
        if any(word in query_text for word in ['row', 'rows', 'record', 'records']):
            structure_hints.append("has_rows")
        
        if any(word in query_text for word in ['name', 'title', 'label']):
            structure_hints.append("has_identifiers")
        
        if any(word in query_text for word in ['number', 'count', 'amount', 'value']):
            structure_hints.append("has_numeric_data")
        
        if any(word in query_text for word in ['date', 'time', 'year', 'month']):
            structure_hints.append("has_temporal_data")
        
        # Create a simple structure description
        if structure_hints:
            return f"Expected structure: {', '.join(structure_hints)}"
        else:
            return "General table structure query"
    
    def compute_structural_similarity(self, query: Query, retrieved_chunks: List[RetrievedChunk]) -> List[float]:
        """
        Compute structural similarity scores for retrieved chunks.
        
        Args:
            query: The search query
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            List of similarity scores (0-1) for each chunk
        """
        self._ensure_loaded()
        
        if not retrieved_chunks:
            return []
        
        # Infer query structure and compute embedding
        query_structure_text = self._infer_query_structure(query)
        query_embedding = self._embed_texts([query_structure_text])[0]
        
        similarities = []
        
        for chunk in retrieved_chunks:
            # Get table_id from chunk metadata
            table_id = chunk.chunk.metadata.get('table_id')
            
            if not table_id:
                # Try to extract from chunk_id if table_id not in metadata
                chunk_id = chunk.chunk.chunk_id
                if '_' in chunk_id:
                    table_id = chunk_id.split('_')[0]
            
            if table_id and table_id in self._structural_embeddings:
                # Compute cosine similarity
                struct_embedding = self._structural_embeddings[table_id]
                similarity = self._cosine_similarity(query_embedding, struct_embedding)
                # Normalize to [0, 1] range
                normalized_similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
                similarities.append(normalized_similarity)
            else:
                # No structural information available
                similarities.append(0.0)
        
        return similarities
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))