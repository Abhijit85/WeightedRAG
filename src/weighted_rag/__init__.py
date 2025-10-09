"""
WeightedRAG: modular Matryoshka-aware RAG pipeline.

The package exposes a high level Pipeline class plus modular components for
ingestion, chunking, embedding, indexing, retrieval, scoring, and generation.
"""

from .pipeline import WeightedRAGPipeline

__all__ = ["WeightedRAGPipeline"]
