"""
WeightedRAG: modular Matryoshka-aware RAG pipeline.

The package exposes a high level Pipeline class plus modular components for
ingestion, chunking, embedding, indexing, retrieval, scoring, and generation.
"""

from .utils.env import load_env_file

load_env_file()

from .pipeline import WeightedRAGPipeline

__all__ = ["WeightedRAGPipeline"]
