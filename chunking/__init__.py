"""
WeightedRAG Chunking Module

This package contains all the chunking functionality for the WeightedRAG system,
including table processing, chunk generation, and retrieval preparation.

Package Structure:
- core/: Main chunking logic and orchestration
- processors/: Data processing and transformation modules  
- utils/: Utility functions and helper modules
- scripts/: Standalone scripts for batch processing
"""

from .core.create_retrieval_tables import *
from .core.enhanced_chunk_generator import *

__version__ = "1.0.0"
__author__ = "WeightedRAG Team"

# Export main functions
__all__ = [
    'create_retrieval_tables',
    'enhanced_chunk_generator',
    'chunk_generator', 
    'table_processor'
]