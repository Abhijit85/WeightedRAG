"""Shared dataclasses used across the WeightedRAG pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class Document:
    """Represents a normalized source document."""

    doc_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a chunked segment derived from a document."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Query:
    """Encapsulates an incoming question."""

    query_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk matched during retrieval along with similarity information."""

    chunk: Chunk
    similarity: float
    rank: int
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Aggregated retrieval output per query."""

    query: Query
    chunks: Sequence[RetrievedChunk]


@dataclass
class GenerationResult:
    """Stores the generated answer and references to supporting chunks."""

    query: Query
    answer: str
    references: List[str]
    metadata: Dict[str, float] = field(default_factory=dict)
