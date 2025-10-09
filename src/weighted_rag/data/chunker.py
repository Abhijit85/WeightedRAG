"""Document chunking utilities with overlap-aware splitting."""

from __future__ import annotations

import itertools
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None

from ..config import ChunkingConfig
from ..types import Chunk, Document


@dataclass
class ChunkingStats:
    documents: int = 0
    chunks: int = 0
    avg_chunks_per_doc: float = 0.0


class DocumentChunker:
    """Splits documents into overlapping text chunks."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._tokenizer = None
        if AutoTokenizer is not None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            except Exception:
                self._tokenizer = None

    def _basic_sentence_split(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s]

    def _tokenize(self, text: str) -> List[int]:
        if self._tokenizer:
            encoded = self._tokenizer.encode(text, add_special_tokens=False)
            return encoded
        return text.split()

    def _detokenize(self, tokens: Sequence) -> str:
        if self._tokenizer:
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        return " ".join(tokens)

    def chunk_document(self, document: Document) -> List[Chunk]:
        tokens = self._tokenize(document.text)
        if not tokens:
            return []

        stride = self.config.max_tokens - self.config.overlap_tokens
        if stride <= 0:
            raise ValueError("Chunk overlap must be smaller than max tokens")

        chunks: List[Chunk] = []
        for index in range(0, len(tokens), stride):
            window = tokens[index : index + self.config.max_tokens]
            if not window:
                continue
            text = self._detokenize(window)
            chunk_id = f"{document.doc_id}__{uuid.uuid4().hex[:8]}"
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=text,
                start_char=0,
                end_char=0,
                token_count=len(window),
                metadata={**document.metadata, "chunk_index": str(len(chunks))},
            )
            chunks.append(chunk)
        return chunks

    def chunk_corpus(self, documents: Iterable[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for document in documents:
            all_chunks.extend(self.chunk_document(document))
        return all_chunks
