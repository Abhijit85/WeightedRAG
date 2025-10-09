"""Matryoshka embedding wrapper with graceful fallbacks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from ..config import EmbeddingConfig, RetrievalConfig


@dataclass
class EmbeddingOutput:
    ids: List[str]
    vectors: np.ndarray
    slices: Dict[int, np.ndarray]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class MatryoshkaEmbedder:
    """Generates sliceable embeddings compatible with Matryoshka workflows."""

    def __init__(self, config: EmbeddingConfig, retrieval: Optional[RetrievalConfig] = None):
        self.config = config
        self.retrieval = retrieval
        self._model = None
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(config.model_name, device=config.device)
            except Exception:
                self._model = None

        requested_dims = set(config.truncate_dims or [])
        if retrieval:
            requested_dims.update(stage.dimension for stage in retrieval.stages)
        self.slice_dimensions = sorted(requested_dims)
        self.full_dimension = max(self.slice_dimensions) if self.slice_dimensions else 2048

    def _model_encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            return self._hash_encode(texts)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            device=self.config.device,
        )
        if self.config.use_fp16:
            vectors = vectors.astype(np.float16)
        return vectors

    def _hash_encode(self, texts: Sequence[str]) -> np.ndarray:
        """Deterministic embedding fallback using hashing."""
        vectors = np.zeros((len(texts), self.full_dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeat = math.ceil(self.full_dimension / len(digest))
            data = (digest * repeat)[: self.full_dimension]
            vectors[row] = np.frombuffer(data, dtype=np.uint8) / 255.0
        return vectors

    def embed(self, ids: Sequence[str], texts: Sequence[str]) -> EmbeddingOutput:
        if len(ids) != len(texts):
            raise ValueError("ids and texts length mismatch")

        vectors = self._model_encode(texts)
        if vectors.shape[1] < self.full_dimension:
            pad_width = self.full_dimension - vectors.shape[1]
            vectors = np.pad(vectors, ((0, 0), (0, pad_width)))
        elif vectors.shape[1] > self.full_dimension:
            vectors = vectors[:, : self.full_dimension]

        if self.config.normalize:
            vectors = l2_normalize(vectors)

        slices = {
            dim: l2_normalize(vectors[:, :dim].astype(vectors.dtype, copy=False))
            if dim < vectors.shape[1]
            else vectors
            for dim in self.slice_dimensions
        }
        return EmbeddingOutput(ids=list(ids), vectors=vectors, slices=slices)

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingOutput:
        ids = [f"row-{idx}" for idx in range(len(texts))]
        return self.embed(ids, texts)

    def embed_query(self, text: str) -> Dict[int, np.ndarray]:
        output = self.embed(["query"], [text])
        return {dim: output.slices[dim][0] for dim in self.slice_dimensions}
