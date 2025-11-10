"""Multi-stage in-memory vector index with optional FAISS acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

from ..config import IndexStageConfig, RetrievalConfig
from ..types import Chunk


@dataclass
class StageResult:
    ids: List[str]
    distances: np.ndarray


class StageIndex:
    """Stores a single-dimension vector index."""

    def __init__(self, config: IndexStageConfig):
        self.config = config
        self.ids: List[str] = []
        self.matrix: np.ndarray = np.zeros((0, config.dimension), dtype=np.float32)
        self._faiss_index = None
        if faiss is not None:
            try:
                self._faiss_index = faiss.index_factory(config.dimension, config.index_factory)
                if hasattr(self._faiss_index, "hnsw"):
                    self._faiss_index.hnsw.efSearch = config.ef_search
                    self._faiss_index.hnsw.efConstruction = config.ef_construction
            except Exception:
                self._faiss_index = None

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = vectors.astype(np.float32)
        self.ids.extend(ids)
        self.matrix = np.vstack([self.matrix, vectors]) if self.matrix.size else vectors
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> StageResult:
        if self._faiss_index is not None and len(self.ids) >= top_k:
            distances, indices = self._faiss_index.search(query[np.newaxis, :], top_k)
            hits = [self.ids[idx] for idx in indices[0] if idx != -1]
            dists = distances[0][: len(hits)]
            return StageResult(ids=hits, distances=dists)

        if not self.ids:
            return StageResult(ids=[], distances=np.array([], dtype=np.float32))

        scores = self.matrix @ query.astype(np.float32)
        top_k = min(top_k, len(self.ids))
        best_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        sorted_idx = best_idx[np.argsort(-scores[best_idx])]
        return StageResult(ids=[self.ids[i] for i in sorted_idx], distances=scores[sorted_idx])


class MultiStageVectorIndex:
    """Coordinates stage-specific indexes and chunk metadata."""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.stages = {stage.name: StageIndex(stage) for stage in config.stages}
        self.chunk_store: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Dict[int, np.ndarray]) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        for chunk in chunks:
            self.chunk_store[chunk.chunk_id] = chunk

        for stage in self.config.stages:
            vectors = embeddings[stage.dimension]
            if vectors.shape[0] != len(ids):
                raise ValueError("Embedding/count mismatch")
            self.stages[stage.name].add(ids, vectors[:, : stage.dimension])

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.chunk_store[chunk_id]

    def search(self, query_embeddings: Dict[int, np.ndarray]) -> Dict[str, StageResult]:
        results: Dict[str, StageResult] = {}
        for stage in self.config.stages:
            vector = query_embeddings[stage.dimension]
            result = self.stages[stage.name].search(vector, stage.top_k)
            results[stage.name] = result
        return results
