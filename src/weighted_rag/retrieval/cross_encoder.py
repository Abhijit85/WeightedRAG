"""Cross-encoder reranker adapted from the Enterprise-Chatbot reference."""

from __future__ import annotations

from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None

from ..config import CrossEncoderConfig
from ..types import Query, RetrievedChunk


class CrossEncoderReranker:
    """Applies a cross-encoder to refine retrieval results."""

    def __init__(self, config: CrossEncoderConfig):
        self.config = config
        self._model = None
        if CrossEncoder is not None:
            try:
                self._model = CrossEncoder(config.model_name, device=config.device)
            except Exception:
                self._model = None

    def rerank(self, query: Query, chunks: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        if self._model is None or not chunks:
            return list(chunks)

        pairs = [(query.text, item.chunk.text) for item in chunks]
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        scored = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
        reranked: List[RetrievedChunk] = []
        limit = self.config.top_n if self.config.top_n > 0 else len(scored)
        for rank, (item, score) in enumerate(scored[:limit], start=1):
            item.scores["cross_encoder"] = float(score)
            item.rank = rank
            item.similarity = float(score)
            reranked.append(item)
        return reranked
