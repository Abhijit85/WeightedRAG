"""Lightweight graph-based reranker leveraging token overlap."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from ..types import RetrievedChunk


class GraphReranker:
    """Applies a simple overlap-based centrality adjustment."""

    def __init__(self, threshold: float = 0.1, boost: float = 0.05):
        self.threshold = threshold
        self.boost = boost

    def _token_sets(self, text: str) -> set[str]:
        return {token.lower() for token in text.split() if token}

    def rerank(self, chunks: Iterable[RetrievedChunk]) -> List[RetrievedChunk]:
        items = list(chunks)
        token_cache = {item.chunk.chunk_id: self._token_sets(item.chunk.text) for item in items}
        adjacency = defaultdict(int)
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                left_tokens = token_cache[left.chunk.chunk_id]
                right_tokens = token_cache[right.chunk.chunk_id]
                if not left_tokens or not right_tokens:
                    continue
                overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
                if overlap >= self.threshold:
                    adjacency[left.chunk.chunk_id] += 1
                    adjacency[right.chunk.chunk_id] += 1
        for item in items:
            adjustment = adjacency[item.chunk.chunk_id] * self.boost
            item.similarity += adjustment
            item.scores["graph_boost"] = adjustment
        items.sort(key=lambda item: item.similarity, reverse=True)
        for rank, item in enumerate(items, start=1):
            item.rank = rank
        return items
