"""Weighted multi-stage retrieval with metadata-aware scoring."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..config import RetrievalConfig
from ..index.multi_index import MultiStageVectorIndex, StageResult
from ..types import Chunk, Query, RetrievedChunk, RetrievalResult


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class WeightedRetriever:
    """Implements the Matryoshka-aware weighted scoring strategy."""

    def __init__(self, config: RetrievalConfig, index: MultiStageVectorIndex):
        self.config = config
        self.index = index
        self.stage_weights = {stage.name: stage.weight for stage in config.stages}
        self.stage_dims = {stage.name: stage.dimension for stage in config.stages}

    def _metadata_scores(self, chunk: Chunk) -> Dict[str, float]:
        meta = chunk.metadata
        reliability = _parse_float(meta.get("reliability", "0"), 0.0)
        recency = _parse_float(meta.get("recency_score", "0"), 0.0)
        domain = _parse_float(meta.get("domain_score", "0"), 0.0)
        structure = _parse_float(meta.get("structure_score", "0"), 0.0)
        return {
            "alpha_reliability": reliability,
            "beta_temporal": recency,
            "gamma_domain": domain,
            "delta_structure": structure,
        }

    def _combine_scores(self, stage_results: Dict[str, StageResult]) -> Dict[str, float]:
        combined: Dict[str, float] = defaultdict(float)
        for stage_name, result in stage_results.items():
            weight = self.stage_weights[stage_name]
            for chunk_id, score in zip(result.ids, result.distances):
                combined[chunk_id] += weight * float(score)
        return combined

    def retrieve(self, query: Query, stage_results: Dict[str, StageResult]) -> RetrievalResult:
        combined = self._combine_scores(stage_results)

        items: List[RetrievedChunk] = []
        for chunk_id, similarity in combined.items():
            chunk = self.index.get_chunk(chunk_id)
            meta_scores = self._metadata_scores(chunk)
            weighted_similarity = (
                self.config.lambda_similarity * similarity
                + self.config.alpha_reliability * meta_scores["alpha_reliability"]
                + self.config.beta_temporal * meta_scores["beta_temporal"]
                + self.config.gamma_domain * meta_scores["gamma_domain"]
                + self.config.delta_structure * meta_scores["delta_structure"]
            )
            items.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity=weighted_similarity,
                    rank=0,
                    scores={"combined": similarity, **meta_scores},
                )
            )

        items.sort(key=lambda item: item.similarity, reverse=True)
        for rank, item in enumerate(items, start=1):
            item.rank = rank
        limit = self.config.stages[-1].top_k if self.config.stages else len(items)
        return RetrievalResult(query=query, chunks=items[:limit])
