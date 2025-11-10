"""Weighted multi-stage retrieval with metadata-aware scoring."""

from __future__ import annotations

import json
import re
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

    def _query_filters(self, query: Query) -> Dict[str, List[str]]:
        filters: Dict[str, List[str]] = {}
        for key, raw in query.metadata.items():
            if not key.startswith("filter__"):
                continue
            field = key[len("filter__") :]
            terms = self._parse_filter_terms(raw)
            if terms:
                filters[field] = terms
        return filters

    def _parse_filter_terms(self, raw: str) -> List[str]:
        raw = (raw or "").strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                return [str(item).lower() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [part.strip().lower() for part in re.split(r"[;,]", raw) if part.strip()]

    def _normalize_metadata_entry(self, value: str) -> List[str]:
        value = (value or "").strip()
        if not value:
            return []
        try:
            payload = json.loads(value)
            if isinstance(payload, list):
                return [str(item).lower() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [segment.strip().lower() for segment in re.split(r"[;,]", value) if segment.strip()] or [value.lower()]

    def _matches_filters(self, chunk: Chunk, filters: Dict[str, List[str]]) -> bool:
        for field, terms in filters.items():
            if not terms:
                continue
            value = chunk.metadata.get(field)
            if value is None:
                return False
            normalized_values = self._normalize_metadata_entry(value)
            if not any(term in candidate for term in terms for candidate in normalized_values):
                return False
        return True

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

        filters = self._query_filters(query)
        if filters:
            filtered = [item for item in items if self._matches_filters(item.chunk, filters)]
            if filtered:
                items = filtered

        limit = self.config.stages[-1].top_k if self.config.stages else len(items)
        return RetrievalResult(query=query, chunks=items[:limit])
