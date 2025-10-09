"""Evaluation metrics for retrieval and generation."""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Set, Tuple


def precision_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / max(k, 1)


def recall_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / max(len(relevant), 1)


def mean_reciprocal_rank(relevant: Set[str], retrieved: Sequence[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), ref_tokens.count(token))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
