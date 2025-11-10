"""Dataset loaders for retrieval evaluation (BEIR-style)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ..types import Document, Query


@dataclass
class BEIRSplit:
    corpus: Dict[str, Document]
    queries: Dict[str, Query]
    qrels: Dict[str, Dict[str, float]]


def load_beir_corpus(path: Path) -> Dict[str, Document]:
    """Loads a BEIR corpus.jsonl file into Document objects."""
    corpus: Dict[str, Document] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            doc_id = str(payload["_id"])
            text = payload.get("title", "") + "\n" + payload.get("text", "")
            metadata = {k: str(v) for k, v in payload.items() if k not in {"_id", "title", "text"}}
            corpus[doc_id] = Document(doc_id=doc_id, text=text.strip(), metadata=metadata)
    return corpus


def load_beir_queries(path: Path) -> Dict[str, Query]:
    queries: Dict[str, Query] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            query_id = str(payload["_id"])
            text = payload["text"]
            queries[query_id] = Query(query_id=query_id, text=text)
    return queries


def load_beir_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            query_id, _, doc_id, relevance = line.strip().split()
            qrels.setdefault(query_id, {})[doc_id] = float(relevance)
    return qrels


def load_beir_split(root: Path) -> BEIRSplit:
    corpus = load_beir_corpus(root / "corpus.jsonl")
    queries = load_beir_queries(root / "queries.jsonl")
    qrels = load_beir_qrels(root / "qrels.tsv")
    return BEIRSplit(corpus=corpus, queries=queries, qrels=qrels)
