"""Data ingestion utilities for the WeightedRAG pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from ..types import Document


@dataclass
class IngestionStats:
    total_files: int = 0
    total_documents: int = 0


def load_text_files(input_dir: Path, metadata: Optional[Dict[str, str]] = None) -> Iterator[Document]:
    """Loads UTF-8 text files from a directory into Document objects."""
    metadata = metadata or {}
    for entry in sorted(input_dir.glob("**/*.txt")):
        if not entry.is_file():
            continue
        doc_id = entry.stem
        text = entry.read_text(encoding="utf-8")
        yield Document(doc_id=doc_id, text=text, metadata={**metadata, "source_path": str(entry)})


def load_jsonl(input_path: Path, text_field: str = "text", id_field: str = "id") -> Iterator[Document]:
    """Loads documents from a JSON Lines file."""
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            text = payload[text_field]
            doc_id = str(payload.get(id_field) or payload.get("doc_id") or hash(text))
            metadata = {k: str(v) for k, v in payload.items() if k not in {text_field, id_field}}
            yield Document(doc_id=doc_id, text=text, metadata=metadata)


def load_documents(source: Path, **kwargs) -> List[Document]:
    """Dispatches to the correct loader based on file suffix."""
    documents: List[Document] = []
    if source.is_dir():
        documents.extend(load_text_files(source, kwargs.get("metadata")))
    elif source.suffix == ".jsonl":
        # Filter kwargs for load_jsonl
        jsonl_kwargs = {k: v for k, v in kwargs.items() if k in {"text_field", "id_field"}}
        documents.extend(load_jsonl(source, **jsonl_kwargs))
    else:
        raise ValueError(f"Unsupported source: {source}")
    return documents


def normalize_metadata(documents: Iterable[Document]) -> List[Document]:
    """Ensures metadata keys/values are strings and injects missing IDs."""
    normalized: List[Document] = []
    for index, doc in enumerate(documents):
        doc_id = doc.doc_id or f"doc-{index:08d}"
        metadata = {str(k): str(v) for k, v in doc.metadata.items()}
        normalized.append(Document(doc_id=doc_id, text=doc.text, metadata=metadata))
    return normalized
