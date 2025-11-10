"""PDF ingestion helpers inspired by the Enterprise-Chatbot reference implementation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from ..types import Document

try:  # pragma: no cover - optional enterprise dependencies
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - optional dependency
    PyPDFDirectoryLoader = None
    RecursiveCharacterTextSplitter = None

try:  # pragma: no cover - optional dependency
    import spacy
except Exception:  # pragma: no cover - optional dependency
    spacy = None


@dataclass
class PDFIngestionConfig:
    """Configuration for PDF ingestion mirroring Enterprise-Chatbot defaults."""

    chunk_size: int = 2_000
    chunk_overlap: int = 500
    spacy_model: str = "en_core_web_sm"
    document_types: Sequence[str] = field(
        default_factory=lambda: ["HR Policy", "Financial Report", "Technical Manual", "Internal Memo"]
    )
    departments: Sequence[str] = field(
        default_factory=lambda: ["HR", "Finance", "Engineering", "Operations"]
    )
    confidentiality_levels: Sequence[str] = field(
        default_factory=lambda: ["Public", "Internal", "Confidential"]
    )
    authors: Sequence[str] = field(default_factory=lambda: [f"Author_{idx}" for idx in range(1, 11)])


_NLP = None
_SPACY_FAILURE = False


def _get_nlp(model_name: str):
    global _NLP, _SPACY_FAILURE
    if _SPACY_FAILURE or spacy is None:
        return None
    if _NLP is None:
        try:
            _NLP = spacy.load(model_name)
        except Exception:
            _SPACY_FAILURE = True
            return None
    return _NLP


def _extract_locations(text: str, model_name: str) -> List[str]:
    nlp = _get_nlp(model_name)
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "GPE"]


def _stable_choice(options: Sequence[str], seed: str) -> str:
    if not options:
        return ""
    rng = random.Random(seed)
    return rng.choice(list(options))


def load_pdf_documents(folder: Path, config: PDFIngestionConfig | None = None) -> List[Document]:
    """Loads PDFs, splits them, and enriches metadata similar to the reference bot."""

    if PyPDFDirectoryLoader is None or RecursiveCharacterTextSplitter is None:
        raise ImportError(
            "langchain_community and langchain packages are required for PDF ingestion. "
            "Install them to enable Enterprise-style loaders."
        )

    config = config or PDFIngestionConfig()
    loader = PyPDFDirectoryLoader(str(folder))
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    splits = splitter.split_documents(raw_docs)

    documents: List[Document] = []
    for index, chunk in enumerate(splits):
        text = chunk.page_content.strip()
        if not text:
            continue
        source = chunk.metadata.get("source", f"{folder}/unknown")
        title = chunk.metadata.get("title") or Path(source).name or f"doc_{index}"
        parent_id = chunk.metadata.get("source") or f"{Path(source).stem}"
        locations = _extract_locations(text, config.spacy_model)

        location_str = json.dumps(locations) if locations else "[]"
        seed = f"{parent_id}-{index}"
        metadata = {
            "title": title,
            "source_path": source,
            "parent_id": parent_id,
            "chunk_hint": f"{parent_id}_chunk_{index}",
            "locations": location_str,
            "document_type": _stable_choice(config.document_types, seed + "-type"),
            "department": _stable_choice(config.departments, seed + "-dept"),
            "confidentiality_level": _stable_choice(config.confidentiality_levels, seed + "-conf"),
            "author": _stable_choice(config.authors, seed + "-author"),
        }
        doc_id = f"{parent_id}__split_{index:05d}"
        documents.append(Document(doc_id=doc_id, text=text, metadata=metadata))
    return documents
