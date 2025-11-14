"""High-level orchestration of the WeightedRAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import PipelineConfig
from .data.chunker import DocumentChunker
from .data.ingest import load_documents, normalize_metadata
from .embeddings.matryoshka import MatryoshkaEmbedder
from .generation.generator import LLMGenerator
from .index.multi_index import MultiStageVectorIndex, StageResult
from .retrieval.cross_encoder import CrossEncoderReranker
from .retrieval.graph import GraphReranker
from .retrieval.weighted import WeightedRetriever
from .types import Document, GenerationResult, Query, RetrievalResult


class WeightedRAGPipeline:
    """Coordinates ingestion, indexing, retrieval, and generation."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.chunker = DocumentChunker(config.chunking)
        self.embedder = MatryoshkaEmbedder(config.embedding, config.retrieval)
        self.index = MultiStageVectorIndex(config.retrieval)
        self.retriever = WeightedRetriever(config.retrieval, self.index)
        self.generator = LLMGenerator(config.generation)
        self.graph_reranker = GraphReranker() if config.use_graph_rerank else None
        self.cross_encoder_reranker = CrossEncoderReranker(config.cross_encoder) if config.cross_encoder else None
        self._chunk_count = 0

    def ingest_path(self, path: Path, **kwargs) -> List[Document]:
        documents = load_documents(path, **kwargs)
        return normalize_metadata(documents)

    def add_documents(self, documents: Sequence[Document]) -> int:
        chunks = self.chunker.chunk_corpus(documents)
        if not chunks:
            return 0
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = self.embedder.embed(ids, texts)
        stage_embeddings = {}
        for dim in {stage.dimension for stage in self.config.retrieval.stages}:
            slice_embedding = embeddings.slices.get(dim)
            if slice_embedding is not None:
                stage_embeddings[dim] = slice_embedding
            else:
                stage_embeddings[dim] = embeddings.vectors[:, :dim]
        self.index.add_chunks(chunks, stage_embeddings)
        self._chunk_count += len(chunks)
        return len(chunks)

    def retrieve(self, query: Query) -> RetrievalResult:
        retrieval, _ = self.retrieve_with_details(query)
        return retrieval

    def retrieve_with_details(self, query: Query) -> Tuple[RetrievalResult, Dict[str, StageResult]]:
        if self._chunk_count == 0:
            raise RuntimeError("No chunks indexed. Add documents before querying.")
        query_vectors = self.embedder.embed_query(query.text)
        stage_results = self.index.search(query_vectors)
        retrieval = self.retriever.retrieve(query, stage_results)
        if self.cross_encoder_reranker:
            reranked = self.cross_encoder_reranker.rerank(query, retrieval.chunks)
            retrieval = RetrievalResult(query=query, chunks=reranked)
        if self.graph_reranker:
            reranked = self.graph_reranker.rerank(retrieval.chunks)
            retrieval = RetrievalResult(query=query, chunks=reranked)
        return retrieval, stage_results

    def answer(self, query: Query) -> GenerationResult:
        retrieval = self.retrieve(query)
        return self.generator.generate(retrieval)

    def bulk_answer(self, queries: Iterable[Query]) -> List[GenerationResult]:
        return [self.answer(query) for query in queries]
