# WeightedRAG

Modular implementation of a domain-agnostic Retrieval-Augmented Generation (RAG) pipeline tuned for Matryoshka embeddings and weighted reranking. The system covers ingestion, chunking, multi-resolution embedding, multi-stage retrieval, optional graph-based reranking, and LLM answer generation.

## Features
- **Data ingestion** from text folders or JSONL corpora with metadata normalization.
- **Overlap-aware chunking** with configurable token windows.
- **Matryoshka embeddings** with deterministic hashing fallback when GPU models are unavailable.
- **Multi-stage vector index** (coarse → mid → fine) with optional FAISS acceleration.
- **Weighted scoring** that blends multi-dimensional cosine similarity with metadata priors (reliability, temporal, domain, structural).
- **Optional graph reranker** that boosts well-supported chunks via overlap-based centrality.
- **LLM generator** using Hugging Face models with a graceful fallback when models cannot be loaded.
- **Evaluation helpers** for retrieval (P@K, R@K, MRR, NDCG) and QA (EM/F1).

## Installation

Create a virtual environment and install the dependencies that match your runtime requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy transformers sentence-transformers faiss-cpu
```

If you only need the hashing fallback (e.g., for quick smoke tests without downloading models), NumPy is sufficient.

## Quick Start

1. Prepare your corpus as either a directory of UTF-8 `.txt` files or a `.jsonl` file with at least a `text` field (optional metadata fields are preserved).
2. Run the end-to-end script:

```bash
python scripts/run_pipeline.py \
  --source data/wiki_sample.jsonl \
  --question "Who discovered penicillin?" \
  --metadata '{"source":"wikipedia","reliability":"0.8"}'
```

The pipeline will ingest the corpus, chunk documents, build the multi-stage indexes, retrieve context for the question, and attempt to generate an answer. If an LLM model is not available locally, a context summary is returned instead.

## BEIR / Retrieval Evaluation

Use the evaluation script to score retrieval quality on any BEIR-style split (expects `corpus.jsonl`, `queries.jsonl`, and `qrels.tsv`):

```bash
python scripts/evaluate_retrieval.py \
  --dataset-root data/beir/fiqa \
  --config configs/fiqa.json \
  --ks 1,3,5,10 \
  --max-queries 100 \
  --save-results outputs/fiqa_metrics.json
```

The script:

- Loads the corpus into the pipeline, builds Matryoshka indexes, and retrieves per query.
- Computes Precision@K, Recall@K, NDCG@K, and MRR for the final fused ranking and every retrieval stage.
- Optionally dumps per-query details (ranked doc IDs and metrics) for further analysis.

You can reuse the same script for other datasets by pointing `--dataset-root` at a directory that follows the BEIR format.

## Custom Configuration

All components can be tuned through a JSON configuration file. Example:

```json
{
  "chunking": {
    "max_tokens": 220,
    "overlap_tokens": 30,
    "tokenizer_name": "google-bert/bert-base-uncased"
  },
  "embedding": {
    "model_name": "google/embedding-gemma-002",
    "device": "cuda",
    "truncate_dims": [128, 256, 512, 2048]
  },
  "retrieval": {
    "lambda_similarity": 0.65,
    "alpha_reliability": 0.15,
    "stages": [
      {"name": "coarse", "dimension": 256, "top_k": 300, "weight": 0.2},
      {"name": "mid", "dimension": 512, "top_k": 60, "weight": 0.3},
      {"name": "fine", "dimension": 2048, "top_k": 15, "weight": 0.5, "index_factory": "Flat"}
    ]
  },
  "generation": {
    "model_name": "meta-llama/Llama-2-13b-chat-hf",
    "max_new_tokens": 256
  },
  "use_graph_rerank": true
}
```

Save the configuration and pass it via `--config path/to/config.json`.

## Module Overview

- `weighted_rag.data.ingest`: file loaders and metadata normalization helpers.
- `weighted_rag.data.chunker`: overlap-aware chunker built on Hugging Face tokenizers (with fallback).
- `weighted_rag.embeddings.matryoshka`: embedding wrapper capable of truncating vectors at multiple dimensions.
- `weighted_rag.index.multi_index`: in-memory multi-stage vector store with optional FAISS acceleration.
- `weighted_rag.retrieval.weighted`: weighted scorer that merges similarities and metadata priors.
- `weighted_rag.retrieval.graph`: optional graph-based reranker for evidence consolidation.
- `weighted_rag.generation.generator`: prompt builder and text generation wrapper.
- `weighted_rag.evaluation.metrics`: retrieval and QA metric utilities.
- `weighted_rag.pipeline`: orchestrates the full ingestion → retrieval → generation loop.

## Next Steps

- Integrate real Matryoshka models (e.g., EmbeddingGemma or Voyage Context-3) and adjust `truncate_dims`.
- Swap the hashing fallback for GPU-accelerated embeddings and enable FAISS for production-scale corpora.
- Extend metadata scoring functions with domain- and recency-aware priors tailored to your corpus.
- Add evaluation scripts for Natural Questions and BEIR subsets to track retrieval and QA metrics over time.
