# WeightedRAG

Modular implementation of a domain-agnostic Retrieval-Augmented Generation (RAG) pipeline tuned for Matryoshka embeddings and weighted reranking. The system covers ingestion, chunking, multi-resolution embedding, multi-stage retrieval, optional graph-based reranking, and LLM answer generation.

## Features
- **Data ingestion** from text folders or JSONL corpora with metadata normalization.
- **Overlap-aware chunking** with configurable token windows.
- **Matryoshka embeddings** with deterministic hashing fallback when GPU models are unavailable.
- **Multi-stage vector index** (coarse → mid → fine) with optional FAISS acceleration.
- **Weighted scoring** that blends multi-dimensional cosine similarity with metadata priors (reliability, temporal, domain, structural).
- **Optional graph reranker** that boosts well-supported chunks via overlap-based centrality.
- **Enterprise PDF ingestion** using LangChain loaders with spaCy metadata enrichment (locations, departments, confidentiality).
- **Cross-encoder reranking** and metadata-aware filtering for query-specific restraints (e.g., location or department).
- **TextGrad-inspired prompt optimization** to automatically refine system prompts based on supervision examples.
- **TAPAS-backed table extraction** to normalize Natural Questions tables before chunking.
- **OpenRouter-compatible generation** whenever an API key is provided (fallbacks to local transformers otherwise).
- **LLM generator** using Hugging Face models with a graceful fallback when models cannot be loaded.
- **Evaluation helpers** for retrieval (P@K, R@K, MRR, NDCG) and QA (EM/F1).

## Installation

Create a virtual environment and install the dependencies that match your runtime requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
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

## Enterprise PDF Chatbot

To mirror the [Enterprise-Chatbot](https://github.com/CheerlaChandana/Enterprise-Chatbot) workflow, use the dedicated CLI:

```bash
python scripts/enterprise_chatbot.py \
  --pdf-dir path/to/pdfs \
  --question "Summarize device onboarding policies" \
  --locations "New York,London" \
  --filter department=Engineering,Operations \
  --save-chunks outputs/enterprise_chunks.json
```

The script will:

- Ingest PDFs via LangChain and split them into enterprise-flavored segments enriched with locations, document type, department, confidentiality, and author metadata.
- Index everything inside the WeightedRAG pipeline (token chunker + Matryoshka embeddings).
- Retrieve results with weighted scoring, optional metadata filters (prefix `filter__`), cross-encoder reranking, and optionally graph reranking.
- Display the retrieved chunks (mirroring the original script) and generate an answer grounded on those sources.

> **Note:** PDF ingestion relies on optional dependencies: `langchain`, `langchain-community`, `langchain-huggingface`, `spacy`, and the `en_core_web_sm` model. Install them when you need the enterprise workflow.

### Prompt Optimization with TextGrad

You can auto-tune the generation prompt before answering by supplying supervision examples:

```bash
python scripts/enterprise_chatbot.py \
  --pdf-dir path/to/pdfs \
  --question "List engineering onboarding policies" \
  --textgrad-examples configs/textgrad_examples.jsonl \
  --textgrad-steps 5 \
  --textgrad-history outputs/textgrad_trace.json
```

Each example needs at least `question` and `reference_answer` fields (JSON or JSONL). The optimizer iteratively appends candidate instructions (TextGrad mutations) and keeps only improvements that raise the mean F1 overlap with the reference answers. Set `--textgrad-sample-size` to subsample examples per step, or tweak `--textgrad-min-gain` / `--textgrad-mutations` for tighter control. The resulting prompt is persisted inside the current run and the optional history file.

## Environment Overrides (.env)

WeightedRAG reads a project-level `.env` file so you can swap critical models without editing code. Example:

```bash
model_name=meta-llama/Llama-2-13b-chat-hf
TAPAS_MODEL_NAME=google/tapas-large-finetuned-wtq
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_SITE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_NAME=WeightedRAG
```

- `model_name` selects the LLM used for generation (overrides `generation.model_name`).
- `TAPAS_MODEL_NAME` picks which TAPAS tokenizer powers table extraction inside the chunking pipeline.
- `OPENROUTER_API_KEY` enables hosted inference via OpenRouter (if unset, the generator falls back to local Hugging Face pipelines). `OPENROUTER_SITE_URL` and `OPENROUTER_SITE_NAME` are optional metadata OpenRouter uses for routing/attribution.

Exported shell variables still take precedence, so you can temporarily override either value via `export model_name=...`.

## Natural Questions Tables Evaluation

Use this workflow to regenerate the NQ Tables dataset, convert it to BEIR format, and log retrieval metrics.

1. **Download and decompress raw shards**
   ```bash
   mkdir -p datasets/nq-table/raw
   gsutil -m cp gs://natural_questions/v1.0/train/nq-train-00.jsonl.gz datasets/nq-table/raw/
   gunzip -k datasets/nq-table/raw/nq-train-00.jsonl.gz
   ```
2. **Extract table QA interactions**
   ```bash
   python chunking/utils/corrected_nq_table_extractor.py \
     --input datasets/nq-table/raw/nq-train-00.jsonl \
     --output datasets/nq-table/nq_table_full_extraction.jsonl \
     --sample 50000
   ```
3. **Generate multi-granular chunks**
   ```bash
   cd chunking/core
   python create_retrieval_tables.py --max-entries 50000
   cd ../..
   ```
4. **Convert to BEIR split**
   ```bash
   python scripts/prepare_nq_tables_beir.py \
     --tables retrieval_tables/processed_tables.jsonl \
     --chunks retrieval_tables/retrieval_chunks.jsonl \
     --output-dir datasets/nq-table/beir \
     --chunk-types table_row,table_sample,pure_table
   ```
5. **Run evaluation and append the log**
   ```bash
   python scripts/evaluate_retrieval.py \
     --dataset-root datasets/nq-table/beir \
     --ks 1,3,5,10 \
     --max-queries 1000 \
     --save-results outputs/nq_table_metrics.json | tee -a evaluation_log.txt
   ```

The detailed checklist lives in `evaluation_log.txt`. Each time you run the benchmark, append the stdout via `tee -a evaluation_log.txt` and verify the manual checks at the top of that file.

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
