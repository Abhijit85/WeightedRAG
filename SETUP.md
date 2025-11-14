# WeightedRAG Setup Guide

This guide walks you through setting up the WeightedRAG pipeline from scratch, including data preparation, chunking, and evaluation.

## 📋 Prerequisites

- Python 3.8+
- Git
- Google Cloud SDK (optional, for downloading NQ dataset)
- At least 8GB RAM for processing
- ~10GB disk space for full dataset

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Abhijit85/WeightedRAG.git
cd WeightedRAG

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### 2. Install Package in Development Mode

```bash
# This enables imports from the weighted_rag module
pip install -e .
```

> **Note:** The `pip install -e .` command uses the `setup.py` file in the root directory to install the WeightedRAG package in "editable" mode. This allows you to make changes to the code without reinstalling.

## 📊 Data Preparation

### Option A: Download Natural Questions Dataset (Recommended)

```bash
# Create data directories
mkdir -p datasets/nq-table/raw

# Download NQ training files (requires gsutil)
gsutil -m cp gs://natural_questions/v1.0/train/nq-train-*.jsonl.gz datasets/nq-table/raw/

# Decompress files
for file in datasets/nq-table/raw/*.jsonl.gz; do
  gunzip -k "$file"
done
```

### Option B: Use Sample Data (For Testing)

```bash
# Extract sample from first shard only
python chunking/utils/corrected_nq_table_extractor.py \
  --input datasets/nq-table/raw/nq-train-00.jsonl \
  --output datasets/nq-table/nq_table_full_extraction.jsonl \
  --sample 1000
```

## 🔄 Pipeline Execution

### Step 1: Extract Table QA Pairs

```bash
# Extract table-centric Q&A pairs from Natural Questions
python chunking/utils/corrected_nq_table_extractor.py \
  --input datasets/nq-table/raw/nq-train-00.jsonl \
  --output datasets/nq-table/nq_table_full_extraction.jsonl \
  --sample 50000
```

**Output:** `datasets/nq-table/nq_table_full_extraction.jsonl`

### Step 2: Generate Multi-Granular Chunks

```bash
# Create retrieval chunks with different granularities
cd chunking/core
python create_retrieval_tables.py --max-entries 50000
cd ../..
```

**Outputs:**
- `retrieval_tables/processed_tables.jsonl`
- `retrieval_tables/retrieval_chunks.jsonl`
- `retrieval_tables/chunks_*.jsonl` (per chunk type)

### Step 3: Convert to BEIR Format

```bash
# Transform to evaluation-ready format
python scripts/prepare_nq_tables_beir.py \
  --tables retrieval_tables/processed_tables.jsonl \
  --chunks retrieval_tables/retrieval_chunks.jsonl \
  --output-dir datasets/nq-table/beir \
  --chunk-types full_table,table_only,table_row,sliding_window,table_sample,pure_table
```

**Outputs:**
- `datasets/nq-table/beir/corpus.jsonl`
- `datasets/nq-table/beir/queries.jsonl`
- `datasets/nq-table/beir/qrels.tsv`

### Step 4: Run Retrieval Evaluation

```bash
# Evaluate retrieval performance
python scripts/evaluate_retrieval.py \
  --dataset-root datasets/nq-table/beir \
  --ks 1,3,5,10 \
  --max-queries 500 \
  --save-results outputs/nq_table_metrics.json
```

**Outputs:**
- Console metrics summary
- `outputs/nq_table_metrics.json` (detailed results)

## 🏗️ Architecture Overview

### Key Components

1. **Document Chunking**: Multi-granular table chunking
   - Full tables
   - Individual rows
   - Sliding windows
   - Table samples

2. **Embedding**: Matryoshka embeddings with multiple dimensions
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Dimensions: 384 (configurable)

3. **Indexing**: FAISS HNSW for efficient similarity search
   - Index type: `HNSW32`
   - Approximate nearest neighbor search

4. **Retrieval**: Multi-stage weighted retrieval
   - Dense vector similarity
   - Metadata-based scoring
   - Optional cross-encoder reranking
   - Optional graph-based reranking

### Pipeline Flow

```
Raw Data → Extract Tables → Generate Chunks → Embed → Index → Retrieve → Rank → Generate
```

## ⚙️ Configuration

### Basic Configuration

Create `config.json` for custom settings:

```json
{
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 16,
    "device": "cpu"
  },
  "retrieval": {
    "stages": [
      {
        "name": "dense",
        "dimension": 384,
        "top_k": 100,
        "weight": 1.0,
        "index_factory": "HNSW32"
      }
    ]
  },
  "chunking": {
    "max_tokens": 128,
    "overlap_tokens": 16
  }
}
```

### Enable Optional Features

```json
{
  "use_graph_rerank": true,
  "cross_encoder": {
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "batch_size": 32
  }
}
```

## 🧪 Testing Your Setup

### Quick Verification

```bash
# Test table extraction
python chunking/utils/corrected_nq_table_extractor.py \
  --input datasets/nq-table/raw/nq-train-00.jsonl \
  --output test_extraction.jsonl \
  --sample 10

# Test chunking
cd chunking/core
python create_retrieval_tables.py --max-entries 10
cd ../..

# Test evaluation (small scale)
python scripts/evaluate_retrieval.py \
  --dataset-root datasets/nq-table/beir \
  --ks 1,3,5 \
  --max-queries 5
```

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Run `pip install -e .` from repo root
2. **Memory issues**: Reduce batch sizes or use smaller sample sizes
3. **FAISS errors**: Install `faiss-cpu` or `faiss-gpu`
4. **Slow processing**: Use GPU if available, reduce dataset size

### Performance Tips

- Use GPU for embedding computation
- Increase batch sizes if you have more RAM
- Use FAISS GPU index for large datasets
- Process in smaller chunks if memory constrained

## 📁 Project Structure

```
WeightedRAG/
├── src/weighted_rag/          # Main package
│   ├── embeddings/            # Embedding models
│   ├── index/                 # Vector indexes
│   ├── retrieval/             # Retrieval algorithms
│   └── evaluation/            # Metrics and evaluation
├── chunking/                  # Table chunking logic
│   ├── core/                  # Main chunking algorithms
│   ├── processors/            # Data processors
│   └── utils/                 # Utility functions
├── scripts/                   # Evaluation scripts
├── datasets/                  # Data storage (gitignored)
├── retrieval_tables/          # Generated chunks (gitignored)
├── outputs/                   # Results and metrics
├── setup.py                   # Package installation configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

### Key Files

- **setup.py**: Defines the package structure and dependencies for `pip install -e .`
- **requirements.txt**: Lists all Python dependencies needed for the project
- **SETUP.md**: This setup guide
- **evaluation_log.txt**: Detailed logs from evaluation runs

## 📊 Expected Results

### Retrieval Metrics

After successful setup, you should see metrics like:

```
P@1: 0.25-0.35
P@5: 0.15-0.25
P@10: 0.10-0.20
Recall@10: 0.30-0.50
NDCG@10: 0.30-0.45
MRR: 0.30-0.40
```

### Processing Times

- Table extraction: ~5-10 minutes (50k samples)
- Chunking: ~10-15 minutes (50k samples)
- BEIR conversion: ~2-3 minutes
- Evaluation: ~15-30 minutes (500 queries)

## 🔄 Next Steps

1. **Scale Up**: Process full dataset instead of samples
2. **Tune Parameters**: Adjust chunking, embedding, and retrieval settings
3. **Add Features**: Enable cross-encoder or graph reranking
4. **Custom Data**: Adapt pipeline for your own table datasets
5. **Production**: Deploy with proper error handling and monitoring

## 📚 Additional Resources

- [Original README.md](README.md) - Technical details
- [Evaluation Log](evaluation_log.txt) - Detailed logs
- [Chunking README](chunking/README.md) - Chunking specifics

## 🐛 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the evaluation log for detailed error messages
3. Ensure all dependencies are installed correctly
4. Verify data paths and file permissions
5. Open an issue with detailed error information

## 🎯 Success Criteria

Your setup is working correctly if:

- [ ] All scripts run without errors
- [ ] Generated files exist in expected locations
- [ ] Evaluation produces reasonable metric values
- [ ] Processing times are within expected ranges
- [ ] No import or dependency errors occur