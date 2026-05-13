# Production RAG Application

A production-grade Retrieval Augmented Generation system with hybrid retrieval, cross-encoder reranking, citation enforcement, and CI-gated evaluation.

## Features

- **Hybrid Retrieval**: BM25 keyword search + vector semantic search
- **Cross-Encoder Reranking**: Re-scores retrieved chunks for precision
- **Citation Enforcement**: Declines to answer when evidence is insufficient
- **Evaluation Pipeline**: Golden dataset with faithfulness metrics, CI-gated
- **Versioned Prompts**: Prompt configs stored alongside code

## Architecture

```
Documents → Ingestion → Chunking (500-800 tokens, 100 overlap)
                              ↓
                     Embedding + Indexing
                        ↓           ↓
                   ChromaDB      BM25 Index
                        ↓           ↓
              Query → Hybrid Retrieval → Reranker → LLM → Cited Answer
```

## Tech Stack

- **Orchestration**: LangChain / LangGraph
- **Vector Store**: ChromaDB
- **Keyword Search**: rank-bm25
- **Reranking**: sentence-transformers cross-encoder
- **Evaluation**: RAGAS
- **LLM**: OpenAI GPT-4o / Anthropic Claude

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Usage

```bash
# Ingest documents
python src/ingest.py --docs-dir data/documents/

# Query the system
python src/query.py "What is the refund policy?"

# Run evaluation
python src/evaluate.py
```

## Phases

### Phase 1: Core RAG
- Document ingestion (PDF, Markdown, HTML)
- Chunking with overlap
- Vector store + basic retrieval
- Answer generation with citations

### Phase 2: Production Quality
- Hybrid retrieval (BM25 + vector)
- Cross-encoder reranking
- Citation enforcement (decline when unsupported)
- Versioned prompt configs

### Phase 3: Evaluation & CI
- Golden eval dataset (50-200 QA pairs)
- Faithfulness scoring
- CI pipeline with quality gating
