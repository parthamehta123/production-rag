# AI Portfolio Projects - Master Guide

5 production-level AI engineering projects targeting distinct, in-demand skills.

| # | Project | Directory | Key Skill |
|---|---------|-----------|-----------|
| 1 | Production RAG | `production-rag/` | Retrieval systems, evaluation, CI gating |
| 2 | Local SLM Benchmark | `local-slm-benchmark/` | Local inference, model comparison, structured output |
| 3 | RAG Observability | `rag-observability/` | Monitoring, tracing, latency tracking |
| 4 | Fine-Tuning LoRA+DPO | `fine-tuning-lora-dpo/` | SFT, preference tuning, training pipelines |
| 5 | Voice Assistant | `realtime-voice-assistant/` | Streaming, latency budgets, resilience |

---

## How to Study the Codebase

### Recommended Order

**Start with Project 1 → 3 → 2 → 4 → 5** (not 1-2-3-4-5)

Project 1 (RAG) is the foundation. Project 3 (Observability) builds directly on it.
Project 2 (Local SLM) and 4 (Fine-tuning) are independent. Project 5 ties everything together.

### Project 1: Production RAG

Read in this order:

```
1. configs/prompts.yaml          ← Start here. Understand the domain (SEC 10-K filings)
2. src/ingest.py                 ← How documents get loaded, chunked, and embedded
3. src/retriever.py              ← Core logic: BM25 + vector search + reranking
4. src/query.py                  ← How retrieval connects to LLM with citation enforcement
5. src/evaluate.py               ← RAGAS evaluation pipeline + quality gating
6. eval/golden_dataset.json      ← The 20 finance QA pairs used for evaluation
7. data/documents/               ← Real SEC 10-K filings (AAPL, MSFT, TSLA, JPM, GS)
8. .github/workflows/ci.yml      ← How evaluation gates PRs
```

Key concepts to understand:
- Why hybrid retrieval (BM25 + vector) beats vector-only
- How cross-encoder reranking improves precision
- Why citation enforcement matters (lines 46-60 in query.py)
- How RAGAS faithfulness scoring works

### Project 2: Local SLM Benchmark

```
1. src/inference.py              ← Core: how to measure TPS, TTFT, memory
2. src/structured_output.py      ← JSON schema enforcement + Pydantic validation + retry
3. src/benchmark.py              ← Multi-model comparison framework
```

Key concepts:
- Token-level streaming measurement
- Constrained generation → validation → retry pattern
- Temperature variance analysis (same prompt, different temps)

### Project 3: RAG Observability

```
1. src/tracing.py                ← Langfuse instrumentation + PipelineMetrics dataclass
2. src/metrics_collector.py      ← Prometheus metrics + percentile computation
3. src/regression_eval.py        ← Quality gating with thresholds
4. dashboards/dashboard.py       ← Streamlit dashboard for visualization
```

Key concepts:
- Why P50/P95 matter more than averages
- Cost-per-request tracking
- How regression gating prevents quality degradation

### Project 4: Fine-Tuning LoRA+DPO

```
1. data/function_definitions.json     ← The 5 financial tools the model learns to call
2. src/prepare_data.py                ← Training data format (SFT + DPO pairs)
3. configs/sft_config.yaml            ← QLoRA config (read every comment)
4. configs/sft_lora_fullprecision.yaml ← LoRA config (compare with QLoRA)
5. src/train_sft.py                   ← SFT training pipeline
6. configs/dpo_config.yaml            ← DPO config (stacks on SFT)
7. src/train_dpo.py                   ← DPO preference training
8. src/evaluate.py                    ← Function-calling accuracy metrics
9. src/download_dataset.py            ← Glaive dataset downloader (113K examples)
```

Key concepts:
- LoRA vs QLoRA tradeoffs (VRAM, quality, speed)
- Why target_modules includes both attention AND MLP layers
- SFT → DPO stacking (DPO builds on SFT checkpoint)
- Chosen vs rejected pairs for preference learning

### Project 5: Real-Time Voice Assistant

```
1. src/latency_tracker.py        ← Latency budget system (targets per component)
2. src/server.py                 ← WebSocket pipeline: ASR → LLM → TTS
3. src/latency_analysis.py       ← Visualization of latency breakdowns
```

Key concepts:
- Why decomposing latency into ASR/LLM/TTS budgets matters
- Graceful degradation (timeout handling, fallback responses)
- WebSocket streaming for real-time bidirectional communication

---

## Running the Projects

### Prerequisites

```bash
# Python 3.12+
python3 --version

# Ollama (for Project 2)
curl -fsSL https://ollama.com/install.sh | sh

# Docker (optional)
docker --version
```

---

### Project 1: Production RAG

#### Without Docker

```bash
cd ~/production-rag

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
# Edit .env → add your OPENAI_API_KEY

# Step 1: Ingest the SEC 10-K documents
python src/ingest.py --docs-dir data/documents/

# Step 2: Query the system
python src/query.py "What are Apple's main risk factors?"
python src/query.py "How does Tesla describe competition in EVs?"
python src/query.py "What are JPMorgan's business segments?"

# Step 3: Run evaluation
python src/evaluate.py

# Run tests
pytest tests/ -v
```

#### With Docker

```bash
cd ~/production-rag

# Edit .env with your OPENAI_API_KEY first
docker build -t production-rag .

# Run tests
docker run --env-file .env production-rag

# Run ingestion
docker run --env-file .env -v $(pwd)/data:/app/data production-rag \
  python src/ingest.py --docs-dir data/documents/

# Run a query
docker run --env-file .env -v $(pwd)/data:/app/data production-rag \
  python src/query.py "What competitive risks does Apple face?"
```

---

### Project 2: Local SLM Benchmark

#### Without Docker

```bash
cd ~/local-slm-benchmark

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Pull models with Ollama
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b

# Step 1: Test single model inference
python src/inference.py --model llama3.2:3b --prompt "Explain the CAP theorem"

# Step 2: Test structured output with validation
python src/structured_output.py --model llama3.2:3b

# Step 3: Run full benchmark (all 3 models)
python src/benchmark.py
# Results saved to reports/benchmark_results.json

# Run tests
pytest tests/ -v
```

#### With Docker

```bash
cd ~/local-slm-benchmark
docker build -t local-slm-benchmark .

# Run tests (mocked, no Ollama needed)
docker run local-slm-benchmark

# Run with Ollama (from project root)
cd ~
docker compose up ollama -d
# Wait for Ollama to start, then pull models:
docker exec ollama ollama pull llama3.2:3b
docker exec ollama ollama pull phi4-mini
docker exec ollama ollama pull mistral:7b

# Run benchmark against Ollama container
docker compose run local-slm-benchmark python src/benchmark.py
```

---

### Project 3: RAG Observability

#### Without Docker

```bash
cd ~/rag-observability

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env

# Option A: Use Langfuse Cloud (easiest)
# Sign up at https://langfuse.com, get keys, add to .env

# Option B: Self-host Langfuse
# docker run -d -p 3000:3000 langfuse/langfuse

# Step 1: Run tests to generate sample metrics
pytest tests/ -v

# Step 2: Launch the Streamlit dashboard
streamlit run dashboards/dashboard.py
# Opens http://localhost:8501

# Step 3: Run regression evaluation
python src/regression_eval.py

# Run tests
pytest tests/ -v
```

#### With Docker

```bash
cd ~/rag-observability
docker build -t rag-observability .

# Run tests
docker run rag-observability

# Run dashboard
docker run -p 8501:8501 rag-observability \
  streamlit run dashboards/dashboard.py --server.address 0.0.0.0
# Open http://localhost:8501
```

---

### Project 4: Fine-Tuning LoRA+DPO

#### Without Docker

```bash
cd ~/fine-tuning-lora-dpo

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Step 1: Prepare the local training data
python src/prepare_data.py
# Creates data/sft_train.jsonl, sft_eval.jsonl, dpo_train.jsonl, dpo_eval.jsonl

# Step 2 (optional): Download full Glaive dataset (113K examples)
python src/download_dataset.py --max-examples 5000

# Step 3: Run SFT with QLoRA (needs GPU, ~6GB VRAM)
python src/train_sft.py --config configs/sft_config.yaml

# Step 4: Run SFT with LoRA full precision (needs GPU, ~16GB VRAM)
python src/train_sft.py --config configs/sft_lora_fullprecision.yaml

# Step 5: Run DPO on top of SFT checkpoint
python src/train_dpo.py --config configs/dpo_config.yaml

# Step 6: Evaluate before vs after
python src/evaluate.py --base-model qwen3:8b --finetuned-model ./models/sft-lora

# Run tests (no GPU needed)
pytest tests/ -v
```

**No GPU?** Use one of these:
- Google Colab (free T4 GPU)
- Lambda Labs / RunPod / Vast.ai (rent A100 for ~$1/hr)
- Fireworks AI (managed fine-tuning API)

#### With Docker

```bash
cd ~/fine-tuning-lora-dpo
docker build -t fine-tuning .

# Run tests
docker run fine-tuning

# Run data preparation
docker run -v $(pwd)/data:/app/data fine-tuning python src/prepare_data.py

# Run training (needs nvidia-docker for GPU)
docker run --gpus all -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data fine-tuning \
  python src/train_sft.py --config configs/sft_config.yaml
```

---

### Project 5: Real-Time Voice Assistant

#### Without Docker

```bash
cd ~/realtime-voice-assistant

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
# Edit .env → add OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY

# Step 1: Start the server
python src/server.py
# Opens http://localhost:8000

# Step 2: Check metrics
curl http://localhost:8000/metrics

# Step 3: Generate latency analysis (after some requests)
python src/latency_analysis.py
# Creates reports/latency_breakdown.png and percentile_comparison.png

# Run tests
pytest tests/ -v
```

#### With Docker

```bash
cd ~/realtime-voice-assistant
docker build -t voice-assistant .

# Run tests
docker run voice-assistant

# Run the server
docker run --env-file .env -p 8000:8000 voice-assistant \
  python src/server.py
# Open http://localhost:8000
```

---

### Run All 5 with Docker Compose

```bash
cd ~

# Run all tests across all projects
docker compose up --build

# Run specific project
docker compose up production-rag
docker compose up local-slm-benchmark
docker compose up rag-observability
docker compose up fine-tuning
docker compose up voice-assistant

# Run in background
docker compose up -d

# Check logs
docker compose logs -f production-rag

# Stop everything
docker compose down
```

---

## API Keys Needed

| Project | Keys Required |
|---------|--------------|
| 1. Production RAG | `OPENAI_API_KEY` (for embeddings + LLM) |
| 2. Local SLM | None (runs locally via Ollama) |
| 3. Observability | `OPENAI_API_KEY` + `LANGFUSE_*` keys (optional) |
| 4. Fine-Tuning | None for local training; `WANDB_API_KEY` optional for logging |
| 5. Voice Assistant | `OPENAI_API_KEY` + `DEEPGRAM_API_KEY` + `ELEVENLABS_API_KEY` |

Tests for all projects run without any API keys (all external calls are mocked).
