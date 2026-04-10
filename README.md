# RAG Framework Benchmark: LangChain vs LlamaIndex vs DSPy

A high-performance benchmarking platform comparing three production RAG frameworks across **450 queries**, evaluated with string metrics, BERTScore, and a cross-family LLM judge (Qwen3-14B judging Llama-3.1-8B outputs).

The central question: **does framework choice matter, and does it matter differently depending on the domain?**

It does — dramatically.

---

## Architecture

```
vLLM (port 8000) — Llama-3.1-8B-Instruct     (worker / answer generation)
vLLM (port 8001) — Qwen/Qwen3-14B             (judge / cross-family to avoid bias)

Python RAG servers (FastAPI, one per framework):
  port 8100 — LangChain   (Chroma + bge-m3 embeddings)
  port 8101 — LlamaIndex  (in-memory vector store + bge-m3)
  port 8102 — DSPy        (FAISS + bge-m3 + ChainOfThought)

Go orchestrator — fires concurrent requests to all three servers simultaneously
                  exposes Prometheus metrics on :9091

Observability stack (docker-compose):
  Prometheus  :9090  — scrapes vLLM + Go orchestrator metrics
  Grafana     :3000  — latency, throughput, KV cache dashboards
  Arize Phoenix :6006 — LLM quality traces via OpenTelemetry
```

**Why Go for orchestration?** Python's GIL prevents true concurrency. Goroutines dispatch queries to all three RAG servers simultaneously, keeping the GPU saturated. At RPS=5 with 3 servers, 15 requests are in-flight at peak — goroutines handle this with minimal overhead compared to Python async.

---

## Results

### Latency (single-query, no queue pressure)

| Framework | Retrieval | Generation |
|-----------|-----------|------------|
| LangChain | ~137ms | ~5,387ms |
| LlamaIndex | ~443ms | ~5,683ms |
| DSPy | ~135ms | ~14,240ms (ChainOfThought generates longer reasoning) |

### Quality (450 queries, 150 per framework)

| Metric | LangChain | LlamaIndex | DSPy |
|--------|-----------|------------|------|
| Token F1 | 0.455 | 0.441 | **0.498** |
| BERTScore F1 | 0.831 | 0.834 | **0.844** |
| Correctness (Qwen judge) | **0.544** | 0.506 | 0.456 |
| Faithfulness (Qwen judge) | **0.833** | 0.696 | 0.644 |
| Completeness (Qwen judge) | **0.513** | 0.474 | 0.424 |

### Per-Domain Breakdown — Where Rankings Flip

| Domain | LangChain | LlamaIndex | DSPy | Winner |
|--------|-----------|------------|------|--------|
| covidqa correctness | 0.595 | 0.576 | 0.559 | LangChain |
| techqa correctness | 0.631 | **0.701** | 0.609 | LlamaIndex |
| finqa correctness | 0.406 | 0.276 | 0.227 | LangChain (all collapse) |
| finqa Token F1 | 0.487 | 0.431 | **0.567** | DSPy |

### Adversarial Robustness

| Framework | Non-OOD Robustness | OOD Refusal Rate | Multi-hop | Contradictory |
|-----------|-------------------|-----------------|-----------|---------------|
| LangChain | **0.730** | **0.867** | 0.717 | **0.773** |
| LlamaIndex | 0.709 | 0.600 | 0.700 | 0.727 |
| DSPy | 0.710 | 0.200 | 0.703 | 0.727 |

OOD refusal rate = fraction of out-of-distribution questions where the framework correctly refused to answer instead of hallucinating.

### DSPy MIPROv2 Prompt Optimization

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| Token F1 | 0.483 | 0.483 | +0.001 |
| Correctness | 0.551 | 0.512 | **-0.039** |
| Faithfulness | 0.693 | 0.628 | **-0.065** |
| Completeness | 0.475 | 0.434 | **-0.041** |

---

## Key Findings

**1. LangChain wins overall quality**
Highest correctness (0.544), faithfulness (0.833), and completeness (0.513). Conservative retrieval avoids fabrication. Best default choice for production RAG where accuracy matters.

**2. finqa breaks everything**
Financial reasoning collapses across all frameworks — LlamaIndex correctness drops to 0.276, DSPy to 0.227. Financial QA requires precise numerical facts; verbose or approximate answers fail hard. Aggregate benchmarks hide this.

**3. Token F1 ≠ Quality**
DSPy has the highest token F1 (0.498) and BERTScore (0.844) but the lowest judge correctness (0.456). ChainOfThought generates verbose answers that hit keywords but are imprecise. String metrics systematically overrate verbose frameworks — making LLM-as-judge essential.

**4. LlamaIndex dominates techqa**
Correctness of 0.701 — the highest single-domain score in the benchmark. LlamaIndex's retrieval strategy suits structured technical documentation.

**5. MIPROv2 prompt optimization made DSPy worse**
All judge metrics dropped after optimization (correctness -0.039, faithfulness -0.065). Prompt optimization overfit to the training distribution and degraded on the test set. Automated prompt optimization doesn't always generalize.

**6. DSPy hallucinates on 80% of out-of-distribution questions**
OOD refusal rate: LangChain=0.867, LlamaIndex=0.600, DSPy=0.200. ChainOfThought reasoning makes DSPy confidently answer questions it shouldn't. LangChain's conservative retrieval makes it refuse when context is absent. For production systems where hallucination is costly, DSPy is the highest-risk choice.

**7. Cross-family judging matters**
Qwen3-14B (Alibaba) judges Llama-3.1-8B (Meta) outputs — different training lineage eliminates same-family favoritism. Both run locally on vLLM; evaluation costs near zero after instance startup.

---

## Evaluation Design

### Three complementary metrics

| Method | What it measures | Limitation |
|--------|-----------------|------------|
| Token F1 | Exact word overlap with ground truth | Penalizes correct paraphrases |
| BERTScore | Semantic similarity (distilbert-base) | Rewards verbose answers |
| Qwen3-14B judge | Correctness, faithfulness, completeness | Slower, LLM bias |

Running all three on the same 450 outputs reveals where they agree and where they disagree. The disagreement is itself a finding.

### Adversarial robustness
Four hard query types generated from the test set:
- **Multi-hop** — requires connecting info across multiple documents
- **Ambiguous** — underspecified, missing key context
- **Out-of-distribution** — plausibly related but not in corpus (correct refusal = good)
- **Contradictory** — frames question as if the ground truth is false

---

## Project Structure

```
├── src/
│   ├── langchain_rag/pipeline.py      # Chroma + bge-m3, LCEL chain
│   ├── llamaindex_rag/pipeline.py     # In-memory VectorStoreIndex + bge-m3
│   ├── dspy_rag/pipeline.py           # FAISS + ChainOfThought, LiteLLM cache fix
│   ├── rag_server.py                  # Unified FastAPI server (--framework flag)
│   └── evaluation/
│       ├── metrics.py                 # Token F1, BERTScore, LLM judge
│       ├── adversarial_agent.py       # Adversarial query gen + robustness eval
│       └── tracing.py                 # Arize Phoenix / OTel instrumentation
├── orchestrator/
│   ├── main.go                        # Go orchestrator, rate limiter, Prometheus
│   ├── run_servers.sh                 # Start all servers + run benchmark
│   └── generate_queries.py           # Balanced query sampling (per domain)
├── infra/
│   ├── grafana/dashboards/            # Grafana dashboard JSON
│   └── prometheus.yml
├── docker-compose.yml                 # Prometheus + Grafana + Arize Phoenix
├── run_eval.py                        # String + Qwen judge on results
├── run_eval_domains.py               # Per-domain eval (covidqa/techqa/finqa)
├── run_bertscore.py                  # BERTScore eval (local, no GPU)
├── run_dspy_optimized.py             # MIPROv2 baseline vs optimized comparison
└── setup_lambda.sh                   # Lambda Cloud GPU instance setup
```

---

## Running It

### On Lambda Cloud (H100 / A100 / GH200)

```bash
git clone https://github.com/sharle21/RAG-Framework-Comparison-LangChain-vs-LlamaIndex-vs-DSPy.git rag-bench
cd rag-bench && bash setup_lambda.sh <hf-token>

# Start vLLM (single GPU — 80GB)
source ~/vllm_env/bin/activate
nohup python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000 --gpu-memory-utilization 0.45 > /tmp/vllm_worker.log 2>&1 &
nohup python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --port 8001 --gpu-memory-utilization 0.45 --max-model-len 4096 > /tmp/vllm_judge.log 2>&1 &

# Run benchmark
export PATH="/usr/local/go/bin:$PATH"
RPS=5 WORKERS=8 bash orchestrator/run_servers.sh

# Evaluate
nohup python3 -u run_eval_domains.py > /tmp/eval_domains.log 2>&1 &
```

### With full observability stack

```bash
docker compose up -d phoenix prometheus grafana
TRACING=1 RPS=5 WORKERS=8 bash orchestrator/run_servers.sh
# Traces:  http://localhost:6006
# Metrics: http://localhost:3000
```

### BERTScore locally (no GPU)

```bash
pip install bert-score && python run_bertscore.py
```

---

## Bugs Found and Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| DSPy showing 4ms generation | LiteLLM disk cache active despite `cache=False` in `dspy.configure()` | Pass `cache=False` to `dspy.LM()` constructor directly |
| LangChain all queries failing | Chroma index built with OpenAI 1536-dim embeddings, queried with bge-m3 1024-dim | Delete stale index, rebuild |
| LlamaIndex `ValueError: Unknown model` | `OpenAI` class rejects custom `base_url` | Switch to `OpenAILike` |
| Qwen3 `<think>` blocks breaking JSON parse | Qwen3-14B outputs `<think>reasoning</think>` before JSON response | Strip with `re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)` |
| OOM on server startup | All three servers embedding corpus simultaneously | Sequential startup in `run_servers.sh` |

---

## References

- [RAGBench Dataset](https://huggingface.co/datasets/rungalileo/ragbench)
- [DSPy](https://github.com/stanfordnlp/dspy) — Stanford NLP
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention inference
- [MIPROv2](https://arxiv.org/abs/2406.11695) — Automatic prompt optimization
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — LLM observability
