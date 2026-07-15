# RAG Framework Benchmark: LangChain vs LlamaIndex vs DSPy

A high-performance benchmarking platform comparing three production RAG frameworks across **450 queries**, evaluated with Token F1, ROUGE, BERTScore, and a cross-family LLM judge (Qwen3-14B judging Llama-3.1-8B outputs). All quality differences reported with 95% bootstrap confidence intervals and Mann-Whitney significance tests.

The central question: **does framework choice matter, and does it matter differently depending on the domain?**

It does — but which framework "wins" depends entirely on which metric you trust.

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

### Latency (150 queries per framework, concurrent benchmark)

| Framework | Retrieval median | Retrieval p95 | Generation median | Generation p95 |
|-----------|-----------------|---------------|-------------------|----------------|
| LangChain | 117ms | 197ms | 1,635ms | 12,096ms |
| LlamaIndex | 387ms | 567ms | **1,130ms** | 19,105ms |
| DSPy | 119ms | 192ms | 3,580ms | 60,262ms |

**Use median, not mean.** A few GPU queue spikes (p99 > 170s) drag the mean to 5–14s and make LlamaIndex appear slowest. At median, LlamaIndex generation is actually fastest. Mean numbers are in `results/stats_with_ci.json` for reference but should not be cited.

**Caveat:** Generation times measured under concurrent load — all 3 frameworks share one vLLM endpoint. Queue wait is included in generation_ms. A serial single-framework benchmark (Lambda TODO L6) is needed for clean isolation. DSPy's higher median is real (ChainOfThought generates 3–4× more tokens) but the absolute numbers will change with serial measurement. ms/token normalization pending Lambda rerun (TODO L4).

### Quality (450 queries, 150 per framework)

| Metric | LangChain | LlamaIndex | DSPy | Significant? |
|--------|-----------|------------|------|-------------|
| Token F1 | 0.455 [0.425, 0.485] | 0.441 [0.414, 0.469] | **0.498** [0.468, 0.527] | DSPy > others (p<0.05) |
| ROUGE-1 | 0.425 [0.395, 0.460] | 0.414 [0.384, 0.446] | **0.453** [0.422, 0.484] | — |
| ROUGE-L | 0.343 [0.315, 0.376] | 0.328 [0.302, 0.357] | **0.384** [0.353, 0.416] | — |
| BERTScore F1 | 0.831 | 0.834 | **0.844** | — |
| Correctness (Qwen judge) | **0.541** [0.482, 0.599] | 0.506 [0.449, 0.564] | 0.456 [0.394, 0.517] | LangChain > DSPy (p<0.05); LC vs LI: ns |
| Faithfulness (Qwen judge) | **0.828** [0.782, 0.874] | 0.696 [0.640, 0.752] | 0.644 [0.583, 0.706] | LangChain > both (p<0.001) |
| Completeness (Qwen judge) | **0.510** [0.454, 0.567] | 0.474 [0.420, 0.528] | 0.424 [0.367, 0.481] | LangChain > DSPy (p<0.05); rest: ns |

CIs are 95% bootstrap (seed=42, n=1000). Significance: Mann-Whitney U + permutation test, both p<0.05 required. `ns` = not significant.

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

**1. String metrics and LLM judge point in opposite directions**
Every string metric (Token F1, ROUGE-1, ROUGE-L, BERTScore) ranks DSPy #1. Every judge metric (correctness, faithfulness, completeness) ranks LangChain #1. Spearman correlation between string and judge metrics: -0.5 to -1.0. They are measuring different things. Which metric you trust determines which framework you ship.

**2. LangChain faithfulness is the only statistically solid claim**
After Mann-Whitney U + permutation testing: LangChain faithfulness beats both LlamaIndex and DSPy at p<0.001. All other pairwise correctness differences are either marginal (p<0.05 with small effect) or not significant. LangChain vs LlamaIndex on correctness: p=0.42, not significant.

**3. finqa breaks everything**
Financial reasoning collapses across all frameworks — LlamaIndex correctness drops to 0.276, DSPy to 0.227. Financial QA requires precise numerical facts; verbose or approximate answers fail hard. Aggregate benchmarks hide this.

**4. LlamaIndex generation is fastest (at median)**
At median, LlamaIndex generation is 1,130ms vs LangChain 1,635ms vs DSPy 3,580ms. The mean (5,683ms) was an artifact of GPU queue spikes — a few p99 outliers (>170s) dominated. Always report median for latency under concurrent load.

**5. LlamaIndex dominates techqa**
Correctness of 0.701 — the highest single-domain score in the benchmark. LlamaIndex's retrieval strategy suits structured technical documentation.

**6. MIPROv2 prompt optimization made DSPy worse**
All judge metrics dropped after optimization (correctness -0.039, faithfulness -0.065). Prompt optimization overfit to the training distribution and degraded on the test set. Automated prompt optimization doesn't always generalize.

**7. DSPy hallucinates on OOD questions (caveat: n=30)**
OOD refusal rate: LangChain=0.867, LlamaIndex=0.600, DSPy=0.200. ChainOfThought reasoning makes DSPy confidently answer questions it shouldn't. Caveat: 30 examples per framework gives ±14% CI — directionally real but not precise. Expanding to n=150 is in progress.

**8. Cross-family judging matters**
Qwen3-14B (Alibaba) judges Llama-3.1-8B (Meta) outputs — different training lineage eliminates same-family favoritism. Both run locally on vLLM; evaluation costs near zero after instance startup.

---

## Evaluation Design

### Five complementary metrics

| Method | What it measures | Limitation | DSPy rank | LangChain rank |
|--------|-----------------|------------|-----------|----------------|
| Token F1 | Exact word overlap | Rewards verbosity | **1** | 2 |
| ROUGE-1/L | N-gram overlap | Rewards verbosity | **1** | 2 |
| BERTScore | Semantic similarity (distilbert-base) | Rewards verbosity | **1** | 3 |
| Qwen3-14B judge correctness | Factual accuracy vs ground truth | LLM bias, slow | 3 | **1** |
| Qwen3-14B judge faithfulness | Grounded in retrieved context | LLM bias, slow | 3 | **1** |

String metrics consistently rank DSPy #1. Judge metrics consistently rank LangChain #1. The ranking reversal is the central finding. Running all five on the same outputs makes the disagreement visible and measurable.

All pairwise differences tested with Mann-Whitney U + permutation test (n=10,000). 95% bootstrap CIs on all means (seed=42).

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
├── run_rouge.py                      # ROUGE-1/2/L eval (local, no GPU)
├── run_statistical_tests.py          # Mann-Whitney + permutation tests (local)
├── run_metric_comparison.py          # Cross-metric ranking + correlation (local)
├── compute_stats_local.py            # Bootstrap CIs + latency percentiles (local)
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
