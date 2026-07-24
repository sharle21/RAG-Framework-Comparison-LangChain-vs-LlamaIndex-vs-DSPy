# RAG Framework Benchmark: LangChain vs LlamaIndex vs DSPy

Benchmarking three production RAG frameworks across **450 queries** on three domains (covidqa, techqa, finqa). Evaluated with Token F1, ROUGE, BERTScore, semantic similarity, pairwise preference, and a cross-family LLM judge. All quality differences reported with 95% bootstrap confidence intervals and Mann-Whitney significance tests.

The central question: **does framework choice matter, and does it matter differently depending on the domain?**

It does — but which framework "wins" depends entirely on which metric you trust.

---

## 1. Framework Comparison

All three frameworks use the **same base LLM** (Llama-3.1-8B-Instruct via vLLM), the **same embedding model** (bge-m3, 1024-dim), and the **same 450-question test set**. The RAG framework is the intended variable — but each pipeline was built idiomatically, not execution-graph-controlled, so retrieval *granularity* also differs: LangChain splits documents into ~1000-char chunks (`chunk_overlap=200`), LlamaIndex into ~2000-char chunks, and DSPy indexes each source passage whole with no splitter (median 3081 chars, up to ~51k chars). Same corpus, three different retrieval units — worth keeping in mind when comparing retrieval-quality numbers across frameworks.

| | LangChain | LlamaIndex | DSPy |
|-|-----------|------------|------|
| **Vector store** | Chroma (persistent SQLite) | In-memory VectorStoreIndex | FAISS |
| **Retrieval** | Similarity search, top-k | Similarity search, top-k | Similarity search, top-k |
| **Reasoning** | Standard RAG — retrieve then generate | Standard RAG — retrieve then generate | ChainOfThought — explicit reasoning chain before answer |
| **Prompt control** | LCEL chain, manual prompt | Query engine, internal prompt | DSPy signature + optimizer |
| **Index persistence** | Disk (survives restart) | In-memory (rebuilds on restart) | FAISS file + corpus JSON |
| **Token output** | Concise | Concise | Verbose (CoT adds reasoning tokens) |

**What ChainOfThought changes:** DSPy generates a reasoning chain before the final answer. This produces longer outputs, higher token F1 (more overlap chance), but also more hallucination risk on OOD questions where the chain fabricates steps.

---

## 2. Benchmark Pipeline

```
vLLM (port 8000) — Llama-3.1-8B-Instruct     (worker / answer generation)
vLLM (port 8001) — Qwen/Qwen3-14B             (judge / cross-family to reduce same-model self-preference)

Python RAG servers (FastAPI, one per framework):
  port 8100 — LangChain   (Chroma + bge-m3)
  port 8101 — LlamaIndex  (in-memory + bge-m3)
  port 8102 — DSPy        (FAISS + bge-m3 + ChainOfThought)

Go orchestrator — fires concurrent requests to all three servers simultaneously
                  exposes Prometheus metrics on :9091

Observability stack (docker-compose):
  Prometheus  :9090  — scrapes vLLM + Go orchestrator metrics
  Grafana     :3000  — latency, throughput, KV cache dashboards
  Arize Phoenix :6006 — LLM quality traces via OpenTelemetry
```

**Why Go for orchestration?** Python's GIL prevents true concurrency. Goroutines dispatch queries to all three RAG servers simultaneously, keeping the GPU saturated. At RPS=5 with 3 servers, 15 requests are in-flight at peak.

**What's controlled:** same LLM, same embeddings, same question set, same judge, same hardware.
**What varies:** framework's retrieval implementation, vector store, and prompting strategy.

**Latency caveat:** generation times measured under concurrent load — all 3 frameworks share one vLLM endpoint. Queue wait is included. DSPy's higher latency is real (more tokens), but absolute numbers are inflated by concurrent GPU pressure.

### Latency (150 queries per framework, concurrent)

| Framework | Retrieval median | Retrieval p95 | Generation median | Generation p95 |
|-----------|-----------------|---------------|-------------------|----------------|
| LangChain | 117ms | 197ms | 1,635ms | 12,096ms |
| LlamaIndex | 387ms | 567ms | **1,130ms** | 19,105ms |
| DSPy | **63ms** | — | 3,580ms | 60,262ms |

At median, LlamaIndex generation is fastest. LangChain and DSPy have similar retrieval speed (Chroma and FAISS). LlamaIndex's slower retrieval reflects in-memory index rebuild on startup.

---

## 3. Results

### Quality (450 queries, 150 per framework)

| Metric | LangChain | LlamaIndex | DSPy | Winner |
|--------|-----------|------------|------|--------|
| Token F1 | 0.461 | 0.462 | **0.488** | DSPy |
| ROUGE-1 | 0.425 [0.395, 0.460] | 0.414 [0.384, 0.446] | **0.453** [0.422, 0.484] | DSPy |
| ROUGE-L | 0.343 [0.315, 0.376] | 0.328 [0.302, 0.357] | **0.384** [0.353, 0.416] | DSPy |
| BERTScore F1 | 0.831 | 0.834 | **0.844** | DSPy |
| Semantic Sim (bge-m3) | **0.830** | 0.816 | 0.823 | LangChain |
| Context Coverage | 0.646 | 0.721 | **0.734** | DSPy |
| Correctness (Qwen judge) | **0.592** [0.532, 0.650] | 0.559 [0.499, 0.620] | 0.488 [0.416, 0.560] | LangChain |
| Faithfulness (Qwen judge) | **0.825** [0.778, 0.870] | 0.712 [0.652, 0.767] | 0.648 [0.578, 0.715] | LangChain (p<0.001) |
| Completeness (Qwen judge) | **0.550** [0.485, 0.611] | 0.505 [0.447, 0.565] | 0.422 [0.362, 0.485] | LangChain |
| Judge ECE (↓ better) | **0.318** | 0.332 | 0.413 | LangChain |

CIs are 95% bootstrap (seed=42, n=1000). Significance: Mann-Whitney U + permutation test, both p<0.05.

- **Context Coverage** = fraction of ground-truth tokens in retrieved passages. DSPy/LlamaIndex retrieve more relevant content yet LangChain wins all judge metrics — generation quality from context matters more than raw coverage.
- **Judge ECE** (Expected Calibration Error) = gap between Qwen's stated confidence and actual correctness. Lower = better calibrated.

### Pairwise Preference (143 questions, Qwen3-14B judge)

Judge picks the better answer directly without numeric scoring — avoids scale anchoring bias.

| Matchup | Winner | Score |
|---------|--------|-------|
| LangChain vs LlamaIndex | **LangChain** | 87 – 51 |
| LangChain vs DSPy | **LangChain** | 85 – 51 |
| LlamaIndex vs DSPy | **LlamaIndex** | 82 – 52 |

**Total wins:** LangChain 172, LlamaIndex 133, DSPy 103. LangChain wins every head-to-head (~63% win rate). Pairwise and absolute judge scores agree.

### Per-Domain Breakdown

| Domain | Metric | LangChain | LlamaIndex | DSPy |
|--------|--------|-----------|------------|------|
| **covidqa** | F1 | 0.397 | 0.413 | **0.454** |
| | correctness | **0.721** | 0.649 | 0.681 |
| | faithfulness | **0.920** | 0.835 | 0.887 |
| **techqa** | F1 | 0.482 | **0.485** | 0.456 |
| | correctness | **0.703** | 0.687 | 0.630 |
| | faithfulness | **0.832** | 0.733 | 0.659 |
| **finqa** | F1 | 0.504 | 0.490 | **0.553** |
| | correctness | 0.696 | 0.690 | **0.840** |
| | faithfulness | 0.843 | 0.833 | **0.839** |

DSPy finqa correctness (0.840) is the highest single-domain score in the benchmark — 14 points above LangChain. DSPy collapses on techqa (0.630) where factual lookup doesn't benefit from chain-of-thought.

### Adversarial Robustness (n=30 per framework)

| Framework | Non-OOD | OOD Refusal | Multi-hop | Contradictory |
|-----------|---------|-------------|-----------|---------------|
| LangChain | **0.730** | **0.867** | 0.717 | **0.773** |
| LlamaIndex | 0.709 | 0.600 | 0.700 | 0.727 |
| DSPy | 0.710 | 0.200 | 0.703 | 0.727 |

OOD refusal rate = fraction of out-of-distribution questions correctly refused rather than hallucinated. DSPy's chain fabricates reasoning steps when no context is available (0.200 vs LangChain 0.867).

### Reference-Document Overlap Rate (`run_retrieval_overlap.py`, local, no GPU)

Distinct from **Context Coverage** above (which checks retrieved text against the *ground-truth answer*). This checks retrieved text against the *labeled relevant source documents* from RAGBench (`qa_pairs.json`'s `relevant_doc_ids`) — a closer approximation of retrieval recall, without needing doc IDs persisted at benchmark time. Match = word-containment ratio ≥0.6 in either direction (retrieved chunk mostly inside the relevant doc, or vice versa — needed because chunk sizes differ across frameworks, see above).

| Framework | Overall | covidqa | finqa | techqa |
|-----------|---------|---------|-------|--------|
| LangChain | 0.827 | 0.880 | 0.760 | 0.840 |
| LlamaIndex | 0.820 | 0.860 | 0.720 | 0.880 |
| DSPy | 0.780 | 0.860 | 0.640 | 0.840 |

Not Recall@k/NDCG (no ranking signal, approximate text matching rather than exact ID lookup) — reported as a directional retrieval-quality proxy. All three frameworks land in a similar 0.78-0.83 band; no framework shows a clear retrieval-recall advantage at this resolution.

### DSPy MIPROv2 Prompt Optimization

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| Token F1 | 0.483 | 0.483 | +0.001 |
| Correctness | 0.551 | 0.512 | **-0.039** |
| Faithfulness | 0.693 | 0.628 | **-0.065** |
| Completeness | 0.475 | 0.434 | **-0.041** |

All judge metrics dropped after MIPROv2 optimization. Automated prompt optimization overfit to the training distribution.

---

## 4. Key Findings

See [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) for 18 manually reviewed real cases (concrete examples behind the findings below, plus a likely ground-truth label error and a retrieval-metric artifact caught and fixed during review).

**1. String metrics and LLM judge point in opposite directions**
Every string metric ranks DSPy #1. Every judge metric ranks LangChain #1. Pairwise preference confirms the judge direction. Spearman correlation between string and judge metrics: -0.5 to -1.0. Which metric you trust determines which framework you ship. Concretely: token-F1 on numeric finqa answers can reward a wrong number with a near-matching sentence template and punish a right number wrapped in extra words — see `ERROR_ANALYSIS.md` for two real examples where this is the entire explanation for a "DSPy win."

**2. LangChain faithfulness is the only statistically solid claim**
LangChain faithfulness beats both LlamaIndex and DSPy at p<0.001 (gaps of 0.113 and 0.177). All other pairwise differences are marginal (p<0.05, small effect) or not significant. LangChain generates more grounded answers despite retrieving *less* context (0.646 vs DSPy 0.734).

**3. DSPy dominates financial reasoning**
DSPy finqa correctness 0.840 — highest single-domain score in the benchmark. ChainOfThought's step-by-step reasoning fits financial QA's structured numerical problems. But reasoning overhead hurts on techqa (0.630) where direct lookup is faster.

**4. Domain rankings diverge from aggregate**
Aggregate: LangChain wins. Per-domain: DSPy wins finqa by 14 points, frameworks are competitive on covidqa, LangChain leads techqa narrowly. Aggregate benchmarks hide these reversals.

**5. LangChain wins every pairwise head-to-head**
87–51 vs LlamaIndex, 85–51 vs DSPy (Qwen3-14B judge, 143 questions). Pairwise and absolute scores agree — corroborating evidence across two evaluation protocols.

**6. MIPROv2 prompt optimization made DSPy worse**
Faithfulness dropped 0.065 after optimization. Automated prompt search overfit the training set and degraded on held-out test queries.

**7. DSPy hallucinates on OOD questions**
OOD refusal rate 0.200 vs LangChain 0.867. Chain-of-thought fabricates reasoning steps when the corpus has no answer, producing confident hallucinations.

---

## 5. Evaluation Methodology

### Why six metrics?

No single metric captures answer quality. Running all six on the same outputs makes disagreements visible and measurable.

| Method | What it measures | Known limitation | DSPy rank | LangChain rank |
|--------|-----------------|-----------------|-----------|----------------|
| Token F1 | Exact word overlap with ground truth | Rewards verbosity | **1** | 2 |
| ROUGE-1/L | N-gram overlap | Rewards verbosity | **1** | 2 |
| BERTScore (distilbert) | Contextual semantic similarity | Rewards verbosity | **1** | 3 |
| Semantic sim (bge-m3) | Embedding similarity (same model as retrieval) | Rewards verbosity | 2 | **1** |
| Qwen3-14B judge correctness | Factual accuracy vs ground truth | LLM position/length bias | 3 | **1** |
| Qwen3-14B judge faithfulness | Grounded in retrieved context | LLM position/length bias | 3 | **1** |

The string metrics rank DSPy first because ChainOfThought generates longer answers with higher token overlap chance. The judge penalizes the same verbosity when answers drift from the retrieved context. Running both makes this tradeoff explicit.

### Statistical approach

- **Bootstrap CIs across questions** (n=1000, seed=42): captures question-sampling variance — "would results change with a different set of benchmark questions?"
- **Mann-Whitney U + permutation test** (n=10,000): both must reach p<0.05 to claim significance. Controls false discovery from multiple comparisons.
- **Pairwise preference**: direct A-vs-B comparison avoids scale anchoring; more robust than absolute numeric scores.

### Cross-family judging

Qwen3-14B (Alibaba) judges Llama-3.1-8B (Meta) outputs. Different training lineage, different RLHF, different company — reduces same-model self-preference bias, though it doesn't rule out other judge biases (rubric bias, verbosity bias, position bias). Both run locally on vLLM; evaluation cost is near zero after instance startup.

### Adversarial evaluation

Four hard query types probing failure modes beyond standard accuracy:
- **Multi-hop** — requires connecting information across multiple documents
- **Ambiguous** — underspecified, missing key context
- **Out-of-distribution** — plausibly related but not in corpus; correct answer is refusal
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
│       ├── metrics.py                 # Token F1, BERTScore, LLM judge, ECE
│       ├── adversarial_agent.py       # Adversarial query gen + robustness eval
│       └── tracing.py                 # Arize Phoenix / OTel instrumentation
├── orchestrator/
│   ├── main.go                        # Go orchestrator, rate limiter, Prometheus
│   ├── run_servers.sh                 # Start all servers + run benchmark
│   └── generate_queries.py           # Balanced query sampling (per domain)
├── docker-compose.yml                 # Prometheus + Grafana + Arize Phoenix
├── run_eval_unified.py               # Single-pass eval: judge once, aggregate globally + by domain
├── run_pairwise_eval.py              # Pairwise preference (Qwen judge picks A vs B)
├── run_judge_stability.py            # Judge repeatability audit (stratified subset, 5 runs each)
├── run_serial_latency.py             # Serial latency benchmark (no concurrent GPU pressure)
├── run_semantic_sim.py               # bge-m3 cosine similarity
├── run_bertscore.py                  # BERTScore (local, no GPU)
├── run_rouge.py                      # ROUGE-1/2/L (local, no GPU)
├── run_statistical_tests.py          # Mann-Whitney + permutation tests (local)
├── run_metric_comparison.py          # Cross-metric ranking + Spearman correlation (local)
├── compute_stats_local.py            # Bootstrap CIs + latency percentiles (local)
├── run_dspy_optimized.py             # MIPROv2 baseline vs optimized comparison
└── setup_lambda.sh                   # Lambda Cloud GPU instance setup
```

---

## Running It

### On Lambda Cloud (2x H100 recommended)

```bash
git clone https://github.com/sharle21/RAG-Framework-Comparison-LangChain-vs-LlamaIndex-vs-DSPy.git rag-bench
cd rag-bench && bash setup_lambda.sh <hf-token>

source ~/vllm_env/bin/activate
export LD_LIBRARY_PATH=/home/ubuntu/vllm_env/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

# Start worker model on GPU 0, wait until ready, then start judge on GPU 1
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 > /tmp/vllm_worker.log 2>&1 &
until curl -s http://localhost:8000/v1/models | grep -q "Llama"; do sleep 10; done

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B --port 8001 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 > /tmp/vllm_judge.log 2>&1 &

# Run benchmark (starts RAG servers sequentially, then fires Go orchestrator)
export PATH="/usr/local/go/bin:$PATH"
RPS=5 WORKERS=8 bash orchestrator/run_servers.sh

# Evaluate
PYTHONUNBUFFERED=1 nohup python -u run_eval_unified.py > /tmp/eval.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python -u run_pairwise_eval.py > /tmp/pairwise.log 2>&1 &
```

### With observability stack

```bash
docker compose up -d phoenix prometheus grafana
TRACING=1 RPS=5 WORKERS=8 bash orchestrator/run_servers.sh
# Traces:  http://localhost:6006
# Metrics: http://localhost:3000
```

### Local evaluation (no GPU needed)

```bash
pip install bert-score rouge-score scipy
python run_bertscore.py
python run_rouge.py
python run_statistical_tests.py
python run_metric_comparison.py
python compute_stats_local.py
```

---

## Bugs Found and Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| DSPy showing 4ms generation | LiteLLM disk cache active despite `cache=False` in `dspy.configure()` | Pass `cache=False` to `dspy.LM()` directly |
| LangChain all queries failing | Chroma index built with OpenAI 1536-dim, queried with bge-m3 1024-dim | Delete stale index, rebuild; remove from git with `git rm --cached` |
| LlamaIndex `ValueError: Unknown model` | `OpenAI` class rejects custom `base_url` | Switch to `OpenAILike` |
| Qwen3 `<think>` blocks breaking JSON parse | Qwen3-14B outputs `<think>...</think>` before JSON | Strip with `re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)` |
| OOM on server startup | All three servers embedding corpus simultaneously | Sequential startup in `run_servers.sh` |
| vLLM CUDA error 802 on fresh instance | CUDA system not initialized at driver level | Verify: `python -c "import ctypes; c=ctypes.CDLL('libcuda.so.1'); print(c.cuInit(0))"` — if 802, terminate and get new instance |
| Double judge calls for domain eval | `run_eval.py` + `run_eval_domains.py` each called judge on same 450 responses | Replaced with `run_eval_unified.py`: judge once, compute domain stats from same per-question rows |
| `qa_pairs.json` relevant_doc_ids pointing at documents that don't exist | `prepare_data.py` deduped passages by content, but ~51% of techqa passages (and similarly finqa) are reused verbatim across different questions — dedup silently dropped documents that later questions' IDs still referenced. Broke 312/450 queries' retrieval-recall lookups | Removed the dedup entirely — every `(question_idx, passage_idx)` is saved even if content repeats; corpus grew from 5,704 to 56,072 passages, all IDs now resolve |

---

## References

- [RAGBench Dataset](https://huggingface.co/datasets/rungalileo/ragbench)
- [DSPy](https://github.com/stanfordnlp/dspy) — Stanford NLP
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention inference
- [MIPROv2](https://arxiv.org/abs/2406.11695) — Automatic prompt optimization
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — LLM observability

