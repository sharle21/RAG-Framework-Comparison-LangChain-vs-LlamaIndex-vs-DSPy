# Improvement Plan

This plan has two parts:
- **Part 1** — Fix methodological gaps and suspicious results in the existing benchmark
- **Part 2** — Expand into a High-Performance LLM Benchmarking Platform (target: Glean eval role + CoreWeave serving role)

---

## Part 1: Fix the Existing Benchmark

### Priority Order

| # | Step | Effort | Impact |
|---|---|---|---|
| 1 | Increase test set size | Low | High |
| 2 | Per-domain breakdown | Low | Medium |
| 3 | Fix latency measurement | Medium | High |
| 4 | Investigate LlamaIndex hallucination vs correctness tension | Medium | High |
| 5 | Fix adversarial scoring | Medium | Medium |
| 6 | Track noise documents | Medium | Medium |
| 7 | Run DSPy optimized | High | High |
| 8 | Replace string F1 with semantic similarity (BERTScore) | Low | High |
| 9 | Run LLM judge multiple times and average scores | Low | Medium |

---

## 1. Increase Test Set Size

**Problem:** 30 questions total (10 per domain) is too small to trust the scores. Numbers like "LlamaIndex correctness = 0.698 vs LangChain = 0.644" are not statistically meaningful at this sample size.

**Fix:**
- Re-run benchmark with `--n-pairs 150` (50 per domain)
- Report confidence intervals alongside scores, not just point estimates
- Run the full benchmark 3 times and average results — LLM judges have randomness even at temperature=0

**Files to change:** `run_benchmark.py` — change default `--n-pairs` from 10 to 50.

---

## 2. Per-Domain Breakdown

**Problem:** techqa, finqa, and covidqa are averaged together. LlamaIndex may be strong on finqa (numerical, precise language) but weaker on covidqa (long scientific passages) — the average hides this.

**Fix:** Group results by domain before evaluating and report scores per domain alongside the overall average.

**Files to change:** `run_benchmark.py` — add domain grouping in `evaluate_all()`:

```python
by_domain = defaultdict(list)
for r in results:
    by_domain[r.get("domain", "unknown")].append(r)
for domain, domain_results in by_domain.items():
    domain_scores = evaluate_all(domain_results, f"{name}_{domain}")
```

---

## 3. Fix the Latency Comparison

**Problem:** DSPy averages 20ms vs ~2,000ms for LangChain and LlamaIndex. This is not apples-to-apples — LangChain and LlamaIndex make OpenAI embedding API calls at query time, while DSPy searches an in-memory FAISS index with no network call.

**Fix — Split latency into two components:**
- `retrieval_ms` — time to find relevant documents
- `generation_ms` — time for the LLM to produce the answer

Report both separately so it's clear: "DSPy retrieval is faster because it's in-memory; LLM generation time is roughly equal across all three."

**Alternative fix:** Give DSPy a Chroma-backed retriever that also calls the embedding API at query time, making infrastructure identical across all three.

**Files to change:** `src/langchain_rag/pipeline.py`, `src/llamaindex_rag/pipeline.py`, `src/dspy_rag/pipeline.py` — add separate timers around the retrieval and generation steps inside `query()`.

---

## 4. Investigate LlamaIndex Hallucination vs Correctness Tension

**Problem:** LlamaIndex scores highest on correctness (0.698) but hallucinates the most (12.2%). These are contradictory — the Mistral judge likely rewards fluent, confident answers even when they contain fabricated details.

**Fix:**
- Add a cross-check: for every answer marked "correct" by the LLM judge, check if the failure mode classifier also marks it "hallucination" — flag any overlaps
- Break down correctness scores by failure mode category to see if hallucinated answers are scoring artificially high
- Re-judge just the hallucinated answers with a stricter prompt that explicitly penalizes claims not grounded in the retrieved context

**Files to change:** `src/evaluation/metrics.py` — add cross-check logic; `run_benchmark.py` — add cross-check reporting to summary output.

---

## 5. Fix the Adversarial Scoring

**Problem:** All 3 frameworks score *higher* on adversarial questions than on normal ones (ratios > 1.0). This means the metric is broken — `correct_refusal` on out-of-distribution questions inflates the overall robustness score and masks real degradation on multi-hop and contradictory queries.

**Fix:**
- Separate OOD queries from the rest — `correct_refusal` should be its own metric, not mixed into overall robustness
- For multi-hop, ambiguous, and contradictory questions: score them using the same correctness/faithfulness metrics as normal questions, not a separate robustness rubric
- Report degradation as: `(adversarial score on hard types) / (baseline score on normal questions)` — this gives a clean, interpretable number

**Files to change:** `src/evaluation/adversarial_agent.py` — split `evaluate_adversarial_results()` to handle OOD separately; update `compute_degradation()`.

---

## 6. Track Noise Documents

**Problem:** 3 deliberately false documents are injected into the corpus in `synthetic_data.py` (fake facts about IBM, COVID vaccines, and financial derivatives) but their effect is never measured in the results. There is no way to tell if any framework retrieved them or was fooled by them.

**Fix:**
- Add `"is_noise": True` metadata to noise documents (already present in `synthetic_data.py`)
- In each pipeline's `query()` return value, flag if any retrieved context came from a noise document
- Add a `poison_rate` metric to the summary: "what % of queries retrieved at least one noise document and used it in the answer?"

**Files to change:** `src/langchain_rag/pipeline.py`, `src/llamaindex_rag/pipeline.py`, `src/dspy_rag/pipeline.py` — surface noise doc metadata in query results. `run_benchmark.py` — compute and report poison rate.

---

## 7. Run DSPy Optimized

**Problem:** DSPy's core value proposition is automatic prompt optimization (MIPROv2, BootstrapFewShot). Comparing unoptimized DSPy to LangChain/LlamaIndex is unfair — it's like testing a car without tuning the engine.

**Fix:** Add an `dspy_optimized` variant that runs the prompt optimizer before benchmarking:

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=your_metric)
optimized_module = optimizer.compile(rag_module, trainset=train_pairs[:20])
```

Run two DSPy rows in the results table: `dspy_baseline` and `dspy_optimized`. This shows whether DSPy's real advantage actually moves the needle on quality metrics vs the other frameworks.

**Files to change:** `src/dspy_rag/pipeline.py` — add `optimize()` method. `run_benchmark.py` — add `dspy_optimized` as a fourth framework entry.

**Note:** Requires a small training set (20 labeled QA pairs). Can use a split from `qa_pairs.json`.

---

## 8. Replace String F1 with Semantic Similarity (BERTScore)

**Problem:** Token F1 compares words, not meaning. Synonyms and paraphrases score incorrectly low:

```
Ground truth: "December 2019"
Answer:       "late in the year 2019"
F1 score: ~0.2   ← almost no word overlap, but same meaning
```

Negation also isn't caught:
```
Ground truth: "The vaccine was effective"
Answer:       "The vaccine was not effective"
F1 score: ~0.8   ← high word overlap, completely opposite meaning
```

**Fix:** Replace `evaluate_string_overlap` in `metrics.py` with BERTScore, which compares meaning using embeddings:

```python
from bert_score import score as bert_score

def evaluate_semantic_similarity(results: list[dict]) -> dict:
    predictions = [r["answer"] for r in results if r.get("answer")]
    references  = [r["ground_truth"] for r in results if r.get("answer")]
    P, R, F1 = bert_score(predictions, references, lang="en")
    return {
        "bert_precision": float(P.mean()),
        "bert_recall":    float(R.mean()),
        "bert_f1":        float(F1.mean()),
    }
```

Keep the old F1 as well for backwards compatibility — run both and report both. BERTScore requires `pip install bert-score`. Runs locally, no API key needed.

**Files to change:** `src/evaluation/metrics.py` — add `evaluate_semantic_similarity()`. `run_benchmark.py` — call it alongside `evaluate_string_overlap()`.

---

## 9. Run LLM Judge Multiple Times and Average

**Problem:** Mistral is an LLM — it can be inconsistent. The same answer scored twice might get different results even at temperature=0. A single judge call per answer means one unlucky score can skew the whole framework's average.

Also, LLM judges have known biases:
- **Sycophancy** — they favor confident, fluent answers even when wrong (likely why LlamaIndex scores high despite hallucinating)
- **Length bias** — longer answers tend to score higher regardless of quality
- **Prompt sensitivity** — slightly different wording gives different scores

**Fix:** Run the judge N times per answer and report mean + standard deviation:

```python
def evaluate_llm_judge(results: list[dict], n_runs: int = 3) -> dict:
    all_runs = []
    for _ in range(n_runs):
        run_scores = {"correctness": [], "faithfulness": [], "completeness": []}
        for r in results:
            # ... existing judge logic ...
            run_scores["correctness"].append(score)
        all_runs.append(run_scores)

    # Average across runs + report std deviation
    result = {}
    for dim in ["correctness", "faithfulness", "completeness"]:
        all_scores = [s for run in all_runs for s in run[dim]]
        result[f"{dim}_mean"] = sum(all_scores) / len(all_scores)
        result[f"{dim}_std"]  = statistics.stdev(all_scores)
    return result
```

High std deviation = judge was inconsistent = less trustworthy score. This surfaces the judge reliability problem rather than hiding it.

**Optional:** Add a second judge from a third model family (e.g. Gemini) and only fully trust scores where both judges agree within 0.1. Flag disagreements for manual review.

**Files to change:** `src/evaluation/metrics.py` — update `evaluate_llm_judge()` to accept `n_runs` parameter.

---

## Part 1 Expected Outcome

After these changes:
- Latency comparison will be fair and interpretable
- Quality scores will be statistically reliable (larger n, confidence intervals)
- The LlamaIndex hallucination paradox will be explained or resolved
- Adversarial results will show real degradation instead of inflated refusal scores
- DSPy will be tested at its actual best-case performance
- Domain-specific strengths and weaknesses will be visible

---

## Part 2: High-Performance LLM Benchmarking Platform

Transform the project from a Python benchmark script into a production-grade platform with four tiers.

### Target Architecture

```
┌─────────────────────────────────────────────────────┐
│              Go Orchestrator (CLI)                  │
│   Worker pool · Rate limiter · DB streaming         │
└───────────────┬─────────────────────────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌──────────────┐
│Inference│ │  Eval  │ │   Metrics    │
│  Tier  │ │  Tier  │ │    Tier      │
│ vLLM / │ │ DSPy · │ │ Prometheus · │
│ Triton │ │ RAGAS · │ │ Grafana ·    │
│        │ │ Judges │ │ Arize Phoenix│
└────────┘ └────────┘ └──────────────┘
```

---

### A. Go Concurrency Layer

**Why:** Demonstrates distributed systems thinking for both Glean (distributed data pipelines) and CoreWeave (networked services under load).

**What to build:** A CLI in Go that replaces the current Python `run_benchmark.py` orchestrator.

Key components:
- **Worker pool** using goroutines — maintain a constant requests-per-second (RPS) load against the inference server
- **Token-bucket rate limiter** using `golang.org/x/time/rate` — prevent blowing past API limits during the 5,000-query run
- **Result streaming** — write results to a database (Postgres or SQLite) as they come in rather than buffering in memory

```go
// Example: token-bucket rate limiter
limiter := rate.NewLimiter(rate.Limit(10), 20) // 10 RPS, burst of 20
for _, query := range queries {
    limiter.Wait(ctx)
    go worker(query, resultsChan)
}
```

**Talking point for CoreWeave:** "I used goroutines to maintain a constant RPS load to stress-test the inference server, and a token-bucket rate limiter to simulate realistic API traffic patterns."

**Talking point for Glean:** "I built a concurrent Go pipeline that distributes 5,000 evaluation queries across a worker pool and streams results into a database for real-time monitoring."

---

### B. Inference Tier — vLLM / Triton

**Why:** CoreWeave specifically wants to see that you've deployed and measured a real inference server, not just called an API.

**What to build:**
- Run a local **Llama-3-8B** model using vLLM in a Docker container
- Run your benchmark queries against the local model instead of (or alongside) OpenAI
- Measure the impact of **micro-batching** on throughput and latency

```bash
# Run vLLM locally
docker run --gpus all -p 8000:8000 vllm/vllm-openai \
  --model meta-llama/Meta-Llama-3-8B-Instruct

# Then point your pipelines at localhost:8000 instead of api.openai.com
```

Key metrics to capture:
- **KV Cache utilization** — how full is the cache during the 5,000-query run?
- **Continuous batching** — how does throughput change as concurrent requests increase?
- **Latency vs Throughput trade-off** — plot a curve: as you increase batch size, throughput goes up but p95 latency goes up too

**Talking point for CoreWeave:** "I deployed vLLM in a Docker container and measured the impact of micro-batching on throughput. I tracked KV Cache utilization during a 5,000-query run to understand memory pressure under load."

---

### C. Observability Split

Two separate observability tools, one for system health and one for LLM quality.

#### System Health — Prometheus + Grafana (for CoreWeave)

Metrics to export from the Go orchestrator:
- `request_latency_seconds` — histogram of query latency
- `gpu_utilization` — GPU % during the benchmark run
- `kv_cache_utilization` — how full the vLLM KV cache is
- `requests_per_second` — actual RPS being achieved vs target
- `error_rate` — % of queries that failed

```go
// Example: Prometheus counter in Go
queryLatency := prometheus.NewHistogramVec(
    prometheus.HistogramOpts{Name: "request_latency_seconds"},
    []string{"framework"},
)
```

**Talking point for CoreWeave:** "I used Prometheus to track GPU memory (KV Cache) during my 5,000-query run and built a Grafana dashboard to visualize the latency vs throughput trade-off in real time."

#### LLM Quality — Arize Phoenix (for Glean)

Traces to export from the Python eval tier:
- `retrieval_context_relevancy` — per-query score
- `judge_score` — Mistral judge correctness per query
- `failure_mode` — classification per query
- `hallucination_flag` — whether the answer was flagged as hallucinated

Arize Phoenix gives you a UI to drill into individual queries, compare frameworks side-by-side, and spot regressions across benchmark runs.

**Talking point for Glean:** "I used Arize Phoenix to trace retrieval quality and judge scores at the individual query level, which lets me identify exactly which query types each framework struggles with."

---

### D. The "Closing the Loop" Story (for Glean)

This is the most compelling narrative for an eval role: your system doesn't just measure failures — it automatically fixes them.

**The loop:**
1. Go orchestrator runs 5,000 queries and identifies failures (low judge score, hallucination flag)
2. Failures are written to the database with their query, retrieved context, and bad answer
3. DSPy MIPROv2 optimizer reads the failures as a training signal and generates a better prompt
4. The new prompt is re-deployed and the benchmark re-runs to confirm improvement

```python
# DSPy MIPROv2 — automatic prompt improvement from failure cases
from dspy.teleprompt import MIPROv2

failures_as_trainset = load_failures_from_db()  # the bad answers
optimizer = MIPROv2(metric=correctness_metric, auto="light")
improved_module = optimizer.compile(rag_module, trainset=failures_as_trainset)
```

**Talking point for Glean:** "My Go runner identifies failures and feeds them into DSPy MIPROv2 to automatically generate a better prompt — closing the loop between evaluation and improvement."

---

### E. The Golden Set (for Glean)

**The keyword Glean wants to hear:** "I built a Golden Set of 5,000 queries across three domains to prevent regressions during framework migrations."

**What this means in practice:**
- Expand the current 90-question test set to 5,000 using `synthetic_data.py` (run `--testset-size 5000`)
- Lock the set — once created, never modify it so comparisons are stable over time
- Use it as a regression test: any time you swap a framework, model, or prompt, re-run against the golden set and verify scores don't drop
- Store results per-run in the database so you can track score trends over time

---

### Implementation Order for Part 2

| # | Component | Effort | Who it impresses |
|---|---|---|---|
| 1 | vLLM Docker setup + run queries against local model | Medium | CoreWeave |
| 2 | Prometheus metrics export from Python eval | Medium | CoreWeave |
| 3 | Go worker pool + rate limiter CLI | High | Both |
| 4 | Arize Phoenix traces | Medium | Glean |
| 5 | DSPy MIPROv2 "closing the loop" | High | Glean |
| 6 | Grafana dashboard | Low (once Prometheus is up) | CoreWeave |
| 7 | Expand to 5,000-query golden set | Low (just run the script) | Glean |

Start with vLLM + Prometheus — they give you the most CoreWeave-specific talking points with moderate effort, and the Docker setup will take less than an hour.

---

## Part 3: Lambda Cloud Setup

Use Lambda Cloud credits to run the full 5,000-query benchmark. No local GPU needed, no API costs for inference.

---

### Recommended Instance

**1x A100 80GB — $1.99/hr**

Fits both models on one instance comfortably:
- Llama-3-8B-Instruct (worker) on port 8000 — ~16GB VRAM
- Qwen2.5-14B-Instruct (judge) on port 8001 — ~28GB VRAM
- ~36GB VRAM remaining for KV cache headroom during concurrent requests

Alternative if 80GB is unavailable: **2x A100 40GB at $2.58/hr** — one model per GPU, cleaner isolation, slightly better for demonstrating multi-instance serving.

---

### Model Stack

| Role | Model | Port | Training family | VRAM |
|---|---|---|---|---|
| Worker | Llama-3-8B-Instruct | 8000 | Meta | ~16GB |
| Judge | Qwen2.5-14B-Instruct | 8001 | Alibaba | ~28GB |
| Embeddings | bge-m3 (sentence-transformers) | local | BAAI | CPU only |

Three different companies, three different training lineages — cross-family evaluation maintained with zero API cost.

---

### Estimated Credit Usage

| Session | Task | Duration | Cost |
|---|---|---|---|
| Session 1 | Setup: install vLLM, pull models from HuggingFace | ~1.5 hrs | ~$3 |
| Session 2 | Run 5,000-query benchmark + Prometheus scraping | ~3 hrs | ~$6 |
| Session 3 | Rerun with different batch sizes (latency/throughput curve) | ~2 hrs | ~$4 |
| Session 4 | DSPy MIPROv2 optimization loop | ~1 hr | ~$2 |
| **Total** | | **~7.5 hrs** | **~$15** |

Model downloads (Llama ~16GB, Qwen2.5-14B ~28GB) happen in Session 1 and are the slowest part. Do them first before the clock runs long.

---

### Instance Setup Commands

```bash
# 1. SSH into Lambda instance
ssh ubuntu@<instance-ip>

# 2. Install vLLM
pip install vllm

# 3. Pull models (do this first — takes 20-30 mins)
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
huggingface-cli download Qwen/Qwen2.5-14B-Instruct

# 4. Start worker model (background)
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --tensor-parallel-size 1 &

# 5. Start judge model (background)
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --port 8001 \
  --tensor-parallel-size 1 &

# 6. Verify both are running
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

---

### Saving Results Before Shutdown

Lambda instances do not persist storage between sessions by default. Always save results before terminating:

```bash
# From your local machine — copy results off the instance
scp -r ubuntu@<instance-ip>:~/RAG-Framework-Comparison/results/ ./results_lambda/
```

**Better option:** Attach a Lambda persistent storage volume (~$0.20/mo) at instance creation. Mount it at `~/persistent/` and write all results there — survives instance termination.

---

### What to Point Your Pipelines At

Once vLLM is running on Lambda, update your pipeline configs to hit the Lambda instance instead of OpenAI:

```python
# In langchain_rag/pipeline.py, llamaindex_rag/pipeline.py, dspy_rag/pipeline.py
# Replace:
llm = ChatOpenAI(model="gpt-4o-mini")
# With:
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    base_url="http://<lambda-instance-ip>:8000/v1",
    api_key="dummy",  # vLLM doesn't require a real key
)

# Same pattern for the Qwen judge in metrics.py
judge = ChatOpenAI(
    model="Qwen/Qwen2.5-14B-Instruct",
    base_url="http://<lambda-instance-ip>:8001/v1",
    api_key="dummy",
)
```

vLLM exposes an OpenAI-compatible API so no other code changes are needed — just swap the `base_url`.

---

### Security Note

Lambda instances have public IPs. Don't leave ports 8000/8001 open to the internet during the run. Either:
- Use an SSH tunnel: `ssh -L 8000:localhost:8000 ubuntu@<instance-ip>` and hit `localhost:8000` from your machine
- Or restrict inbound traffic to your IP in the Lambda firewall settings

---

## Part 4: Docker Compose

All services are defined in `docker-compose.yml` so the entire stack starts with one command instead of manually launching each service every session.

### Files Created

```
RAG-Framework-Comparison/
├── docker-compose.yml
└── infra/
    ├── prometheus.yml                        # scrape targets
    └── grafana/
        └── datasources/
            └── prometheus.yml                # auto-wires Prometheus into Grafana
```

### Services in the Compose File

| Service | Image | Port | Purpose |
|---|---|---|---|
| `vllm-worker` | vllm/vllm-openai | 8000 | Llama-3-8B — answer generation |
| `vllm-judge` | vllm/vllm-openai | 8001 | Qwen2.5-14B — cross-family scoring |
| `prometheus` | prom/prometheus | 9090 | Scrapes metrics from vLLM + Go |
| `grafana` | grafana/grafana | 3000 | Dashboards — latency, throughput, KV cache |
| `phoenix` | arizephoenix/phoenix | 6006 / 4317 | LLM quality traces |

All services share a `huggingface-cache` volume so models are only downloaded once, not re-downloaded each session.

### Common Commands

```bash
# Start everything
docker compose up -d

# Start only observability stack (if running vLLM as bare processes)
docker compose up -d prometheus grafana phoenix

# Check all services are healthy
docker compose ps

# Follow logs for a specific service
docker compose logs -f vllm-worker

# Tear everything down (keeps volumes)
docker compose down

# Tear down and delete all data (fresh start)
docker compose down -v
```

### GPU Caveat on Single A100 80GB

Both vLLM containers are pinned to `device_ids: ["0"]`. The compose file sets:
- `vllm-worker` → `--gpu-memory-utilization 0.45` (~36GB)
- `vllm-judge` → `--gpu-memory-utilization 0.50` (~40GB)

Combined ~76GB which fits on an 80GB A100. If you hit OOM errors, lower one of the utilization values or switch to the hybrid approach:

```bash
# Hybrid: run vLLM as bare processes, Compose only for observability
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000 --gpu-memory-utilization 0.45 &
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8001 --gpu-memory-utilization 0.50 &

docker compose up -d prometheus grafana phoenix
```

### On 2x A100 40GB Setup

Change `vllm-judge` device assignment in `docker-compose.yml`:

```yaml
vllm-judge:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["1"]   # GPU 1 instead of GPU 0
            capabilities: [gpu]
```

Clean one-GPU-per-model separation, no memory utilization tuning needed.
