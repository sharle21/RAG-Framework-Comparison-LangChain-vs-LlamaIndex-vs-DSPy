# ADR-0003: Cross-Family LLM Judge (Qwen3-14B judging Llama-3.1-8B)

**Status:** Accepted

## Context

LLM-as-judge evaluation requires choosing a judge model. The worker model generating RAG answers is Llama-3.1-8B-Instruct (Meta). Several options for the judge:

**Same model as worker (Llama-3.1-8B)**
- Cheapest — one model, one GPU
- Known failure mode: same-family models share training data, RLHF preferences, and output style. The judge tends to rate outputs that resemble its own generation style higher, regardless of factual quality. This is same-family favoritism.

**GPT-4o / GPT-4o-mini (OpenAI)**
- Different family — avoids same-family bias
- Requires API calls per judgment, costs scale with query count (450 queries × 3 metrics × n_runs = thousands of API calls)
- Adds external dependency; results can't be reproduced without API access and incur ongoing cost

**Qwen3-14B (Alibaba)**
- Different company, different training lineage, different RLHF pipeline from Llama
- Runs locally on vLLM — no API cost, fully reproducible
- 14B parameters — larger than the 8B worker, so has sufficient reasoning capacity to evaluate
- Qwen3 outputs `<think>...</think>` reasoning blocks before answering, which are stripped before JSON parsing

## Decision

Use Qwen3-14B as the judge, served locally on a dedicated GPU via vLLM (port 8001).

Cross-family is the minimum bar for credible LLM-as-judge evaluation. Qwen3-14B satisfies this while keeping the evaluation fully local and reproducible. A dedicated GPU (CUDA_VISIBLE_DEVICES=1 on 2x H100) ensures the judge doesn't compete with the worker for memory.

## Consequences

- Requires 2x H100 (or equivalent 80GB GPU) to run worker and judge simultaneously without memory pressure.
- Qwen3's `<think>` blocks must be stripped before JSON parsing: `re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)`.
- Judge bias is reduced but not eliminated — Qwen3 has its own preferences (e.g., may favor structured answers, penalize brevity differently than humans would).
- Judge ECE (Expected Calibration Error) measured to quantify how well Qwen's confidence scores predict actual correctness.
