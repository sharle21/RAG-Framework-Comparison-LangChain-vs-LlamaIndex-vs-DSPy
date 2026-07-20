# ADR-0004: Evaluation Uncertainty — Bootstrap CIs over Repeated Judge Calls

**Status:** Accepted

## Context

LLM judge scores have two distinct sources of variance:

1. **Question-sampling variance** — would the framework rankings change if we benchmarked on a different set of questions?
2. **Judge stochasticity** — would Qwen score the same answer differently if asked again?

An earlier approach (`n_runs=5`) called the judge 5 times per answer and averaged the scores. This addresses source (2) but at 5× the cost: 450 queries × 5 runs × 3 frameworks = 6,750 judge calls, taking 6–10 hours on a single Qwen endpoint.

Bootstrap confidence intervals (resampling questions with replacement, n=1000) address source (1) at negligible compute cost — no additional LLM calls required.

These two sources are independent. Bootstrapping across questions does not substitute for measuring judge repeatability, and repeated judge calls do not capture question-sampling variance. They must be reported separately if both matter.

## Decision

**Full benchmark:** n_runs=1 across all 450 answers. Bootstrap CIs (n=1000, seed=42) across questions for all reported means.

**Rationale for n_runs=1:**
- At temperature=0, Qwen3-14B is nearly deterministic. Judge stochasticity is low relative to question-sampling variance for a 450-query benchmark.
- Question-sampling variance (does the ranking hold across different questions?) is the more important uncertainty source for a framework comparison.
- n_runs=5 would take 6–10 hours for marginal gain; n_runs=1 + bootstrap takes ~45 minutes with equivalent statistical insight.

**What bootstrap CIs mean in this context:**
- A 95% CI of [0.532, 0.650] on LangChain correctness means: if we resampled 450 different questions from the same distribution, the mean correctness would fall in this range 95% of the time.
- It does NOT mean Qwen would produce scores in this range if asked again.

## Consequences

- All reported means include 95% bootstrap CIs. Comparisons without overlapping CIs are treated as practically significant.
- Statistical significance additionally requires Mann-Whitney U + permutation test (both p<0.05) to control false discovery from multiple comparisons.
- Judge repeatability can be assessed separately on a stratified subset (e.g., 45 responses × 5 runs) without bloating the main evaluation cost.
