# ADR-0005: Pairwise Preference Alongside Absolute Scoring

**Status:** Accepted

## Context

The primary evaluation protocol uses absolute scores: Qwen3-14B rates each answer on correctness, faithfulness, and completeness (0.0–1.0). This has a known failure mode called **scale anchoring bias**: LLM judges often rate answers relative to a perceived baseline rather than on an absolute scale, compressing scores toward the middle and making small differences hard to interpret.

Absolute scoring also conflates two questions:
1. Is this answer good in absolute terms?
2. Is this answer better than the alternative?

For a framework comparison, question (2) is what matters. A judge that rates LangChain 0.59 and LlamaIndex 0.56 is making a much stronger claim than those numbers suggest — the 0.03 difference could be noise.

Pairwise preference removes the scale entirely: the judge reads both answers and picks the better one. No numeric scores, no anchoring, no scale calibration required.

## Decision

Run pairwise preference evaluation in addition to absolute scoring, using the same Qwen3-14B judge.

For each question where all three frameworks produced a valid answer (143 questions), run all three pairwise comparisons (LC vs LI, LC vs DSPy, LI vs DSPy). Report win counts and head-to-head ratios.

## Consequences

- Pairwise and absolute scores should agree directionally if the evaluation is consistent. Agreement is a corroborating signal; disagreement would indicate scale anchoring or position bias in the absolute judge.
- In this benchmark they agree: LangChain wins both pairwise (87–51 vs LlamaIndex) and absolute scoring. This consistency strengthens the conclusion.
- Pairwise cannot produce domain-level breakdowns easily (would require 3× more questions per domain). Domain analysis relies on absolute scores.
- Win margin (87–51 = 63% win rate) is more interpretable to stakeholders than "correctness 0.592 vs 0.559."
