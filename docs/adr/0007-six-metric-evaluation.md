# ADR-0007: Six-Metric Evaluation Suite

**Status:** Accepted

## Context

RAG answer quality has no single ground truth metric. Each existing metric captures a different aspect and has a known failure mode:

| Metric | What it captures | Known failure mode |
|--------|-----------------|-------------------|
| Token F1 | Exact word overlap with reference | Rewards verbosity; "the the the" scores higher than a concise correct answer |
| ROUGE-1/L | N-gram overlap | Same verbosity bias; doesn't understand paraphrase |
| BERTScore | Contextual embedding similarity | Still rewards longer answers; distilbert doesn't understand domain jargon |
| LLM judge (correctness) | Factual accuracy vs ground truth | LLM length/position bias; slow; expensive |
| LLM judge (faithfulness) | Grounded in retrieved context | Same LLM biases |
| Semantic similarity (bge-m3) | Cosine distance in retrieval embedding space | Uses same model as retrieval — correlated with what the retriever finds, not ground truth |

Using only string metrics would declare DSPy the winner (ChainOfThought generates longer answers with higher overlap). Using only the judge would miss the systematic verbosity pattern that explains DSPy's string metric advantage.

## Decision

Run all six metrics on the same 450 outputs and report results across all of them.

The central finding of this benchmark — that string metrics and judge metrics produce opposite rankings — is only visible when you run both. A benchmark that only ran one family of metrics would produce a correct but misleading conclusion.

Additionally, running the **same embedding model for retrieval and semantic similarity** (bge-m3 in both cases) creates an intentional correlation: semantic similarity measures how close the answer is to the ground truth in the same space the retriever uses to select documents. This is a retrieval-aware quality signal that BERTScore (which uses distilbert) does not provide.

## Consequences

- Reporting six metrics requires the reader to form a view on which to trust. The README and findings section explicitly address this rather than picking a winner by metric.
- String metrics (F1, ROUGE, BERTScore) run locally with no GPU — fast and reproducible anywhere.
- LLM judge metrics require vLLM and a ~45-minute evaluation run — not reproducible without GPU access.
- Cross-metric Spearman correlation analysis (`run_metric_comparison.py`) quantifies the disagreement: -0.5 to -1.0 between string and judge metrics, making the divergence statistically documented rather than anecdotal.
