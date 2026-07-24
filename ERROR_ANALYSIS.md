# Error Analysis

Manually reviewed cases from the actual benchmark data (`results/go_results_20260408_013644.json`, `results/adversarial_*.json`) — not aggregate stats. Selected by scanning for metric disagreement (high F1/low context-coverage, low F1/high context-coverage) and real adversarial-eval failure modes, then reading the actual question/context/answer for each to diagnose what happened. No LLM judge was re-run for this — Qwen3-14B judge scores where cited come from the existing adversarial eval output; everything else is direct inspection of stored answers/contexts.

## Numeric QA: token-F1 rewards template match, not correctness

| Query | Framework | Ground truth | Answer | F1 | Diagnosis |
|---|---|---|---|---|---|
| "what is the net change in net revenue during 2015 for entergy corporation?" | DSPy | decrease of $6 million | decrease of **$92.9 million** | 0.80 | Wrong number, matching sentence template. F1 rewards the near-identical wording ("net change in net revenue... is a decrease of $X million") and barely penalizes the wrong digit. LangChain (F1=0.00) and LlamaIndex (F1=0.25) phrased differently and scored lower despite being no more or less correct. |
| "what is the five year total return on the goldman sachs group inc.?" | DSPy | $148.36 | **$248.36** | 0.92 | Same pattern — wrong number (off by exactly $100), near-identical phrasing, near-perfect F1. |
| "what portion of the total 2015 restructuring programs is related to termination benefits?" | DSPy | $5009 | "...total is $6304. The portion related to termination benefits is **$5009**." | 0.12 | Inverse problem: DSPy's number is *correct*, but F1 punishes it for wrapping the right answer in a full sentence when ground truth is a bare `$5009` string. Precision craters even though the extracted fact is right. |

**Conclusion:** these three rows are the concrete mechanism behind the README's "string metrics and judge point in opposite directions" finding. Token-F1 on numeric answers is closer to a sentence-template similarity score than a correctness check — it can reward a wrong number and punish a right one depending on surrounding phrasing. This isn't specific to DSPy; it's inherent to F1 on short numeric ground truths and should weigh heavily against trusting the string-metric leaderboard for finqa.

## Likely ground-truth label error, not a pipeline failure

| Query | Framework(s) | Ground truth | Retrieved context | Answer | Diagnosis |
|---|---|---|---|---|---|
| "what are the net revenues in investment management in 2016, in billions?" | LangChain, LlamaIndex, DSPy (all three) | $6.22 billion | Explicitly states *"net revenues in investment management were $5.79 billion for 2016, 7% lower than 2015"* | **$5.79 billion** (all three, independently) | All three frameworks retrieved the correct passage and correctly transcribed the number it contains. Ground truth says $6.22B, which appears nowhere in any retrieved context — the closest nearby figure is $6.21B for *2015*, one context passage over. This looks like a RAGBench label error (possibly a mixed-up year), not a retrieval or generation failure. Scored as wrong (F1≈0.14) for all three frameworks alike. |

**Conclusion:** worth flagging in Limitations — the benchmark's ground truth isn't infallible, and at least one case exists where all three pipelines agreed with the source document against the label. Given this was found by spot-checking ~20 cases, there are likely more.

## Real generation failures (retrieval succeeded, answer still wrong)

| Query | Framework | Retrieved context | Answer | Diagnosis |
|---|---|---|---|---|
| "How many people have persistent hepatitis C virus?" | LlamaIndex | Contains "as many as 160 million people are chronically infected worldwide" | "It's estimated that the majority of infected patients fail to clear the virus and develop chronic persistent infection." | The number was sitting in the retrieved context. Model answered qualitatively instead of extracting it — a real generation failure, not a retrieval one. |
| "what was the minimum amount of foreign currency translation loss, in millions?" | LlamaIndex | Financial table with multiple line items/years | GT: $6.3 million — Answer: **-227 (227)** | Wrong row/column pulled from a multi-column table — genuine numeric-extraction error. |
| "what portion of the total restricted units will vest in 2011?" | LlamaIndex | Vesting schedule across 2009-2010 grants | GT: 8000 units — Answer trails off into reasoning about vesting periods without stating a final number | Incomplete reasoning chain — started the calculation, never closed it out. |
| "What is the treatment for MERS-COV?" | LlamaIndex | Passage naming experimental antivirals (interferon, ribavirin, chloroquine) tested in cell culture | GT emphasizes supportive/symptomatic care, no specific approved treatment — Answer lists the experimental antivirals as if they were the treatment | Content is real and present in context, but the answer adopts the wrong framing (experimental-drug list vs. standard-of-care statement) — a question-interpretation mismatch rather than hallucination. |

## OOD / adversarial: confident hallucination vs. correct refusal

Real cases from `results/adversarial_*.json`, judged by Qwen3-14B (`failure_mode` field).

| Query | Framework | Query type | Behavior | Diagnosis |
|---|---|---|---|---|
| "How does WebSphere Transformation Extender (WTX) relate to the historical development of integration technologies?" | DSPy | out_of_distribution | Confidently answered: *"...Originating from the Ascential DataStage TX suite, WTX reflects the evolution toward..."* — fabricated, not in corpus | `confident_hallucination`. Nothing in the retrieved corpus supports this historical claim; DSPy's ChainOfThought produced a fluent, plausible-sounding invention instead of refusing. |
| "What are the legal implications of using COBOL copybooks in software development?" | LlamaIndex | out_of_distribution | Answered with detailed (fabricated) legal analysis | `confident_hallucination`. Confirms this failure mode isn't unique to DSPy's CoT — LlamaIndex's standard retrieve-then-generate also fabricates under OOD pressure. |
| "Is Microsoft Edge not supported by IBM Content Collector at any version?" | DSPy, LangChain (both) | contradictory | Both answered as if the false premise ("not supported at any version") were true, when context actually shows Edge *is* supported from a specific version | `confident_hallucination` for both. Shared failure mode across frameworks on negated/contradictory premises — not framework-specific. |
| "What are the compatibility details for WebSphere Transformation Extender (WTX) and IBM Integration Bus V10?" | DSPy | multi_hop | "The provided context does not include specific compatibility details... recommended to consult..." | `correct_refusal` — included for balance. DSPy does correctly decline when it recognizes the gap, just not consistently (see OOD row above). Refusal behavior is present but not reliable. |

## Retrieval-granularity artifact (not a retrieval failure — resolved during this review)

| Query | Framework | Initial signal | Root cause | Resolution |
|---|---|---|---|---|
| "Is WebSphere Transformation Extender (WTX) supported for IBM Integration Bus V10?" | LangChain | Reference-document overlap scored 0.46 (below 0.6 match threshold) using a one-directional "does the full doc appear in the chunk" metric | LangChain's Chroma index chunks techqa docs to ~1000 chars (`chunk_overlap=200`); the retrieved chunk was a genuine, correct sub-section of a longer relevant document (3793 chars), not a bad retrieval. The one-directional overlap metric penalized the smaller chunk unfairly. | Fixed `run_retrieval_overlap.py` to use symmetric containment (either direction can clear the threshold). LangChain's techqa score moved from 0.26 → 0.84 (see README), landing in the same band as LlamaIndex/DSPy. No real per-framework retrieval-quality gap on techqa — see README's chunk-size note in Framework Comparison for the underlying (deliberate) pipeline difference. |

---

**What this doesn't cover:** this is a spot-check (18 cases) selected via automatic flagging (F1/context-coverage disagreement, `confident_hallucination` failure mode) plus manual reading — not a systematic random sample. It's meant to demonstrate concrete failure/success patterns behind the aggregate numbers, not to be a statistically representative error rate. A larger stratified random sample, ideally with a second human rater, would be needed for that.
