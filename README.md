# RAG Framework Comparison: LangChain vs LlamaIndex vs DSPy

A systematic benchmark comparing three production RAG frameworks across 90 questions, three evaluation methods, and adversarial stress testing. The central question: **does the ranking of RAG frameworks change depending on which metric you use to evaluate them?**

It does.

---

## What This Is

Most RAG comparisons pick one metric and declare a winner. This project runs the same outputs through three independent evaluation methods — a no-LLM string baseline, a cross-family LLM judge, and RAGAS — and shows that the ranking shifts depending on what you measure. LlamaIndex wins on correctness. LangChain wins on faithfulness. The "best" framework depends on whether your application can tolerate hallucinations.

The evaluation design ended up being harder than the pipelines themselves.

---

## Frameworks

**LangChain** — The most widely used RAG framework. Uses LCEL (LangChain Expression Language) chains with Chroma vector store and OpenAI embeddings. Known for its extensive ecosystem and composability. In this benchmark it is the most conservative framework: high faithfulness, near-zero hallucination rate, but weaker retrieval diversity.

**LlamaIndex** — Purpose-built for document indexing and retrieval. Has more sophisticated default chunking and retrieval strategies than LangChain. Wins on answer correctness and context recall in this benchmark, but trades off a higher hallucination rate when retrieved context is ambiguous.

**DSPy** — Fundamentally different philosophy. Instead of writing prompts manually, you define input/output signatures and DSPy figures out the prompt structure. Supports automatic prompt optimization via MIPROv2 (not run here — see Future Work). Uses a custom FAISS-based retriever with OpenAI embeddings. Run unoptimized for fair baseline comparison.

---

## Design Choices

**Why three evaluation methods?**

String overlap (token F1) is fast, reproducible, and requires no API calls — but it penalizes correct answers that use different wording. LLM-as-judge is more semantically aware but introduces model bias. RAGAS provides standardized RAG-specific metrics but is slower and more expensive. Running all three on the same outputs lets you see where they agree and disagree. When they disagree, that's information.

**Why Mistral as the judge for GPT-4o-mini outputs?**

Same-family self-evaluation is a known bias — OpenAI models score OpenAI outputs more generously. Using Mistral Large (different company, different training lineage, different RLHF) as the judge avoids this. RAGAS runs separately on OpenAI because it is tightly coupled to the OpenAI SDK.

**Why 90 questions across 3 domains?**

90 pairs balanced across covidqa, finqa, and techqa gives enough data for directional conclusions while keeping API costs manageable. A naive slice of the RAGBench dataset would give heavily techqa-skewed results — the sampling is explicitly balanced. 90 is not enough for statistical significance claims, but it is enough to see consistent patterns across three domains and three evaluation methods.

**Why adversarial testing?**

Standard benchmarks only test whether a framework answers known questions correctly. They don't test what happens when queries are ambiguous, contradictory, out-of-distribution, or require multi-hop reasoning. The adversarial suite generates four hard query types from the existing test set and measures degradation ratio (adversarial performance / standard performance). A ratio close to 1.0 means the framework handles distribution shift gracefully.

---

## Results

### Answer Quality

| Metric | LangChain | LlamaIndex | DSPy |
|---|---|---|---|
| Answer F1 (string) | 0.536 | **0.572** | 0.537 |
| Correctness (Mistral judge) | 0.644 | **0.698** | 0.611 |
| Faithfulness (Mistral judge) | **0.962** | 0.842 | 0.832 |
| Completeness (Mistral judge) | 0.581 | **0.642** | 0.533 |
| RAGAS Faithfulness | 0.762 | **0.766** | 0.622 |
| RAGAS Answer Relevancy | 0.569 | **0.684** | 0.600 |
| RAGAS Context Recall | 0.469 | **0.633** | 0.618 |

### Failure Modes (n=90)

| Framework | Correct | Incomplete | Wrong Context | Hallucination |
|---|---|---|---|---|
| LangChain | 26.7% | 46.7% | 21.1% | **2.2%** |
| LlamaIndex | 18.9% | 50.0% | 17.8% | 12.2% |
| DSPy | 14.4% | 53.3% | 26.7% | 5.6% |

LlamaIndex has the highest correct rate on the LLM judge (0.698) but also the highest hallucination rate (12.2%). It synthesizes from context more aggressively — which helps when context is relevant and hurts when it is ambiguous. LangChain's conservative retrieval makes it safe but incomplete. DSPy's ChainOfThought reasoning tends to produce hedged, partial answers rather than wrong ones.

### Adversarial Robustness

| Framework | Standard | Adversarial | Degradation Ratio |
|---|---|---|---|
| LangChain | 0.644 | 0.765 | 1.19× |
| LlamaIndex | 0.698 | 0.716 | **1.03×** |
| DSPy | 0.611 | 0.703 | 1.15× |

LlamaIndex's behavior barely changes under adversarial pressure (1.03×). LangChain's top adversarial failure mode is `correct_refusal` — it appropriately declines to answer out-of-distribution questions rather than fabricating, which the adversarial judge scores positively.

### A Note on the Deduplication Fix

LangChain's initial correctness score was 0.518. After fixing a retrieval deduplication bug — Chroma was returning the same chunk multiple times, filling all k=4 retrieval slots with copies of the same passage — correctness jumped to 0.644 and context coverage went from 0.430 to 0.561. The framework was not underperforming. The benchmark was measuring a configuration bug. This is the kind of issue that doesn't show up if you only run one evaluation method.

---

## Methodology

### Data

RAGBench dataset from HuggingFace, balanced across three domains:
- **covidqa** — biomedical QA on COVID-19 literature
- **finqa** — financial document QA
- **techqa** — IBM technical support documentation

30 questions per domain, 90 total. Documents chunked at 1000 characters with 200 character overlap.

### Evaluation Pipeline

```
results_langchain.json ─┐
results_llamaindex.json ─┼─▶ string_overlap ──┐
results_dspy.json ───────┘   llm_judge ────────┼──▶ summary.json
                             ragas ────────────┤
                             failure_modes ────┘
```

Each framework produces answers and retrieved contexts for all 90 questions. The same outputs are then passed to all three evaluators independently. No evaluator sees the results of another.

### Adversarial Query Generation

Four query types generated by GPT-4o-mini from the existing test set:
- **Multi-hop**: requires synthesizing across multiple documents
- **Ambiguous**: underspecified, missing key context
- **Out-of-distribution**: plausibly related but not in the corpus
- **Contradictory**: frames the question as if the ground truth is false

Adversarial queries are saved to `data/adversarial_queries.json` for reproducibility. Evaluation uses a separate robustness judge prompt that scores appropriate refusals positively.

---

## Project Structure

```
├── src/
│   ├── langchain_rag/
│   │   └── pipeline.py          # LCEL chain, Chroma retriever, dedup fix
│   ├── llamaindex_rag/
│   │   └── pipeline.py          # LlamaIndex VectorStoreIndex
│   ├── dspy_rag/
│   │   └── pipeline.py          # DSPy ChainOfThought, FAISS retriever
│   └── evaluation/
│       ├── metrics.py           # String overlap, LLM judge, RAGAS
│       ├── adversarial_agent.py # Query generation + robustness eval
│       ├── run_benchmark.py     # Main runner
│       ├── prepare_data.py      # RAGBench download + preprocessing
│       └── synthetic_data.py    # Synthetic QA generation (optional)
├── data/
│   ├── raw/                     # RAGBench source documents
│   ├── qa_pairs.json            # 90 balanced test questions
│   └── adversarial_queries.json # Generated hard queries
├── results/
│   ├── results_{framework}.json # Per-query answers + contexts
│   ├── adversarial_{framework}.json
│   └── summary.json             # Aggregated scores + rankings
└── rerun_failure_modes.py       # Rerun failure analysis on saved results
```

---

## Running It

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add: OPENAI_API_KEY, MISTRAL_API_KEY

# Download and prepare data
python src/evaluation/prepare_data.py

# Run benchmark (90 pairs, all three frameworks)
python src/evaluation/run_benchmark.py --n-pairs 90

# Run adversarial stress test
python src/evaluation/run_benchmark.py --n-pairs 90 --adversarial

# Rerun failure mode analysis on saved results (no re-querying)
python rerun_failure_modes.py
```

**Cost estimate:** Full 90-pair benchmark + adversarial ≈ $2–3 in OpenAI + Mistral API calls.

---

## What's Novel

Most RAG benchmarks do one of these things. This project does all of them together:

1. **Multi-method evaluation on the same outputs** — ranking instability across metrics is itself a finding, not a flaw in the methodology
2. **Cross-family judging** — Mistral evaluating GPT-4o-mini outputs to avoid same-family bias
3. **Adversarial stress testing with four distinct query types** — not just "hard questions" but specific failure categories
4. **Full 90-sample failure mode classification** — most failure analyses use small samples that miss rare failure categories like hallucination
5. **Explicit retrieval bug detection** — the deduplication finding demonstrates that benchmark results can reflect configuration issues rather than framework quality

---

## Future Work

**MIPROv2 optimization + adversarial evaluation**

The most interesting next experiment: compile DSPy with MIPROv2 using the existing QA pairs as training data, then run the compiled program through the same adversarial suite. The question is whether an optimized prompt that is tightly tuned to the training distribution becomes more brittle under distribution shift, or whether the better prompt generalizes. Almost nobody has adversarially evaluated compiled vs uncompiled DSPy behavior. Running `auto="light"` with gpt-4o-mini would cost approximately $2 and take 20 minutes.

**GraphRAG evaluation**

Replace the flat vector retrieval with a knowledge graph retrieval layer (Microsoft GraphRAG or similar). Multi-hop questions in this benchmark — where the answer requires connecting information across documents — are the hardest failure category. Graph-based retrieval is specifically designed for this. Testing whether it closes the multi-hop gap is a direct follow-on.

**Distribution shift testing**

All three frameworks were indexed and evaluated on the same distribution (RAGBench). Testing what happens when the query distribution shifts — e.g., index on techqa, query with covidqa terminology — would surface retrieval brittleness that standard benchmarks hide.

**Agentic red-teaming**

Replace the static adversarial query generator with an iterative red-teaming agent that adapts based on each framework's responses. Static adversarial queries test known failure modes. An agent that observes what each framework gets wrong and generates harder follow-up queries would find failure modes that aren't in the predefined taxonomy.

**Synthetic data as a separate evaluation track**

Generate synthetic QA pairs from the same corpus using the RAGAS TestsetGenerator and compare framework performance on synthetic vs real questions. Synthetic questions tend to be cleaner and more answerable — the performance gap between synthetic and real is a measure of how well each framework handles messy real-world queries.

---

## Limitations

- 90 questions supports directional conclusions, not statistically significant claims
- DSPy latency numbers are unreliable due to LiteLLM internal caching — exclude from any latency comparison
- Failure mode percentages are based on LLM classification and should be treated as approximate
- All three frameworks use the same underlying LLM (gpt-4o-mini) for generation — differences reflect retrieval and prompting strategy, not model capability
- Results are specific to this corpus, chunk size, and k=4 retrieval setting

---

## References

- [RAGBench Dataset](https://huggingface.co/datasets/rungalileo/ragbench)
- [DSPy](https://github.com/stanfordnlp/dspy) — Stanford NLP
- [RAGAS](https://docs.ragas.io) — RAG evaluation framework
- Chip Huyen, *Designing Machine Learning Systems* — on evaluation methodology
- [MIPROv2 Paper](https://arxiv.org/abs/2406.11695)
