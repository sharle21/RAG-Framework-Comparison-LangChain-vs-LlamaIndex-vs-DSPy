# ADR-0001: Framework Selection — LangChain, LlamaIndex, DSPy

**Status:** Accepted

## Context

### What these frameworks actually are

These are not three competing "RAG frameworks." They have different primary purposes:

- **LangChain** — general-purpose LLM application framework. RAG is one pattern among many (agents, tools, chains, memory). Its core abstraction is composition: connect LLM calls, retrievers, and tools into chains.
- **LlamaIndex** — data indexing and retrieval layer for LLMs. RAG is closer to its primary purpose, but it also handles agents, multi-modal retrieval, and structured data queries. Core abstraction is the index: how documents are stored and queried.
- **DSPy** — programmatic LM pipeline optimizer. RAG is not a primary abstraction — it is one program you write in DSPy. The framework's core value is that pipelines are optimizable (via MIPROv2, BootstrapFewShot) rather than just composable. Prompts are not written by hand; they are learned from examples.

### What this benchmark compares

We are comparing their **RAG capability specifically** — not their general utility, ecosystem breadth, developer ergonomics, or production reliability. The question is:

> When used to answer questions from a retrieved document corpus, how does each framework's retrieval strategy and generation approach affect answer quality?

This is a narrower question than "which framework is better."

### What makes the comparison valid

All three frameworks are capable of RAG. To isolate the framework's contribution, we control every other variable:

| Variable | Controlled value |
|----------|-----------------|
| Base LLM | Llama-3.1-8B-Instruct (same for all) |
| Embedding model | bge-m3 1024-dim (same for all) |
| Dataset | RAGBench — covidqa, techqa, finqa (same for all) |
| Hardware | Same Lambda H100 instance |
| Judge | Qwen3-14B (same for all) |

**What actually varies** across frameworks:
- Vector store implementation (Chroma vs in-memory vs FAISS)
- Retrieval indexing and search logic
- Prompting strategy (standard retrieve-then-generate vs ChainOfThought)
- How retrieved context is passed to the LLM

The embedding model and LLM are deliberately controlled away. We are not testing bge-m3 vs OpenAI embeddings or Llama vs GPT — we are testing what the framework does with the same model and data.

### What we can and cannot claim

**Can claim:**
- Answer quality (correctness, faithfulness, completeness) on RAGBench with Llama-3.1-8B + bge-m3
- Relative latency under concurrent load on H100
- OOD refusal behavior (directional, n=30)
- That ChainOfThought generates more tokens, inflating string-overlap metrics

**Cannot claim:**
- Retrieval quality independently — doc IDs were not stored, so Recall@k, Precision@k, MRR, NDCG cannot be computed. Context coverage (fraction of ground-truth tokens in retrieved passages) is a proxy, not a formal retrieval metric.
- Generalization beyond RAGBench's three domains
- Results with a different LLM or embedding model
- Developer experience, maintainability, community support, production reliability

## Decision

Compare LangChain, LlamaIndex, and DSPy on their RAG capability under controlled conditions.

These three are a meaningful comparison set because:
1. All three are production choices engineers face when building document QA systems
2. They represent three distinct approaches to the retrieve-then-generate problem: chain composition, index-first retrieval, and programmatic optimization
3. Controlling for LLM + embeddings makes the comparison attributable to the framework's retrieval and prompting strategy

## Consequences

- DSPy's ChainOfThought produces longer outputs, causing string metrics to rank it first and judge metrics to rank it last. This divergence is a finding, not a flaw — it reveals that string metrics reward verbosity regardless of grounding.
- DSPy's programmatic approach enables MIPROv2 prompt optimization, which LangChain and LlamaIndex cannot replicate without significant custom code. The optimization experiment showed degraded performance, which is itself a finding about automated prompt search.
- The controlled comparison means findings are specifically about "LangChain's default RAG pipeline with bge-m3 on RAGBench," not about LangChain as a framework in general.
