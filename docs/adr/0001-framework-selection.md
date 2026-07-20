# ADR-0001: Framework Selection — LangChain, LlamaIndex, DSPy

**Status:** Accepted

## Context

The goal is to benchmark production RAG frameworks that represent meaningfully different design philosophies, not just different APIs wrapping the same ideas. Picking three similar frameworks would produce small, noisy differences with little insight.

Candidate frameworks evaluated:

- **LangChain** — chain composition model; most widely adopted RAG framework; large ecosystem; manual prompt construction via LCEL
- **LlamaIndex** — purpose-built for document indexing and retrieval; index-first design philosophy; strong defaults for RAG workflows
- **DSPy** — programmatic prompting; treats the prompt as an optimizable program, not a handwritten template; uses ChainOfThought reasoning by default
- **Haystack** — excluded; similar chain-composition model to LangChain, would not add a distinct paradigm
- **Semantic Kernel** — excluded; primarily .NET/enterprise-oriented, not a common Python RAG choice

## Decision

Compare LangChain, LlamaIndex, and DSPy.

These three represent three distinct paradigms:

| Framework | Paradigm | Core abstraction |
|-----------|----------|-----------------|
| LangChain | Chain composition | LCEL chain: retrieve → format → generate |
| LlamaIndex | Index-first retrieval | VectorStoreIndex with query engine |
| DSPy | Programmatic prompting | Typed signatures + ChainOfThought + optimizer |

Keeping the same base LLM (Llama-3.1-8B-Instruct), same embedding model (bge-m3), and same dataset across all three isolates the framework's contribution from the model's contribution.

## Consequences

- DSPy's ChainOfThought generates significantly more tokens than the other two, making latency comparisons require token-normalized measurement.
- DSPy's programmatic approach enables MIPROv2 prompt optimization — an experiment the other frameworks can't replicate without significant custom code.
- The paradigm difference makes metric disagreements more interpretable: string metrics reward DSPy's verbosity, while judge metrics penalize context drift from the reasoning chain.
