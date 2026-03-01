"""
DSPy RAG Pipeline — v3.x API

DSPy v3 changed significantly from v2:
  - dspy.LM('openai/gpt-4o-mini') replaces dspy.OpenAI()
  - dspy.configure(lm=lm) replaces dspy.settings.configure()
  - dspy.Embedder + dspy.retrievers.Embeddings replaces ChromadbRM
  - No more dspy.Retrieve module — retrieval is just a Python function call
  - dspy.ChainOfThought('context, question -> response') is the clean pattern

The interesting thing about DSPy vs LangChain/LlamaIndex:
  Instead of manually writing prompts, you define input/output signatures
  and DSPy figures out the prompt structure. You can also run an optimizer
  (MIPROv2, BootstrapFewShot) to automatically improve prompts — we run
  unoptimized here for a fair baseline comparison, and note optimization
  as future work / the adversarial blind spot experiment.
"""

import time
from typing import Any

import dspy
from dotenv import load_dotenv

load_dotenv()


class RAGModule(dspy.Module):
    """
    Simple DSPy RAG module.
    Takes retrieved context + question, produces a response.
    ChainOfThought adds reasoning before committing to the answer.

    forward() returns both the response AND the retrieved passages
    so query() only needs one search call per query — no double embedding.
    """
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question: str, search_fn):
        result = search_fn(question)
        passages = result.passages
        prediction = self.respond(context=passages, question=question)
        # Attach passages to prediction so query() can read them
        prediction.passages = passages
        return prediction


class DSPyRAG:
    def __init__(self, model: str = "gpt-4o-mini", k: int = 4):
        self.model_name = model
        self.k = k
        self.rag_module = None
        self.search = None
        self.lm = None

    def build(self, documents: list[dict]) -> None:
        """
        Index documents and configure DSPy with token-limit protection.
        cache=False ensures real API calls are made each query —
        without this DSPy's LiteLLM cache returns sub-millisecond responses
        that are not comparable to LangChain/LlamaIndex latency.
        """
        self.lm = dspy.LM(
            f"openai/{self.model_name}",
            temperature=0,
        )
        dspy.configure(lm=self.lm, cache=False)  # cache=False for real latency measurement

        # Pre-process corpus and truncate clearly over-sized docs
        # 1 token ≈ 4 chars, so 28,000 chars is a safe buffer for the 8,191 limit
        corpus = []
        for doc in documents:
            content = doc.get('content', '')
            if content:
                text = f"Title: {doc['title']}\n\n{content}"
                corpus.append(text[:28000])

        embedder = dspy.Embedder(
            'openai/text-embedding-3-small',
            dimensions=512,
        )

        print(f"Building DSPy index for {len(corpus)} documents...")

        # BATCH_SIZE 10 is much safer for long technical/financial docs
        BATCH_SIZE = 10
        all_embeddings = []
        final_corpus = []  # track corpus in lockstep with embeddings

        for i in range(0, len(corpus), BATCH_SIZE):
            batch = corpus[i:i + BATCH_SIZE]
            try:
                batch_embeddings = embedder(batch)
                all_embeddings.extend(batch_embeddings)
                final_corpus.extend(batch)  # only add if embedding succeeded
            except Exception as e:
                print(f"  Batch {i//BATCH_SIZE} failed (likely token limit), falling back to sequential...")
                for single_doc in batch:
                    try:
                        single_emb = embedder([single_doc])
                        all_embeddings.extend(single_emb)
                        final_corpus.append(single_doc)  # add in lockstep
                    except Exception as single_e:
                        print(f"  Skipping a document that is too long: {len(single_doc)} chars")
                        continue  # skip both doc and embedding — stays aligned

        import numpy as np
        import faiss

        emb_matrix = np.array(all_embeddings).astype('float32')
        index = faiss.IndexFlatL2(emb_matrix.shape[1])
        index.add(emb_matrix)

        class BatchedEmbeddingsRetriever:
            def __init__(self, embedder, corpus, index, k):
                self.embedder = embedder
                self.corpus = corpus
                self.index = index
                self.k = k

            def __call__(self, query: str):
                q_emb = np.array(self.embedder([query])).astype('float32')
                _, indices = self.index.search(q_emb, self.k)
                passages = [self.corpus[i] for i in indices[0] if i < len(self.corpus)]

                class Result:
                    pass
                r = Result()
                r.passages = passages
                return r

        self.search = BatchedEmbeddingsRetriever(
            embedder=embedder,
            corpus=final_corpus,
            index=index,
            k=self.k,
        )

        self.rag_module = RAGModule()

    # NOTE: DSPy's unique value — you can optimize prompts automatically:
    # from dspy.teleprompt import MIPROv2
    # optimizer = MIPROv2(metric=your_metric, auto="light")
    # self.rag_module = optimizer.compile(self.rag_module, trainset=trainset)
    #
    # Running unoptimized for fair baseline comparison.

    def query(self, question: str) -> dict[str, Any]:
        """
        Run a query — single search call, contexts returned from module.
        No double embedding call — latency timer covers the full operation.
        """
        if self.rag_module is None:
            raise RuntimeError("Call build() first.")

        start = time.perf_counter()
        prediction = self.rag_module(question=question, search_fn=self.search)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": prediction.response,
            "contexts": prediction.passages,
            "latency_ms": latency_ms,
            "framework": "dspy",
        }