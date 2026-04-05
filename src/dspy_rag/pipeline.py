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


class OptimizableRAGModule(dspy.Module):
    """
    Variant of RAGModule used during MIPROv2 optimization.

    RAGModule.forward() takes (question, search_fn) — but MIPROv2 only knows
    about the fields in your trainset. It can't pass search_fn. So we bake
    search_fn into __init__ and expose forward(question) only, which is the
    signature the optimizer expects.

    After optimization, we copy optimized.respond back into the main rag_module
    so the regular query() path benefits from the improved prompt.
    """
    def __init__(self, search_fn):
        self.search_fn = search_fn
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question: str):
        result = self.search_fn(question)
        return self.respond(context=result.passages, question=question)


class DSPyRAG:
    def __init__(self, model: str = "gpt-4o-mini", k: int = 4, base_url: str = None):
        self.model_name = model
        self.base_url = base_url
        self.k = k
        self.rag_module = None
        self.search = None
        self.lm = None
        self._optimized = False

    def build(self, documents: list[dict]) -> None:
        """
        Index documents and configure DSPy with token-limit protection.
        cache=False ensures real API calls are made each query —
        without this DSPy's LiteLLM cache returns sub-millisecond responses
        that are not comparable to LangChain/LlamaIndex latency.
        """
        lm_kwargs = {"temperature": 0}
        if self.base_url:
            lm_kwargs["api_base"] = self.base_url
            lm_kwargs["api_key"] = "none"
        self.lm = dspy.LM(f"openai/{self.model_name}", **lm_kwargs)
        dspy.configure(lm=self.lm, cache=False)  # cache=False for real latency measurement

        # Pre-process corpus and truncate clearly over-sized docs
        # 1 token ≈ 4 chars, so 28,000 chars is a safe buffer for the 8,191 limit
        # Also track which corpus entries are noise docs (DSPy has no metadata —
        # we store the formatted string and check membership in query())
        corpus = []
        self._noise_texts = set()
        for doc in documents:
            content = doc.get('content', '')
            if content:
                text = f"Title: {doc['title']}\n\n{content}"
                text = text[:28000]
                corpus.append(text)
                if doc.get("is_noise"):
                    self._noise_texts.add(text)

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

    def optimize(self, qa_pairs: list[dict], n_train: int = 20) -> None:
        """
        Run MIPROv2 prompt optimization using QA pairs as training examples.

        MIPROv2 generates candidate instruction variants, evaluates each one
        against a fast metric (token F1 — no extra LLM cost), and picks the
        best. The winning prompt replaces the default ChainOfThought prompt.

        auto="light" tries ~5-10 candidates (cheap). Use "medium" or "heavy"
        for a more thorough search at higher API cost.

        This is DSPy's core value proposition: instead of hand-writing prompts,
        you show it examples and it figures out the prompt structure itself.
        """
        if self.rag_module is None:
            raise RuntimeError("Call build() first.")

        from dspy.teleprompt import MIPROv2

        # Trainset: each example is a question with its expected answer
        trainset = [
            dspy.Example(
                question=qa["question"],
                response=qa["ground_truth"],
            ).with_inputs("question")
            for qa in qa_pairs[:n_train]
        ]

        # Token F1 metric — fast and deterministic, no LLM cost during search
        def f1_metric(example, prediction, trace=None):
            pred = getattr(prediction, "response", "") or ""
            gt = example.response or ""
            pred_tokens = set(pred.lower().split())
            gt_tokens = set(gt.lower().split())
            if not pred_tokens or not gt_tokens:
                return 0.0
            common = pred_tokens & gt_tokens
            if not common:
                return 0.0
            p = len(common) / len(pred_tokens)
            r = len(common) / len(gt_tokens)
            return 2 * p * r / (p + r)

        # OptimizableRAGModule bakes in search_fn so the optimizer only sees question
        optimizable = OptimizableRAGModule(self.search)

        print(f"  MIPROv2: optimizing on {len(trainset)} examples (auto='light')...")
        optimizer = MIPROv2(metric=f1_metric, auto="light")
        optimized = optimizer.compile(
            optimizable,
            trainset=trainset,
            requires_permission_to_run=False,
        )

        # Copy the optimized ChainOfThought prompt back into the main rag_module.
        # MIPROv2 updates respond.predict.extended_signature (instructions + demos).
        self.rag_module.respond = optimized.respond
        self._optimized = True
        print("  MIPROv2 done — optimized prompt installed.")

    def query(self, question: str) -> dict[str, Any]:
        """
        Run a query with separate retrieval and generation timers.
        Calls search and ChainOfThought respond directly instead of
        going through RAGModule.forward() so we can put a timer between them.
        """
        if self.rag_module is None:
            raise RuntimeError("Call build() first.")

        # Step 1: retrieval — embed question + FAISS nearest-neighbor search
        t0 = time.perf_counter()
        search_result = self.search(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Step 2: generation — DSPy ChainOfThought calls the LLM
        t1 = time.perf_counter()
        prediction = self.rag_module.respond(
            context=search_result.passages,
            question=question,
        )
        generation_ms = (time.perf_counter() - t1) * 1000

        # Check if any retrieved passage is a noise doc (string membership check)
        noise_retrieved = any(p in self._noise_texts for p in search_result.passages)

        return {
            "answer": prediction.response,
            "contexts": search_result.passages,
            "retrieved_noise": noise_retrieved,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "latency_ms": retrieval_ms + generation_ms,
            "framework": "dspy_optimized" if self._optimized else "dspy",
        }