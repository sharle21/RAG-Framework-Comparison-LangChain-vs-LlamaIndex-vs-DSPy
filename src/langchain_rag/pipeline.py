"""
LangChain RAG Pipeline
LCEL chain with Chroma vector store and OpenAI embeddings.

Single retrieval call per query using RunnableParallel + assign pattern:
  - retrieves docs and passes question simultaneously
  - assigns answer from those docs
  - returns both answer and contexts from one invoke() call
  - no double embedding call, fair latency comparison with LlamaIndex/DSPy

build() accepts persist_dir and collection_name so the adversarial
poisoning test can build clean and poisoned instances in separate
Chroma namespaces without collision.

Chroma v0.4+ auto-persists when persist_directory is set —
no explicit vectorstore.persist() call needed or available.
"""

import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIR = Path(__file__).parent.parent.parent / "data" / "chroma_langchain"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class LangChainRAG:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        base_url: str = None,
        local_embeddings: bool = False,
    ):
        self.model_name = model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.local_embeddings = local_embeddings
        self.retrieval_chain = None
        self.vectorstore = None

    def build(
        self,
        documents: list[dict],
        persist_dir: Path = None,
        collection_name: str = "langchain_rag",
    ) -> None:
        """
        Index documents and build the RAG chain.

        persist_dir: override default Chroma storage location.
            Used by adversarial poisoning test to keep clean and
            poisoned indexes in separate directories.
        collection_name: Chroma collection namespace.
            Must differ between clean and poisoned instances even
            if persist_dir differs — belt and suspenders.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        texts = []
        metadatas = []
        for doc in documents:
            chunks = splitter.split_text(doc.get("content", ""))
            texts.extend(chunks)
            meta = {"id": doc["id"], "title": doc["title"]}
            if doc.get("is_noise"):
                meta["is_noise"] = True
            metadatas.extend([meta] * len(chunks))

        if self.local_embeddings:
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        else:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=collection_name,
            persist_directory=str(persist_dir or PERSIST_DIR),
        )
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        llm_kwargs = {"model": self.model_name, "temperature": 0}
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
            llm_kwargs["api_key"] = "none"
        llm = ChatOpenAI(**llm_kwargs)
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
        )

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        )

        self.retriever = retriever
        self.generation_chain = rag_chain_from_docs
        self.retrieval_chain = RunnableParallel(
            context=retriever,
            question=RunnablePassthrough(),
        ).assign(answer=rag_chain_from_docs)

    def query(self, question: str) -> dict[str, Any]:
        """Run a query with separate retrieval and generation timers."""
        if self.retrieval_chain is None:
            raise RuntimeError("Call build() first.")

        # Step 1: retrieval — embed question + search Chroma
        t0 = time.perf_counter()
        docs = self.retriever.invoke(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Deduplicate retrieved chunks
        seen = set()
        unique_contexts = []
        for doc in docs:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                unique_contexts.append(content)
            if len(unique_contexts) == 4:
                break

        # Check if any retrieved doc is a noise (false) document
        noise_retrieved = any(doc.metadata.get("is_noise", False) for doc in docs)

        # Step 2: generation — format prompt + LLM call
        t1 = time.perf_counter()
        answer = self.generation_chain.invoke({"context": docs, "question": question})
        generation_ms = (time.perf_counter() - t1) * 1000

        return {
            "answer": answer,
            "contexts": unique_contexts,
            "retrieved_noise": noise_retrieved,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "latency_ms": retrieval_ms + generation_ms,
            "framework": "langchain",
        }