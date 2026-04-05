"""
LlamaIndex RAG Pipeline
Uses VectorStoreIndex with SimpleDirectoryReader pattern adapted for in-memory docs.
"""

import time
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()


class LlamaIndexRAG:
    def __init__(self, model: str = "gpt-4o-mini", chunk_size: int = 1000, chunk_overlap: int = 200, base_url: str = None):
        self.model_name = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.query_engine = None
        self.index = None

        # Configure global settings
        llm_kwargs = {"model": model, "temperature": 0}
        if base_url:
            llm_kwargs["api_base"] = base_url
            llm_kwargs["api_key"] = "none"
        Settings.llm = OpenAI(**llm_kwargs)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def build(self, documents: list[dict]) -> None:
        """Index documents and build the query engine."""
        llama_docs = [
            Document(
                text=doc.get("content", ""),
                metadata={"id": doc["id"], "title": doc["title"]},
            )
            for doc in documents
        ]

        self.index = VectorStoreIndex.from_documents(llama_docs, show_progress=True)
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=4,
            response_mode="compact",
        )
        # Store retriever and synthesizer separately for split latency timing
        self.retriever = self.index.as_retriever(similarity_top_k=4)
        self.synthesizer = get_response_synthesizer(response_mode="compact")

    def query(self, question: str) -> dict[str, Any]:
        """Run a query with separate retrieval and generation timers."""
        if self.query_engine is None:
            raise RuntimeError("Call build() first.")

        # Step 1: retrieval — embed question + search in-memory vector index
        t0 = time.perf_counter()
        nodes = self.retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Step 2: generation — synthesizer reads nodes and calls the LLM
        t1 = time.perf_counter()
        response = self.synthesizer.synthesize(question, nodes=nodes)
        generation_ms = (time.perf_counter() - t1) * 1000

        contexts = [node.get_content() for node in nodes]

        return {
            "answer": str(response),
            "contexts": contexts,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "latency_ms": retrieval_ms + generation_ms,
            "framework": "llamaindex",
        }