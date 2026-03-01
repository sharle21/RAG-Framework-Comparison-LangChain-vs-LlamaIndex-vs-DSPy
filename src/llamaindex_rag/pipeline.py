"""
LlamaIndex RAG Pipeline
Uses VectorStoreIndex with SimpleDirectoryReader pattern adapted for in-memory docs.
"""

import time
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()


class LlamaIndexRAG:
    def __init__(self, model: str = "gpt-4o-mini", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.query_engine = None
        self.index = None

        # Configure global settings
        Settings.llm = OpenAI(model=model, temperature=0)
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

    def query(self, question: str) -> dict[str, Any]:
        """Run a query and return answer + latency."""
        if self.query_engine is None:
            raise RuntimeError("Call build() first.")

        start = time.perf_counter()
        response = self.query_engine.query(question)
        latency_ms = (time.perf_counter() - start) * 1000

        contexts = [node.get_content() for node in response.source_nodes]

        return {
            "answer": str(response),
            "contexts": contexts,
            "latency_ms": latency_ms,
            "framework": "llamaindex",
        }