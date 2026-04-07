"""
Unified RAG HTTP server — wraps all three frameworks behind POST /query.

Each framework runs as its own process on its own port. The Go orchestrator
hits them concurrently — goroutines keep the GPU saturated while Python
handles RAG-specific logic (chunking, vector search, prompt construction).

The index is built ONCE at startup and held in memory. Every incoming
request reuses the same index — no rebuild per query.

Usage (run once per framework, in separate terminals or via run_servers.sh):

    python src/rag_server.py --framework langchain   --port 8100 --base-url http://localhost:8000/v1 --local-embeddings
    python src/rag_server.py --framework llamaindex  --port 8101 --base-url http://localhost:8000/v1 --local-embeddings
    python src/rag_server.py --framework dspy        --port 8102 --base-url http://localhost:8000/v1 --local-embeddings

Then point the Go orchestrator at e.g. http://localhost:8100 instead of the vLLM URL directly.
"""

import argparse
import json
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Pipeline is imported lazily inside lifespan so --framework controls which one loads ──
_rag = None
_framework: str = ""


def _load_documents() -> list[dict]:
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "raw" / "ragbench_documents.json") as f:
        return json.load(f)


def _build_pipeline(args: argparse.Namespace):
    """Instantiate and build the right pipeline based on --framework."""
    kwargs = dict(
        model=args.model,
        base_url=args.base_url,
        local_embeddings=args.local_embeddings,
    )

    if args.framework == "langchain":
        from src.langchain_rag.pipeline import LangChainRAG
        return LangChainRAG(**kwargs)

    if args.framework == "llamaindex":
        from src.llamaindex_rag.pipeline import LlamaIndexRAG
        return LlamaIndexRAG(**kwargs)

    if args.framework == "dspy":
        from src.dspy_rag.pipeline import DSPyRAG
        return DSPyRAG(**kwargs)

    raise ValueError(f"Unknown framework: {args.framework!r}. Choose langchain, llamaindex, or dspy.")


def make_app(args: argparse.Namespace) -> FastAPI:
    global _rag, _framework
    _framework = args.framework

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _rag
        try:
            documents = _load_documents()
            print(f"[{args.framework}] Loaded {len(documents)} documents")
            print(f"[{args.framework}] Building index (model={args.model}, local_embeddings={args.local_embeddings})...")
            t0 = time.perf_counter()
            rag = _build_pipeline(args)
            rag.build(documents)
            _rag = rag  # assign only after build completes — health check returns ready=True from this point
            print(f"[{args.framework}] Index ready in {time.perf_counter() - t0:.1f}s — listening on :{args.port}")
        except Exception:
            print(f"[{args.framework}] STARTUP FAILED:")
            traceback.print_exc()
            raise
        yield
        print(f"[{args.framework}] Shutting down.")

    return FastAPI(lifespan=lifespan)


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    domain: str = "unknown"
    ground_truth: str = ""


class QueryResponse(BaseModel):
    question: str
    ground_truth: str
    domain: str
    framework: str
    answer: str
    contexts: list[str]
    retrieved_noise: bool
    retrieval_ms: float
    generation_ms: float
    latency_ms: float
    error: str = ""


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified RAG HTTP server")
    parser.add_argument("--framework", required=True, choices=["langchain", "llamaindex", "dspy"])
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None, help="vLLM base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--local-embeddings", action="store_true", help="Use bge-m3 instead of OpenAI embeddings")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = make_app(args)

    @app.get("/health")
    def health():
        return {"status": "ok", "framework": _framework, "ready": _rag is not None}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        if _rag is None:
            raise HTTPException(status_code=503, detail="Index not ready")
        try:
            result = _rag.query(req.question)
            return QueryResponse(
                question=req.question,
                ground_truth=req.ground_truth,
                domain=req.domain,
                **result,
            )
        except Exception as e:
            return QueryResponse(
                question=req.question,
                ground_truth=req.ground_truth,
                domain=req.domain,
                framework=_framework,
                answer="",
                contexts=[],
                retrieved_noise=False,
                retrieval_ms=0,
                generation_ms=0,
                latency_ms=0,
                error=str(e),
            )

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
