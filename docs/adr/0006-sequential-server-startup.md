# ADR-0006: Sequential RAG Server Startup to Prevent OOM

**Status:** Accepted

## Context

Each RAG server embeds the full corpus with bge-m3 on startup. bge-m3 (1024-dim) loads a ~570MB model into memory and processes documents in batches. Three servers starting simultaneously would each try to load bge-m3 and embed concurrently.

On a single 80GB GPU instance shared with two vLLM processes (Llama-3.1-8B + Qwen3-14B using ~78GB combined), RAM is the constraint, not GPU memory. Each embedding process requires ~4–6GB RAM for the model plus batch buffers. Three simultaneous embedding processes on a CPU-only embedding path (bge-m3 runs on CPU to avoid competing with vLLM for GPU) caused OOM on 32GB RAM instances.

**Parallel startup** (original approach):
```bash
python src/rag_server.py --framework langchain &
python src/rag_server.py --framework llamaindex &
python src/rag_server.py --framework dspy &
```
Result: OOM during index build on 32GB RAM instances; race condition on shared resources.

**Sequential startup** (current approach):
```bash
start langchain → wait for /health ready → start llamaindex → wait → start dspy → wait
```

## Decision

Start RAG servers sequentially in `run_servers.sh`. Each server's `/health` endpoint returns `{"ready": true}` only after index build completes. The script polls health before launching the next server.

## Consequences

- Cold startup takes 3× longer (~90s per framework for bge-m3 embedding = ~4.5 minutes total vs ~90s parallel).
- Index persistence on disk (Chroma SQLite, LlamaIndex docstore.json, FAISS file) means subsequent restarts skip the embedding step entirely — sequential startup only applies on first run or after index deletion.
- The `wait_ready()` function in `run_servers.sh` also detects server crashes early (checks PID liveness while polling), preventing silent failures from blocking the benchmark indefinitely.
