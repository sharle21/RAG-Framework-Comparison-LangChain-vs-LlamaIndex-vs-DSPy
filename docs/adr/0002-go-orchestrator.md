# ADR-0002: Go Orchestrator for Concurrent Query Dispatch

**Status:** Accepted

## Context

The benchmark needs to fire queries to three RAG servers simultaneously and collect timing metrics with high precision. Two options considered:

**Python async (asyncio / aiohttp)**
- Natural choice given the rest of the stack is Python
- `asyncio` provides cooperative concurrency but runs on one OS thread
- Python's GIL prevents true parallel execution of CPU-bound work
- For I/O-bound HTTP requests, asyncio works — but timing precision degrades when the event loop is shared with metric collection and JSON serialization

**Go (net/http + goroutines)**
- Goroutines are multiplexed across OS threads by the Go runtime scheduler
- At RPS=5 with 3 servers, 15 requests are in-flight at peak — goroutines handle this with ~2KB stack overhead each vs ~1MB for OS threads
- `time.Since()` gives nanosecond-precision wall-clock timing per request
- Prometheus client library (`prometheus/client_golang`) integrates natively
- Binary ships as a single static executable; no interpreter or venv required on Lambda

## Decision

Use Go for the orchestrator.

The primary driver is timing precision and true concurrency under load. When 15 requests are in-flight simultaneously, a Python event loop introduces scheduling jitter that inflates latency measurements. Go's goroutines dispatch independently and time each request in isolation.

Secondary driver: Prometheus integration. The Go Prometheus client exposes `/metrics` directly from the orchestrator process, enabling Grafana dashboards without a separate sidecar.

## Consequences

- Requires Go installed on Lambda (`/usr/local/go/bin` must be in PATH).
- `orchestrator/main.go` must be compiled before each benchmark run (`go build`).
- Two-language codebase adds a build step, but the Go binary is self-contained and doesn't interact with the Python venv.
- RAG server implementations remain in Python — only the dispatch layer is Go.
