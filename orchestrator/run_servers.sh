#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Start all three RAG servers, then run the Go orchestrator.
#
# Each Python server builds its index once on startup (takes ~60s for bge-m3
# embeddings). Go waits at the health check until all three are ready, then
# fires queries concurrently.
#
# Usage:
#   bash orchestrator/run_servers.sh
#
# Optional env overrides:
#   VLLM_URL=http://localhost:8000/v1   (default shown)
#   WORKERS=16                          (Go goroutines, default 16)
#   RPS=20                              (requests/sec, default 20)
#   PER_DOMAIN=50                       (QA pairs per domain, default 50)
# ─────────────────────────────────────────────────────────────────────────────
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
WORKERS="${WORKERS:-16}"
RPS="${RPS:-20}"
PER_DOMAIN="${PER_DOMAIN:-50}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TRACING="${TRACING:-0}"
PHOENIX_ENDPOINT="${PHOENIX_ENDPOINT:-http://localhost:4317}"

TRACING_FLAG=""
if [ "$TRACING" = "1" ]; then
    TRACING_FLAG="--tracing --phoenix-endpoint $PHOENIX_ENDPOINT"
fi

echo "=================================================="
echo "  RAG Server Stack"
echo "=================================================="
echo "  vLLM URL:    $VLLM_URL"
echo "  Model:       $MODEL"
echo "  Workers:     $WORKERS goroutines"
echo "  RPS:         $RPS"
echo "  Per domain:  $PER_DOMAIN QA pairs"
echo "  Tracing:     ${TRACING:-off} (set TRACING=1 to enable Phoenix)"
echo "=================================================="

# ── 1. Generate queries.json ──────────────────────────────────────────────────
echo ""
echo "[1/4] Generating queries.json..."
python orchestrator/generate_queries.py --per-domain "$PER_DOMAIN"

# ── 2. Start the three Python RAG servers one at a time ───────────────────────
# Each server embeds the full corpus with bge-m3 on CPU (~16GB RAM each).
# Starting them in parallel causes OOM — start sequentially and wait for each
# to report ready before starting the next.
echo ""
echo "[2/4] Starting RAG servers (sequential index build to avoid OOM)..."

COMMON_ARGS="--base-url $VLLM_URL --model $MODEL --local-embeddings $TRACING_FLAG"

wait_ready() {
    local name=$1 url=$2 logfile=$3
    echo -n "  [$name] building index..."
    until curl -s "$url/health" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') else 1)" 2>/dev/null; do
        # If the process died, print its log and abort
        if ! kill -0 $4 2>/dev/null; then
            echo ""
            echo "  [$name] CRASHED — log:"
            cat "$logfile"
            exit 1
        fi
        echo -n "."
        sleep 5
    done
    echo " ready!"
}

# PYTHONUNBUFFERED=1 forces Python to flush stdout/stderr immediately —
# without it, log files stay empty until the process exits (OS buffers the output)
PYTHONUNBUFFERED=1 python src/rag_server.py --framework langchain  --port 8100 $COMMON_ARGS > /tmp/langchain_server.log 2>&1 &
LANGCHAIN_PID=$!
wait_ready "langchain" "http://localhost:8100" /tmp/langchain_server.log $LANGCHAIN_PID

PYTHONUNBUFFERED=1 python src/rag_server.py --framework llamaindex --port 8101 $COMMON_ARGS > /tmp/llamaindex_server.log 2>&1 &
LLAMAINDEX_PID=$!
wait_ready "llamaindex" "http://localhost:8101" /tmp/llamaindex_server.log $LLAMAINDEX_PID

PYTHONUNBUFFERED=1 python src/rag_server.py --framework dspy       --port 8102 $COMMON_ARGS > /tmp/dspy_server.log 2>&1 &
DSPY_PID=$!
wait_ready "dspy" "http://localhost:8102" /tmp/dspy_server.log $DSPY_PID

echo "  All servers ready."
echo "  langchain  PID $LANGCHAIN_PID  → port 8100"
echo "  llamaindex PID $LLAMAINDEX_PID → port 8101"
echo "  dspy       PID $DSPY_PID       → port 8102"

# Shut down all servers when this script exits (Ctrl+C or after Go finishes)
cleanup() {
    echo ""
    echo "Shutting down RAG servers..."
    kill "$LANGCHAIN_PID" "$LLAMAINDEX_PID" "$DSPY_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── 3. Build Go orchestrator binary ──────────────────────────────────────────
echo ""
echo "[3/4] Building Go orchestrator..."
cd orchestrator
go build -o rag-orchestrator .
cd "$REPO_DIR"

# ── 4. Run the Go orchestrator ────────────────────────────────────────────────
# Go polls /health on each server and waits until all three are ready.
echo ""
echo "[4/4] Running Go orchestrator (waiting for servers to finish indexing)..."

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT="results/go_results_${TIMESTAMP}.json"
mkdir -p results

orchestrator/rag-orchestrator \
    --input    orchestrator/queries.json \
    --output   "$OUTPUT" \
    --workers  "$WORKERS" \
    --rps      "$RPS" \
    --langchain-url  http://localhost:8100 \
    --llamaindex-url http://localhost:8101 \
    --dspy-url       http://localhost:8102

echo ""
echo "Done. Results at $OUTPUT"
