#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline: wait for benchmark → run KV cache sweep → push results
#
# Run this and go to sleep. Everything GPU-dependent finishes automatically.
# Check GitHub when you wake up.
# ─────────────────────────────────────────────────────────────────────────────
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/tmp/pipeline.log"

echo "=== Full Pipeline Started ===" | tee "$LOG"
echo "$(date): Starting benchmark..." | tee -a "$LOG"

# Step 1: Run the server-based benchmark (starts Python RAG servers + Go orchestrator)
bash "$REPO_DIR/orchestrator/run_servers.sh" 2>&1 | tee -a "$LOG"
echo "$(date): Benchmark finished!" | tee -a "$LOG"

# Step 2: Run KV cache sweep
echo "$(date): Starting KV cache sweep..." | tee -a "$LOG"
cd "$REPO_DIR/orchestrator"
bash sweep.sh 2>&1 | tee -a "$LOG"
echo "$(date): Sweep finished!" | tee -a "$LOG"

# Step 3: Push results to GitHub
echo "$(date): Pushing results to GitHub..." | tee -a "$LOG"
cd "$REPO_DIR"
git add -f results/ || true
git add orchestrator/queries.json || true
git commit -m "add benchmark results and KV cache sweep from H100" || echo "Nothing to commit"
git push origin main 2>&1 | tee -a "$LOG"

echo "$(date): All done! Check GitHub for results." | tee -a "$LOG"
