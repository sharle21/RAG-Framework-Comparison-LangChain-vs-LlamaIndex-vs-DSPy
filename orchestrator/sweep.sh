#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# KV Cache Utilization Sweep
#
# Runs the Go orchestrator at different concurrency levels to measure how
# throughput and latency change as we increase parallel requests to vLLM.
#
# The key insight: vLLM's KV cache is shared across concurrent requests.
# As concurrency increases, each request gets less KV cache → more preemption
# → higher latency. The sweep finds the throughput/latency sweet spot.
#
# Output: results/sweep_results.json with per-level stats
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_DIR/results"
VLLM_URL="http://localhost:8000/v1"

# Concurrency levels to test
WORKERS=(1 4 8 16 32)
RPS_VALUES=(5 10 20 40 80)

# Build the Go orchestrator
echo "=== Building Go orchestrator ==="
cd "$SCRIPT_DIR"
go build -o rag-orchestrator .
echo "Built successfully"

# Generate queries if not already done
if [ ! -f "$SCRIPT_DIR/queries.json" ]; then
    echo "=== Generating queries ==="
    python3 "$SCRIPT_DIR/generate_queries.py"
fi

mkdir -p "$RESULTS_DIR"

# Initialize sweep results JSON
echo '{"sweep_results": []}' > "$RESULTS_DIR/sweep_results.json"

echo ""
echo "=== Starting KV Cache Sweep ==="
echo "Levels: ${WORKERS[*]} workers"
echo "RPS:    ${RPS_VALUES[*]}"
echo ""

for i in "${!WORKERS[@]}"; do
    W="${WORKERS[$i]}"
    R="${RPS_VALUES[$i]}"
    OUTPUT="$RESULTS_DIR/sweep_w${W}_rps${R}.json"

    echo "── Level $((i+1))/${#WORKERS[@]}: workers=$W rps=$R ──"

    # Run orchestrator
    START_TIME=$(date +%s%N)
    "$SCRIPT_DIR/rag-orchestrator" \
        --input "$SCRIPT_DIR/queries.json" \
        --output "$OUTPUT" \
        --workers "$W" \
        --rps "$R" \
        --vllm-url "$VLLM_URL" \
        --model "meta-llama/Meta-Llama-3-8B-Instruct" \
        2>&1 | tail -5
    END_TIME=$(date +%s%N)

    # Calculate wall clock time in seconds
    WALL_MS=$(( (END_TIME - START_TIME) / 1000000 ))

    # Extract stats from output file
    if [ -f "$OUTPUT" ]; then
        N_RESULTS=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(len(d.get('results', d if isinstance(d, list) else [])))" 2>/dev/null || echo "0")
        echo "  Completed: $N_RESULTS queries in ${WALL_MS}ms"
        echo "  Throughput: $(python3 -c "print(round($N_RESULTS / ($WALL_MS / 1000), 1) if $WALL_MS > 0 else 0)" 2>/dev/null) queries/sec"
    else
        N_RESULTS=0
        echo "  WARNING: No output file"
    fi

    # Append to sweep results
    python3 -c "
import json
with open('$RESULTS_DIR/sweep_results.json') as f:
    data = json.load(f)
data['sweep_results'].append({
    'workers': $W,
    'rps': $R,
    'wall_ms': $WALL_MS,
    'n_queries': $N_RESULTS,
    'throughput_qps': round($N_RESULTS / ($WALL_MS / 1000), 2) if $WALL_MS > 0 else 0,
})
with open('$RESULTS_DIR/sweep_results.json', 'w') as f:
    json.dump(data, f, indent=2)
"
    echo ""
done

# Print summary table
echo "=== SWEEP SUMMARY ==="
python3 -c "
import json
with open('$RESULTS_DIR/sweep_results.json') as f:
    data = json.load(f)
print(f\"{'Workers':<10} {'RPS':<8} {'Queries':<10} {'Wall(s)':<10} {'Throughput':<12}\")
print('-' * 50)
for r in data['sweep_results']:
    print(f\"{r['workers']:<10} {r['rps']:<8} {r['n_queries']:<10} {r['wall_ms']/1000:.1f}s{'':>4} {r['throughput_qps']:.1f} q/s\")
"

echo ""
echo "Sweep results saved to $RESULTS_DIR/sweep_results.json"
