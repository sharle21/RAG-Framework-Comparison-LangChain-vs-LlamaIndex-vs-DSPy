"""
Run DSPy baseline vs MIPROv2-optimized comparison.

Uses the first 20 DSPy results as training examples for MIPROv2,
then benchmarks both baseline and optimized on the remaining queries.

Usage:
    python run_dspy_optimized.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/rag-bench")
from src.dspy_rag.pipeline import DSPyRAG
from src.evaluation.metrics import evaluate_string_overlap, evaluate_llm_judge, make_vllm_judge

RESULTS_FILE = "/home/ubuntu/rag-bench/results/go_results_20260408_013644.json"
DATA_FILE = "/home/ubuntu/rag-bench/data/raw/ragbench_documents.json"
BASE_URL = "http://localhost:8000/v1"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_URL = "http://localhost:8001/v1"
JUDGE_MODEL = "Qwen/Qwen3-14B"

# ── Load training + test data ─────────────────────────────────────────────────
print("Loading data...")
all_results = json.load(open(RESULTS_FILE))
dspy_results = [r for r in all_results if r["framework"] == "dspy" and r.get("answer") and not r.get("error")]

# Use first 20 as train, rest as test
train_qa = [{"question": r["question"], "ground_truth": r["ground_truth"]} for r in dspy_results[:20]]
test_queries = [{"question": r["question"], "ground_truth": r["ground_truth"], "domain": r["domain"]} for r in dspy_results[20:]]
print(f"  Train: {len(train_qa)} examples  |  Test: {len(test_queries)} queries")

# ── Build DSPy pipeline (loads index from disk) ───────────────────────────────
print("\nBuilding DSPy pipeline (loading index from disk)...")
documents = json.load(open(DATA_FILE))
rag = DSPyRAG(model=MODEL, base_url=BASE_URL, local_embeddings=True)
rag.build(documents)
print("  Index loaded.")

# ── Baseline run ──────────────────────────────────────────────────────────────
print(f"\nRunning BASELINE on {len(test_queries)} queries...")
baseline_results = []
for i, q in enumerate(test_queries):
    try:
        result = rag.query(q["question"])
        baseline_results.append({
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "domain": q["domain"],
            **result,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_queries)}")
    except Exception as e:
        print(f"  Error on query {i}: {e}")

# ── MIPROv2 optimization ──────────────────────────────────────────────────────
print(f"\nRunning MIPROv2 optimization on {len(train_qa)} examples...")
rag.optimize(train_qa, n_train=20)
print("  Optimization complete.")

# ── Optimized run ─────────────────────────────────────────────────────────────
print(f"\nRunning OPTIMIZED on {len(test_queries)} queries...")
optimized_results = []
for i, q in enumerate(test_queries):
    try:
        result = rag.query(q["question"])
        optimized_results.append({
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "domain": q["domain"],
            **result,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_queries)}")
    except Exception as e:
        print(f"  Error on query {i}: {e}")

# ── Evaluate both ─────────────────────────────────────────────────────────────
print("\nEvaluating with string overlap...")
baseline_str = evaluate_string_overlap(baseline_results)
optimized_str = evaluate_string_overlap(optimized_results)

print("Evaluating with Qwen judge...")
judge = make_vllm_judge(JUDGE_URL, JUDGE_MODEL)
baseline_judge = evaluate_llm_judge(baseline_results, n_runs=1, judge=judge)
optimized_judge = evaluate_llm_judge(optimized_results, n_runs=1, judge=judge)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DSPy Baseline vs Optimized (MIPROv2)")
print("=" * 60)
print(f"{'Metric':25} {'Baseline':>10} {'Optimized':>10} {'Delta':>8}")
print("-" * 60)
for metric, b_val, o_val in [
    ("Token F1",        baseline_str["answer_f1"],           optimized_str["answer_f1"]),
    ("Context Coverage",baseline_str["context_coverage"],    optimized_str["context_coverage"]),
    ("Correctness",     baseline_judge["correctness"],        optimized_judge["correctness"]),
    ("Faithfulness",    baseline_judge["faithfulness"],       optimized_judge["faithfulness"]),
    ("Completeness",    baseline_judge["completeness"],       optimized_judge["completeness"]),
]:
    delta = o_val - b_val
    sign = "+" if delta >= 0 else ""
    print(f"{metric:25} {b_val:10.3f} {o_val:10.3f} {sign}{delta:7.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "baseline": {**baseline_str, **baseline_judge},
    "optimized": {**optimized_str, **optimized_judge},
    "baseline_results": baseline_results,
    "optimized_results": optimized_results,
}
out_path = "/home/ubuntu/rag-bench/results/dspy_optimization_comparison.json"
json.dump(output, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")
