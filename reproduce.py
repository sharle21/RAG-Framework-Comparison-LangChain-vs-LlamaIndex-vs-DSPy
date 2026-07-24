"""
reproduce.py

Single entrypoint to regenerate every locally-reproducible number in this
project from the committed results JSON — no GPU, no vLLM, no LLM API calls.
Runs the existing standalone scripts in dependency order and reports which
steps succeeded.

What this does NOT reproduce (needs the original GPU instance):
  - the LLM judge pass itself (results/eval_unified.json) — Qwen3-14B judging
    Llama-3.1-8B answers, run on Lambda
  - the raw benchmark run (results/go_results_*.json) — vLLM serving all three
    RAG pipelines

What this DOES reproduce, from those two files plus the corpus:
  - ROUGE-1/2/L (run_rouge.py)
  - BERTScore (run_bertscore.py — downloads a small local model on first run,
    still no GPU)
  - Bootstrap CIs for Token F1 + judge scores + latency percentiles
    (compute_stats_local.py)
  - Cross-metric ranking + Spearman correlation (run_metric_comparison.py,
    depends on run_rouge.py's output)
  - Mann-Whitney U + permutation significance tests (run_statistical_tests.py)
  - Retrieval-vs-relevant-document overlap rate (run_retrieval_overlap.py)

Run: python reproduce.py
"""

import subprocess
import sys

STEPS = [
    ("ROUGE-1/2/L", ["python3", "run_rouge.py"]),
    ("BERTScore", ["python3", "run_bertscore.py"]),
    ("Bootstrap CIs + latency percentiles", ["python3", "compute_stats_local.py"]),
    ("Cross-metric ranking + Spearman correlation", ["python3", "run_metric_comparison.py"]),
    ("Mann-Whitney + permutation significance tests", ["python3", "run_statistical_tests.py"]),
    ("Retrieval overlap rate", ["python3", "run_retrieval_overlap.py", "results/go_results_20260408_013644.json"]),
]


def main():
    results = []
    for name, cmd in STEPS:
        print(f"\n{'#' * 70}")
        print(f"# {name}")
        print(f"# $ {' '.join(cmd)}")
        print(f"{'#' * 70}\n")
        proc = subprocess.run(cmd)
        results.append((name, proc.returncode == 0))

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for name, ok in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")

    if not all(ok for _, ok in results):
        print("\nOne or more steps failed — see output above.")
        sys.exit(1)

    print("\nAll steps completed. Compare output against README tables —")
    print("small (<0.01) drift is expected from metrics.py revisions since the")
    print("original run; larger drift means something upstream actually changed.")


if __name__ == "__main__":
    main()
