"""
run_eval_unified.py — single-pass eval: judge once, aggregate globally + by domain.

Replaces run_eval.py + run_eval_domains.py.
n_runs=1 across all 450 responses. Domain stats computed from the same
per-question rows — no second judge-call pass.

Run on Lambda after benchmark:
    python -u run_eval_unified.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

random.seed(42)

sys.path.insert(0, "/home/ubuntu/rag-bench")
from src.evaluation.metrics import (
    evaluate_string_overlap,
    evaluate_llm_judge,
    make_vllm_judge,
    compute_stats,
)

RESULTS_FILE = Path("/home/ubuntu/rag-bench/results/go_results_20260718_005025.json")
OUT_FILE = Path("/home/ubuntu/rag-bench/results/eval_unified.json")
FRAMEWORKS = ["langchain", "llamaindex", "dspy"]
DOMAINS = ["covidqa", "techqa", "finqa"]


def domain_stats_from_per_question(
    per_question: list[dict], domain_map: dict
) -> dict[str, dict]:
    dims = ["correctness", "faithfulness", "completeness"]
    by_domain: dict[str, dict[str, list]] = defaultdict(lambda: {d: [] for d in dims})

    for q in per_question:
        domain = domain_map.get(q["question"], "unknown")
        for dim in dims:
            val = q.get(dim)
            if val is not None and val > 0:
                by_domain[domain][dim].append(float(val))

    out: dict[str, dict] = {}
    for domain, dim_scores in by_domain.items():
        out[domain] = {}
        for dim in dims:
            s = compute_stats(dim_scores[dim])
            out[domain][dim] = s["mean"]
            out[domain][f"{dim}_boot_ci95"] = [s["boot_ci95_lower"], s["boot_ci95_upper"]]
            out[domain][f"{dim}_n"] = s["n"]
    return out


def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    judge = make_vllm_judge("http://localhost:8001/v1", "Qwen/Qwen3-14B")
    domain_map = {r["question"]: r.get("domain", "unknown") for r in data}

    result = {"overall": {}, "by_domain": {}}

    for fw in FRAMEWORKS:
        rows = [
            r for r in data
            if r["framework"] == fw and r.get("answer") and not r.get("error")
        ]
        print(f"\n=== {fw} ({len(rows)} valid results) ===")

        overlap = evaluate_string_overlap(rows)
        print(f"  F1={overlap['answer_f1']:.3f}  ctx_coverage={overlap['context_coverage']:.3f}")

        print("  Running LLM judge (n_runs=1)...")
        judge_scores = evaluate_llm_judge(rows, n_runs=1, judge=judge)
        print(
            f"  correctness={judge_scores['correctness']:.3f}"
            f"  faithfulness={judge_scores['faithfulness']:.3f}"
            f"  completeness={judge_scores['completeness']:.3f}"
        )

        result["overall"][fw] = {**overlap, **judge_scores}

        per_q = judge_scores.get("per_question", [])
        dom_judge = domain_stats_from_per_question(per_q, domain_map)

        # Add string-overlap F1 per domain (no extra judge calls)
        for domain in DOMAINS:
            dom_rows = [r for r in rows if r.get("domain") == domain]
            if not dom_rows:
                continue
            dom_overlap = evaluate_string_overlap(dom_rows)
            if domain not in dom_judge:
                dom_judge[domain] = {}
            dom_judge[domain]["answer_f1"] = dom_overlap["answer_f1"]
            dom_judge[domain]["answer_f1_boot_ci95"] = dom_overlap["answer_f1_boot_ci95"]
            dom_judge[domain]["n"] = len(dom_rows)

        result["by_domain"][fw] = dom_judge

    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")

    print("\n\n=== OVERALL ===")
    print(f"{'':12} {'F1':>6} {'correct':>9} {'faithful':>9} {'complete':>9}")
    for fw in FRAMEWORKS:
        s = result["overall"][fw]
        print(
            f"  {fw:10} {s['answer_f1']:6.3f}"
            f" {s['correctness']:9.3f}"
            f" {s['faithfulness']:9.3f}"
            f" {s['completeness']:9.3f}"
        )

    print("\n=== BY DOMAIN ===")
    print(f"{'':22} {'F1':>6} {'correct':>9} {'faithful':>9} {'complete':>9}")
    for fw in FRAMEWORKS:
        print(f"\n{fw.upper()}")
        for domain in DOMAINS:
            s = result["by_domain"][fw].get(domain, {})
            print(
                f"  {domain:20} {s.get('answer_f1', 0):6.3f}"
                f" {s.get('correctness', 0):9.3f}"
                f" {s.get('faithfulness', 0):9.3f}"
                f" {s.get('completeness', 0):9.3f}"
            )


if __name__ == "__main__":
    main()
