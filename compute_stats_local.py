"""
compute_stats_local.py

Compute bootstrap CIs from existing results — no Lambda, no LLM calls.
Loads go_results + eval_scores + bertscore_results and outputs a stats table.

Run: python compute_stats_local.py
"""

import json
import sys

sys.path.insert(0, ".")
from src.evaluation.metrics import f1_score, compute_stats

RESULTS_FILE = "results/go_results_20260408_013644.json"
EVAL_SCORES_FILE = "results/eval_unified.json"  # NOT eval_scores.json — that's the stale
# pre-fix file from the double-judge-call bug (see README Bugs Found). eval_unified.json
# is what the published README numbers actually come from.
BERTSCORE_FILE = "results/bertscore_results.json"
DOMAIN_SCORES_FILE = "results/eval_scores_by_domain.json"

frameworks = ["langchain", "llamaindex", "dspy"]
domains = ["covidqa", "techqa", "finqa"]


def load_per_query_f1(results_file: str) -> dict:
    """Compute per-query F1 from raw results. Returns {fw: [f1, ...]}."""
    d = json.load(open(results_file))
    per_fw = {fw: [] for fw in frameworks}
    for r in d:
        fw = r.get("framework")
        if fw not in per_fw or not r.get("answer"):
            continue
        per_fw[fw].append(f1_score(r["answer"], r["ground_truth"]))
    return per_fw


def load_per_query_latency(results_file: str) -> dict:
    """Returns {fw: {"retrieval": [...], "generation": [...]}}."""
    d = json.load(open(results_file))
    per_fw = {fw: {"retrieval": [], "generation": []} for fw in frameworks}
    for r in d:
        fw = r.get("framework")
        if fw not in per_fw:
            continue
        if r.get("retrieval_ms"):
            per_fw[fw]["retrieval"].append(r["retrieval_ms"])
        if r.get("generation_ms"):
            per_fw[fw]["generation"].append(r["generation_ms"])
    return per_fw


def pct(data: list[float], p: float) -> float:
    s = sorted(data)
    return s[min(int(p / 100 * len(s)), len(s) - 1)]


def print_latency_table(header: str, per_fw: dict, key: str):
    import statistics
    print(f"\n{'='*72}")
    print(f"  {header}")
    print(f"  NOTE: mean misleading due to outliers — use median for comparison")
    print(f"{'='*72}")
    print(f"  {'Framework':<12} {'Median':>8}  {'p95':>8}  {'p99':>8}  {'Mean':>8}  {'Std':>8}  {'n':>4}")
    print(f"  {'-'*64}")
    for fw in frameworks:
        vals = per_fw[fw][key]
        if not vals:
            continue
        print(f"  {fw:<12} {statistics.median(vals):8.1f}  "
              f"{pct(vals,95):8.1f}  {pct(vals,99):8.1f}  "
              f"{statistics.mean(vals):8.1f}  {statistics.stdev(vals):8.1f}  {len(vals):4}")


def print_metric_table(header: str, rows: list[tuple]):
    print(f"\n{'='*70}")
    print(f"  {header}")
    print(f"{'='*70}")
    print(f"  {'Framework':<12} {'Mean':>6}  {'Std':>6}  {'95% CI (analytical)':>22}  {'95% CI (bootstrap)':>22}  {'n':>4}")
    print(f"  {'-'*78}")
    for fw, stats in rows:
        ci_a = f"[{stats['ci95_lower']:.3f}, {stats['ci95_upper']:.3f}]"
        ci_b = f"[{stats['boot_ci95_lower']:.3f}, {stats['boot_ci95_upper']:.3f}]"
        print(f"  {fw:<12} {stats['mean']:6.3f}  {stats['std']:6.3f}  {ci_a:>22}  {ci_b:>22}  {stats['n']:4}")


def main():
    print("Loading results...")
    per_fw_f1 = load_per_query_f1(RESULTS_FILE)
    per_fw_lat = load_per_query_latency(RESULTS_FILE)

    # Token F1
    print_metric_table(
        "TOKEN F1  (bootstrap 95% CI, seed=42)",
        [(fw, compute_stats(per_fw_f1[fw])) for fw in frameworks],
    )

    # Latency — proper percentile table
    print_latency_table("RETRIEVAL LATENCY ms  (median is correct metric)", per_fw_lat, "retrieval")
    print_latency_table("GENERATION LATENCY ms  (median is correct metric, ms/token pending Lambda rerun)", per_fw_lat, "generation")

    # LLM judge scores from saved eval_unified.json (matches published README numbers)
    try:
        eval_unified = json.load(open(EVAL_SCORES_FILE))
        eval_scores = eval_unified.get("overall", eval_unified)  # tolerate flat legacy shape
        dims = ["correctness", "faithfulness", "completeness"]
        for dim in dims:
            rows = []
            for fw in frameworks:
                s = eval_scores.get(fw, {})
                mean = s.get(dim, 0.0)
                # Per-question scores available for bootstrap
                pq = s.get("per_question", [])
                vals = [q[dim] for q in pq if dim in q]
                if vals:
                    rows.append((fw, compute_stats(vals)))
                else:
                    rows.append((fw, {"mean": mean, "std": 0.0, "n": 0,
                                      "ci95_lower": 0.0, "ci95_upper": 0.0,
                                      "boot_ci95_lower": 0.0, "boot_ci95_upper": 0.0}))
            print_metric_table(f"JUDGE {dim.upper()} (from saved eval_unified.json)", rows)
    except FileNotFoundError:
        print(f"\n  {EVAL_SCORES_FILE} not found — skipping judge CI table")

    # BERTScore from saved bertscore_results.json
    try:
        bts = json.load(open(BERTSCORE_FILE))
        print(f"\n{'='*70}")
        print("  BERTSCORE F1 (point estimates — no per-query scores saved)")
        print(f"  {'Framework':<12} {'Overall':>8} {'covidqa':>8} {'techqa':>8} {'finqa':>8}")
        print(f"  {'-'*50}")
        for fw in frameworks:
            row = [bts.get(fw, {}).get(k, {}).get("bertscore_f1", 0)
                   for k in ["overall", "covidqa", "techqa", "finqa"]]
            print(f"  {fw:<12} {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}")
        print("  Note: run run_bertscore.py with per-query output to get BERTScore CIs")
    except FileNotFoundError:
        print("\n  bertscore_results.json not found — skipping")

    # Save stats to JSON
    out = {}
    for fw in frameworks:
        out[fw] = {
            "token_f1": compute_stats(per_fw_f1[fw]),
            "retrieval_latency_ms": compute_stats(per_fw_lat[fw]["retrieval"]),
            "generation_latency_ms": compute_stats(per_fw_lat[fw]["generation"]),
        }
    with open("results/stats_with_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full stats → results/stats_with_ci.json")


if __name__ == "__main__":
    main()
