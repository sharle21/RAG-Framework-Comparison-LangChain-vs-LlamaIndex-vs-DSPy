"""
run_statistical_tests.py

Mann-Whitney U + permutation tests on existing scores.
Answers: "does Framework A actually beat Framework B, or is it noise?"

Run: python run_statistical_tests.py
"""

import json
import random
import sys

random.seed(42)
sys.path.insert(0, ".")

from scipy import stats as scipy_stats
from src.evaluation.metrics import f1_score

RESULTS_FILE = "results/go_results_20260408_013644.json"
EVAL_SCORES_FILE = "results/eval_scores.json"
frameworks = ["langchain", "llamaindex", "dspy"]
FW_PAIRS = [("langchain", "llamaindex"), ("langchain", "dspy"), ("llamaindex", "dspy")]


def permutation_test(a: list[float], b: list[float], n_perm: int = 10000) -> float:
    """Two-sided permutation test. Returns p-value."""
    observed = abs(sum(a) / len(a) - sum(b) / len(b))
    combined = a + b
    na = len(a)
    count = 0
    for _ in range(n_perm):
        random.shuffle(combined)
        diff = abs(sum(combined[:na]) / na - sum(combined[na:]) / len(b))
        if diff >= observed:
            count += 1
    return count / n_perm


def effect_size_rb(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation — effect size for Mann-Whitney."""
    return 1 - (2 * u_stat) / (n1 * n2)


def interpret_p(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def run_tests(scores: dict[str, list[float]], metric_name: str) -> list[dict]:
    results = []
    for fw_a, fw_b in FW_PAIRS:
        a, b = scores[fw_a], scores[fw_b]
        u, p_mw = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        rb = effect_size_rb(u, len(a), len(b))
        p_perm = permutation_test(a, b)
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        results.append({
            "metric": metric_name,
            "fw_a": fw_a, "fw_b": fw_b,
            "mean_a": round(mean_a, 4), "mean_b": round(mean_b, 4),
            "delta": round(mean_a - mean_b, 4),
            "mann_whitney_p": round(p_mw, 4),
            "permutation_p": round(p_perm, 4),
            "effect_size_rb": round(rb, 4),
            "significant": bool(p_mw < 0.05 and p_perm < 0.05),
        })
    return results


def print_test_table(results: list[dict], metric_name: str):
    print(f"\n  {metric_name}")
    print(f"  {'Comparison':<28} {'Delta':>7}  {'MW p':>7}  {'Perm p':>7}  {'Effect':>7}  {'Sig':>4}")
    print("  " + "-" * 62)
    for r in results:
        comp = f"{r['fw_a']} vs {r['fw_b']}"
        sig = interpret_p(min(r["mann_whitney_p"], r["permutation_p"]))
        print(f"  {comp:<28} {r['delta']:>7.4f}  {r['mann_whitney_p']:>7.4f}  "
              f"{r['permutation_p']:>7.4f}  {r['effect_size_rb']:>7.4f}  {sig:>4}")


def main():
    d = json.load(open(RESULTS_FILE))

    # Token F1 per framework
    f1_scores = {fw: [] for fw in frameworks}
    for r in d:
        fw = r.get("framework")
        if fw in f1_scores and r.get("answer"):
            f1_scores[fw].append(f1_score(r["answer"], r["ground_truth"]))

    # Judge scores per framework from eval_scores.json
    judge_scores = {fw: {dim: [] for dim in ["correctness", "faithfulness", "completeness"]}
                    for fw in frameworks}
    try:
        eval_data = json.load(open(EVAL_SCORES_FILE))
        for fw in frameworks:
            for q in eval_data.get(fw, {}).get("per_question", []):
                for dim in ["correctness", "faithfulness", "completeness"]:
                    if dim in q:
                        judge_scores[fw][dim].append(q[dim])
    except FileNotFoundError:
        print("eval_scores.json not found — skipping judge tests")

    all_results = []

    print("\n" + "=" * 68)
    print("  STATISTICAL TESTS  (MW = Mann-Whitney U, Perm = permutation)")
    print("  * p<0.05   ** p<0.01   *** p<0.001   ns = not significant")
    print("=" * 68)

    f1_tests = run_tests(f1_scores, "Token F1")
    print_test_table(f1_tests, "TOKEN F1")
    all_results.extend(f1_tests)

    for dim in ["correctness", "faithfulness", "completeness"]:
        dim_scores = {fw: judge_scores[fw][dim] for fw in frameworks}
        if all(len(v) > 0 for v in dim_scores.values()):
            tests = run_tests(dim_scores, f"Judge {dim}")
            print_test_table(tests, f"JUDGE {dim.upper()}")
            all_results.extend(tests)

    # Summary: what's actually significant
    print("\n" + "=" * 68)
    print("  SUMMARY — significant differences only (both tests p<0.05)")
    print("=" * 68)
    sig = [r for r in all_results if r["significant"]]
    if sig:
        for r in sig:
            winner = r["fw_a"] if r["delta"] > 0 else r["fw_b"]
            loser = r["fw_b"] if r["delta"] > 0 else r["fw_a"]
            print(f"  {r['metric']:20}  {winner} > {loser}  "
                  f"(delta={abs(r['delta']):.3f}, effect={abs(r['effect_size_rb']):.3f})")
    else:
        print("  None")

    json.dump(all_results, open("results/statistical_tests.json", "w"), indent=2)
    print(f"\nSaved → results/statistical_tests.json")


if __name__ == "__main__":
    main()
