"""
run_metric_comparison.py

Show where evaluation metrics agree and disagree.
Loads Token F1, ROUGE, BERTScore, LLM judge scores for all frameworks.
Computes per-metric rankings + Spearman correlation between metrics.

Run: python run_metric_comparison.py (run run_rouge.py first)
"""

import json
import random
import sys

random.seed(42)
sys.path.insert(0, ".")

from scipy.stats import spearmanr
from src.evaluation.metrics import f1_score

RESULTS_FILE = "results/go_results_20260408_013644.json"
ROUGE_FILE = "results/rouge_results.json"
BERTSCORE_FILE = "results/bertscore_results.json"
EVAL_SCORES_FILE = "results/eval_scores.json"

frameworks = ["langchain", "llamaindex", "dspy"]


def rank(scores: dict[str, float]) -> dict[str, int]:
    """1 = best."""
    ordered = sorted(scores, key=lambda fw: scores[fw], reverse=True)
    return {fw: ordered.index(fw) + 1 for fw in frameworks}


def main():
    d = json.load(open(RESULTS_FILE))

    # Token F1
    f1_per_fw: dict[str, list[float]] = {fw: [] for fw in frameworks}
    for r in d:
        fw = r.get("framework")
        if fw in f1_per_fw and r.get("answer"):
            f1_per_fw[fw].append(f1_score(r["answer"], r["ground_truth"]))
    token_f1 = {fw: sum(v) / len(v) for fw, v in f1_per_fw.items()}

    # ROUGE
    rouge_data = json.load(open(ROUGE_FILE))
    rouge1 = {fw: rouge_data[fw]["overall"]["rouge1_f"]["mean"] for fw in frameworks}
    rougeL = {fw: rouge_data[fw]["overall"]["rougeL_f"]["mean"] for fw in frameworks}

    # BERTScore
    bts = json.load(open(BERTSCORE_FILE))
    bertscore = {fw: bts[fw]["overall"]["bertscore_f1"] for fw in frameworks}

    # Judge scores
    eval_data = json.load(open(EVAL_SCORES_FILE))
    correctness = {fw: eval_data[fw]["correctness"] for fw in frameworks}
    faithfulness = {fw: eval_data[fw]["faithfulness"] for fw in frameworks}
    completeness = {fw: eval_data[fw]["completeness"] for fw in frameworks}

    all_metrics = {
        "Token F1": token_f1,
        "ROUGE-1": rouge1,
        "ROUGE-L": rougeL,
        "BERTScore": bertscore,
        "Correctness": correctness,
        "Faithfulness": faithfulness,
        "Completeness": completeness,
    }

    # Rankings table
    print("\n" + "=" * 72)
    print("  METRIC RANKINGS (1=best, 3=worst)")
    print("=" * 72)
    print(f"  {'Metric':<16}" + "".join(f"  {fw:>12}" for fw in frameworks) + "  Winner")
    print("  " + "-" * 62)
    rankings_by_metric: dict[str, dict] = {}
    for name, scores in all_metrics.items():
        r = rank(scores)
        rankings_by_metric[name] = r
        winner = min(r, key=r.get)
        print(f"  {name:<16}" + "".join(f"  {scores[fw]:>8.3f}({r[fw]})" for fw in frameworks) + f"  {winner}")

    # Where rankings flip
    print("\n" + "=" * 72)
    print("  RANKING DISAGREEMENTS — metrics that contradict each other")
    print("=" * 72)
    string_metrics = ["Token F1", "ROUGE-1", "ROUGE-L", "BERTScore"]
    judge_metrics = ["Correctness", "Faithfulness", "Completeness"]

    for fw in frameworks:
        string_ranks = [rankings_by_metric[m][fw] for m in string_metrics]
        judge_ranks = [rankings_by_metric[m][fw] for m in judge_metrics]
        avg_string = sum(string_ranks) / len(string_ranks)
        avg_judge = sum(judge_ranks) / len(judge_ranks)
        direction = "string metrics OVERRATE" if avg_string < avg_judge else "judge metrics OVERRATE"
        print(f"  {fw:<12}  string avg rank={avg_string:.1f}  judge avg rank={avg_judge:.1f}  → {direction}")

    # Spearman correlations between metrics (using per-framework means as vectors)
    print("\n" + "=" * 72)
    print("  SPEARMAN CORRELATION BETWEEN METRICS (across 3 framework means)")
    print("  NOTE: n=3 — treat as directional only, not statistically meaningful")
    print("=" * 72)
    metric_names = list(all_metrics.keys())
    metric_vectors = {name: [all_metrics[name][fw] for fw in frameworks] for name in metric_names}

    print(f"  {'':16}" + "".join(f"  {n[:8]:>8}" for n in metric_names))
    print("  " + "-" * (16 + 10 * len(metric_names)))
    for name_a in metric_names:
        row = f"  {name_a:<16}"
        for name_b in metric_names:
            if name_a == name_b:
                row += "     1.00"
            else:
                rho, _ = spearmanr(metric_vectors[name_a], metric_vectors[name_b])
                row += f"  {rho:>8.2f}"
        print(row)

    # Save
    out = {
        "scores": {name: all_metrics[name] for name in all_metrics},
        "rankings": rankings_by_metric,
    }
    json.dump(out, open("results/metric_comparison.json", "w"), indent=2)
    print(f"\nSaved → results/metric_comparison.json")


if __name__ == "__main__":
    main()
