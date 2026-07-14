"""
run_rouge.py

Compute ROUGE-1, ROUGE-2, ROUGE-L from existing go_results.
No LLM, no GPU — runs locally in seconds.

Run: python run_rouge.py
"""

import json
import random
import sys

random.seed(42)
sys.path.insert(0, ".")

from rouge_score import rouge_scorer
from src.evaluation.metrics import compute_stats

RESULTS_FILE = "results/go_results_20260408_013644.json"
frameworks = ["langchain", "llamaindex", "dspy"]
domains = ["covidqa", "techqa", "finqa"]

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_row(r: dict) -> dict:
    scores = scorer.score(r["ground_truth"], r["answer"])
    return {
        "rouge1_p": scores["rouge1"].precision,
        "rouge1_r": scores["rouge1"].recall,
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def main():
    d = json.load(open(RESULTS_FILE))
    out = {}

    for fw in frameworks:
        out[fw] = {"overall": {}, "by_domain": {}}
        fw_rows = [r for r in d if r["framework"] == fw and r.get("answer")]

        per_metric = {m: [] for m in ["rouge1_f", "rouge2_f", "rougeL_f"]}
        per_query = []
        for r in fw_rows:
            s = score_row(r)
            for m in per_metric:
                per_metric[m].append(s[m])
            per_query.append({**s, "domain": r["domain"]})

        out[fw]["overall"] = {m: compute_stats(v) for m, v in per_metric.items()}
        out[fw]["per_query"] = per_query

        for domain in domains:
            dom_scores = [q for q in per_query if q["domain"] == domain]
            out[fw]["by_domain"][domain] = {
                m: compute_stats([q[m] for q in dom_scores])
                for m in per_metric
            }

    json.dump(out, open("results/rouge_results.json", "w"), indent=2)
    print("Saved → results/rouge_results.json\n")

    # Print summary table
    metrics = ["rouge1_f", "rouge2_f", "rougeL_f"]
    header = f"  {'Framework':<12}" + "".join(f"  {m:>12}" for m in metrics)
    print("=" * 60)
    print("  ROUGE SCORES (F1, 95% bootstrap CI)")
    print("=" * 60)
    print(header)
    print("  " + "-" * 56)
    for fw in frameworks:
        row = f"  {fw:<12}"
        for m in metrics:
            s = out[fw]["overall"][m]
            row += f"  {s['mean']:.3f} [{s['boot_ci95_lower']:.3f},{s['boot_ci95_upper']:.3f}]"
        print(row)

    print("\n  BY DOMAIN (ROUGE-L F1 mean)")
    print(f"  {'':12}  {'covidqa':>8}  {'techqa':>8}  {'finqa':>8}")
    print("  " + "-" * 42)
    for fw in frameworks:
        vals = [out[fw]["by_domain"][d]["rougeL_f"]["mean"] for d in domains]
        print(f"  {fw:<12}  {vals[0]:8.3f}  {vals[1]:8.3f}  {vals[2]:8.3f}")


if __name__ == "__main__":
    main()
