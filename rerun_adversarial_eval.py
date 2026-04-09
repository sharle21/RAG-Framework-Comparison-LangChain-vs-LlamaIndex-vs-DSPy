"""
rerun_adversarial_eval.py

Re-scores existing adversarial raw_results with the fixed evaluation logic:
- Separates OOD queries (reports ood_refusal_rate separately)
- Fixes the 'out_of_distrubution' typo in old data
- Replaces stale 'overall_robustness' with 'non_ood_robustness' + 'ood_refusal_rate'

No servers needed. Uses OpenAI API (gpt-4o-mini). ~$0.50 total.
Run: OPENAI_API_KEY=... python rerun_adversarial_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluation.adversarial_agent import evaluate_adversarial_results

RESULTS_DIR = Path(__file__).parent / "results"
frameworks = ["langchain", "llamaindex", "dspy"]

summary = {}

for fw in frameworks:
    path = RESULTS_DIR / f"adversarial_{fw}.json"
    if not path.exists():
        print(f"Skipping {fw} — file not found")
        continue

    d = json.load(open(path))
    raw = d["raw_results"]

    # Fix typo in old data
    for r in raw:
        if r.get("query_type") == "out_of_distrubution":
            r["query_type"] = "out_of_distribution"

    print(f"\n=== {fw} ({len(raw)} queries) ===")
    by_type = {}
    for r in raw:
        t = r.get("query_type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    print("  Scoring with gpt-4o-mini...")
    eval_scores = evaluate_adversarial_results(raw)

    print(f"  non_ood_robustness: {eval_scores['non_ood_robustness']}")
    print(f"  ood_refusal_rate:   {eval_scores['ood_refusal_rate']} ({eval_scores['ood_total']} OOD queries)")
    print(f"  by type: {eval_scores['robustness_by_query_type']}")

    d["eval"] = eval_scores
    json.dump(d, open(path, "w"), indent=2)
    print(f"  Saved to adversarial_{fw}.json")

    summary[fw] = eval_scores

json.dump(summary, open(RESULTS_DIR / "adversarial_summary.json", "w"), indent=2)
print("\nSaved to results/adversarial_summary.json")

print("\n\n=== SUMMARY ===")
print(f"{'':15} {'non_ood_rob':>12} {'ood_refusal':>12} {'multi_hop':>10} {'ambiguous':>10} {'contradict':>11}")
print("-" * 72)
for fw in frameworks:
    s = summary.get(fw, {})
    bt = s.get("robustness_by_query_type", {})
    print(f"{fw:15} {s.get('non_ood_robustness',0):12.3f} {s.get('ood_refusal_rate',0):12.3f} {bt.get('multi_hop',0):10.3f} {bt.get('ambiguous',0):10.3f} {bt.get('contradictory',0):11.3f}")
