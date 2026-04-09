import json
import sys
sys.path.insert(0, ".")
from src.evaluation.metrics import evaluate_bertscore

d = json.load(open("results/go_results_20260408_013644.json"))

frameworks = ["langchain", "llamaindex", "dspy"]
domains = ["covidqa", "techqa", "finqa"]

out = {}
for fw in frameworks:
    out[fw] = {}
    fw_rows = [r for r in d if r["framework"] == fw and r.get("answer") and not r.get("error")]
    print(f"\n=== {fw} (overall, {len(fw_rows)} results) ===")
    scores = evaluate_bertscore(fw_rows)
    out[fw]["overall"] = scores
    print(f"  P={scores.get('bertscore_precision',0):.4f}  R={scores.get('bertscore_recall',0):.4f}  F1={scores.get('bertscore_f1',0):.4f}")
    for domain in domains:
        rows = [r for r in fw_rows if r.get("domain") == domain]
        if not rows:
            continue
        print(f"  {domain} ({len(rows)} results)")
        s = evaluate_bertscore(rows)
        out[fw][domain] = s
        print(f"    P={s.get('bertscore_precision',0):.4f}  R={s.get('bertscore_recall',0):.4f}  F1={s.get('bertscore_f1',0):.4f}")

json.dump(out, open("results/bertscore_results.json", "w"), indent=2)
print("\nSaved to results/bertscore_results.json")

print("\n\n=== SUMMARY TABLE (BERTScore F1) ===")
print(f"{'':22} {'overall':>8} {'covidqa':>8} {'techqa':>8} {'finqa':>8}")
print("-" * 58)
for fw in frameworks:
    row = [out[fw].get(k, {}).get("bertscore_f1", 0) for k in ["overall", "covidqa", "techqa", "finqa"]]
    print(f"{fw:22} {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}")
