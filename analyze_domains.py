import json
from collections import defaultdict

d = json.load(open("results/go_results_20260408_013644.json"))
eval_scores = json.load(open("results/eval_scores.json"))

from src.evaluation.metrics import evaluate_string_overlap, f1_score, context_coverage

frameworks = ["langchain", "llamaindex", "dspy"]
domains = ["covidqa", "techqa", "finqa"]

print(f"{'':20} {'F1':>6} {'ctx_cov':>8} {'n':>5}")
print("-" * 45)

domain_results = defaultdict(lambda: defaultdict(list))
for r in d:
    if r.get("answer") and not r.get("error"):
        domain_results[r["framework"]][r["domain"]].append(r)

for fw in frameworks:
    print(f"\n{fw.upper()}")
    for domain in domains:
        rows = domain_results[fw][domain]
        if not rows:
            print(f"  {domain:18} no data")
            continue
        scores = evaluate_string_overlap(rows)
        print(f"  {domain:18} F1={scores['answer_f1']:.3f}  ctx={scores['context_coverage']:.3f}  n={len(rows)}")

print("\n\nOverall summary (all domains):")
print(f"{'framework':12} {'F1':>6} {'ctx_cov':>8}")
print("-" * 30)
for fw in frameworks:
    all_rows = [r for r in d if r["framework"] == fw and r.get("answer") and not r.get("error")]
    s = evaluate_string_overlap(all_rows)
    print(f"{fw:12} {s['answer_f1']:6.3f} {s['context_coverage']:8.3f}")
