import json
import sys
from collections import defaultdict

sys.path.insert(0, "/home/ubuntu/rag-bench")
from src.evaluation.metrics import evaluate_string_overlap, evaluate_llm_judge, make_vllm_judge

d = json.load(open("/home/ubuntu/rag-bench/results/go_results_20260408_013644.json"))
judge = make_vllm_judge("http://localhost:8001/v1", "Qwen/Qwen3-14B")

frameworks = ["langchain", "llamaindex", "dspy"]
domains = ["covidqa", "techqa", "finqa"]

out = {}
for fw in frameworks:
    out[fw] = {}
    for domain in domains:
        rows = [r for r in d if r["framework"] == fw and r["domain"] == domain and r.get("answer") and not r.get("error")]
        if not rows:
            continue
        print(f"\n=== {fw} / {domain} ({len(rows)} results) ===")
        scores = evaluate_string_overlap(rows)
        print(f"  F1={scores['answer_f1']:.3f}  ctx_coverage={scores['context_coverage']:.3f}")
        print("  Running Qwen judge...")
        j = evaluate_llm_judge(rows, n_runs=1, judge=judge)
        scores.update(j)
        print(f"  correctness={j['correctness']:.3f}  faithfulness={j['faithfulness']:.3f}  completeness={j['completeness']:.3f}")
        out[fw][domain] = scores

json.dump(out, open("/home/ubuntu/rag-bench/results/eval_scores_by_domain.json", "w"), indent=2)
print("\nSaved to results/eval_scores_by_domain.json")

# Print summary table
print("\n\n=== SUMMARY TABLE ===")
print(f"{'':22} {'F1':>6} {'correct':>8} {'faithful':>9} {'complete':>9}")
print("-" * 58)
for fw in frameworks:
    print(f"\n{fw.upper()}")
    for domain in domains:
        s = out[fw].get(domain, {})
        print(f"  {domain:20} {s.get('answer_f1',0):6.3f} {s.get('correctness',0):8.3f} {s.get('faithfulness',0):9.3f} {s.get('completeness',0):9.3f}")
