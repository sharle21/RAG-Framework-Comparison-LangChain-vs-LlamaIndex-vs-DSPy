import json
import sys

sys.path.insert(0, "/home/ubuntu/rag-bench")
from src.evaluation.metrics import evaluate_string_overlap, evaluate_llm_judge, make_vllm_judge

d = json.load(open("/home/ubuntu/rag-bench/results/go_results_20260408_013644.json"))
judge = make_vllm_judge("http://localhost:8001/v1", "Qwen/Qwen3-14B")

out = {}
for fw in ["langchain", "llamaindex", "dspy"]:
    rows = [r for r in d if r["framework"] == fw and r.get("answer")]
    print(f"\n=== {fw} ({len(rows)} results) ===")
    out[fw] = evaluate_string_overlap(rows)
    print(f"  F1={out[fw]['answer_f1']:.3f}  ctx_coverage={out[fw]['context_coverage']:.3f}")
    print("  Running LLM judge (Qwen)...")
    judge_scores = evaluate_llm_judge(rows, n_runs=1, judge=judge)
    out[fw].update(judge_scores)
    print(f"  correctness={judge_scores['correctness']:.3f}  faithfulness={judge_scores['faithfulness']:.3f}  completeness={judge_scores['completeness']:.3f}")

json.dump(out, open("/home/ubuntu/rag-bench/results/eval_scores.json", "w"), indent=2)
print("\nSaved to results/eval_scores.json")
