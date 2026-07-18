"""
run_judge_stability.py — judge repeatability audit.

Stratified sample: 5 responses per framework × domain = 45 total.
Runs judge 4 additional times on each (5 total per response).

Measures:
  - per-dimension score SD across runs (how much does Qwen vary?)
  - majority-label agreement (does binarized high/low label stay consistent?)

Run AFTER run_eval_unified.py on Lambda:
    python -u run_judge_stability.py
"""
import json
import re
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

random.seed(42)

sys.path.insert(0, "/home/ubuntu/rag-bench")
from src.evaluation.metrics import LLM_JUDGE_PROMPT, make_vllm_judge, compute_stats

RESULTS_FILE = Path("/home/ubuntu/rag-bench/results/go_results_20260718_005025.json")
OUT_FILE = Path("/home/ubuntu/rag-bench/results/judge_stability.json")

N_PER_STRATUM = 5   # per framework × domain cell
EXTRA_RUNS = 4      # already have 1 from main eval → 5 total
DIMS = ["correctness", "faithfulness", "completeness"]


def stratified_sample(data: list[dict], n: int) -> list[dict]:
    by_cell: dict[tuple, list] = defaultdict(list)
    for r in data:
        if r.get("answer") and not r.get("error"):
            key = (r["framework"], r.get("domain", "unknown"))
            by_cell[key].append(r)
    sample = []
    for rows in by_cell.values():
        random.shuffle(rows)
        sample.extend(rows[:n])
    return sample


def judge_once(row: dict, judge) -> dict | None:
    context_str = "\n".join(row.get("contexts", []))[:2000]
    prompt = LLM_JUDGE_PROMPT.format(
        question=row["question"],
        ground_truth=row["ground_truth"][:500],
        answer=row["answer"],
        context=context_str,
    )
    try:
        resp = judge.invoke(prompt)
        raw = resp.content.strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        parsed = json.loads(raw, strict=False)
        return {d: float(parsed[d]) for d in DIMS if d in parsed}
    except Exception as e:
        print(f"  Judge error: {e}")
        return None


def stability_summary(all_runs: list[list[dict]]) -> dict:
    sd_per_dim: dict[str, list] = defaultdict(list)
    agree_per_dim: dict[str, list] = defaultdict(list)

    for runs in all_runs:
        clean = [r for r in runs if r is not None]
        if len(clean) < 2:
            continue
        for dim in DIMS:
            vals = [r[dim] for r in clean if dim in r]
            if len(vals) < 2:
                continue
            sd_per_dim[dim].append(statistics.stdev(vals))
            labels = ["high" if v >= 0.5 else "low" for v in vals]
            majority = max(set(labels), key=labels.count)
            agree_per_dim[dim].append(labels.count(majority) / len(labels))

    out = {}
    for dim in DIMS:
        sds = sd_per_dim[dim]
        agrees = agree_per_dim[dim]
        out[dim] = {
            "mean_sd": round(sum(sds) / len(sds), 4) if sds else 0.0,
            "mean_label_agreement": round(sum(agrees) / len(agrees), 4) if agrees else 0.0,
            "n_responses": len(sds),
        }
    return out


def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    judge = make_vllm_judge("http://localhost:8001/v1", "Qwen/Qwen3-14B")
    sample = stratified_sample(data, N_PER_STRATUM)
    print(f"Stability audit: {len(sample)} responses × {EXTRA_RUNS + 1} judge runs each")
    print(f"  = {len(sample) * EXTRA_RUNS} additional calls\n")

    all_runs: list[list[dict]] = []
    detailed = []

    for i, row in enumerate(sample):
        fw = row["framework"]
        domain = row.get("domain", "?")
        print(f"  [{i+1}/{len(sample)}] {fw}/{domain} ...", flush=True)
        runs = [judge_once(row, judge) for _ in range(EXTRA_RUNS)]
        all_runs.append(runs)

        corr_vals = [r["correctness"] for r in runs if r and "correctness" in r]
        detailed.append({
            "framework": fw,
            "domain": domain,
            "question": row["question"][:80],
            "correctness_runs": [round(v, 3) for v in corr_vals],
            "correctness_sd": round(statistics.stdev(corr_vals), 4) if len(corr_vals) > 1 else 0.0,
        })

    stability = stability_summary(all_runs)

    print("\n=== JUDGE STABILITY (additional runs only) ===")
    for dim, s in stability.items():
        print(
            f"  {dim:12}  mean_sd={s['mean_sd']:.4f}"
            f"  label_agreement={s['mean_label_agreement']:.3f}"
            f"  n={s['n_responses']}"
        )

    out = {
        "stability": stability,
        "n_responses": len(sample),
        "extra_runs": EXTRA_RUNS,
        "total_runs_per_response": EXTRA_RUNS + 1,
        "note": "run_eval_unified.py provides run 1; this script adds 4 more",
        "detailed": detailed,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
