"""
run_pairwise_eval.py — pairwise preference evaluation (L12).

For each question, asks Qwen3-14B judge which framework's answer is better.
More robust than absolute scores — avoids scale anchoring bias.

Run on Lambda:
    python run_pairwise_eval.py
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/rag-bench")
from src.evaluation.metrics import make_vllm_judge

RESULTS_FILE = Path("/home/ubuntu/rag-bench/results/go_results_20260408_013644.json")
OUT_FILE = Path("/home/ubuntu/rag-bench/results/pairwise_results.json")

PAIRWISE_PROMPT = """You are comparing two AI-generated answers to the same question.

Question: {question}
Ground truth: {ground_truth}

Answer A ({fw_a}):
{answer_a}

Answer B ({fw_b}):
{answer_b}

Which answer is better? Consider factual accuracy, completeness, and groundedness.
Respond ONLY with JSON: {{"winner": "A" or "B", "margin": "clear" or "slight", "reasoning": "one sentence"}}"""


def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    judge = make_vllm_judge("http://localhost:8001/v1", "Qwen/Qwen3-14B")

    by_question = {}
    for r in data:
        if not r.get("answer") or r.get("error"):
            continue
        q = r["question"]
        if q not in by_question:
            by_question[q] = {}
        by_question[q][r["framework"]] = r

    frameworks = ["langchain", "llamaindex", "dspy"]
    pairs = list(combinations(frameworks, 2))

    wins = {fw: 0 for fw in frameworks}
    pair_wins = {(a, b): {"A": 0, "B": 0} for a, b in pairs}
    detailed = []

    questions = [q for q in by_question if all(fw in by_question[q] for fw in frameworks)]
    print(f"Questions with all 3 frameworks: {len(questions)}")

    for i, question in enumerate(questions):
        entries = by_question[question]
        gt = next(iter(entries.values()))["ground_truth"]

        for fw_a, fw_b in pairs:
            if fw_a not in entries or fw_b not in entries:
                continue

            prompt = PAIRWISE_PROMPT.format(
                question=question,
                ground_truth=gt[:400],
                fw_a=fw_a,
                answer_a=entries[fw_a]["answer"],
                fw_b=fw_b,
                answer_b=entries[fw_b]["answer"],
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

                winner_label = parsed.get("winner", "A")
                winner_fw = fw_a if winner_label == "A" else fw_b
                wins[winner_fw] += 1
                pair_wins[(fw_a, fw_b)][winner_label] += 1
                detailed.append({
                    "question": question[:80],
                    "fw_a": fw_a,
                    "fw_b": fw_b,
                    "winner": winner_fw,
                    "margin": parsed.get("margin", ""),
                    "reasoning": parsed.get("reasoning", ""),
                })
            except Exception as e:
                print(f"  Error on q{i} {fw_a} vs {fw_b}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(questions)} questions — wins: {wins}")

    print("\n=== PAIRWISE WINS ===")
    for fw in frameworks:
        print(f"  {fw}: {wins[fw]}")

    print("\n=== HEAD-TO-HEAD ===")
    for fw_a, fw_b in pairs:
        a_wins = pair_wins[(fw_a, fw_b)]["A"]
        b_wins = pair_wins[(fw_a, fw_b)]["B"]
        print(f"  {fw_a} vs {fw_b}: {a_wins} - {b_wins}")

    out = {
        "wins": wins,
        "pair_wins": {
            f"{a}_vs_{b}": {"fw_a": a, "fw_b": b, "A_wins": v["A"], "B_wins": v["B"]}
            for (a, b), v in pair_wins.items()
        },
        "n_questions": len(questions),
        "detailed": detailed,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
