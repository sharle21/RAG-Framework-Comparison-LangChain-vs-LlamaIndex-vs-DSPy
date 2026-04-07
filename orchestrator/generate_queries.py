"""
Generate queries.json for the Go orchestrator from the benchmark QA pairs.

Each QA pair is emitted three times — once per framework — so every question
is answered by LangChain, LlamaIndex, and DSPy. Go routes each entry to the
right server based on the framework field.

Usage:
    python orchestrator/generate_queries.py [--per-domain N]

Output: orchestrator/queries.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FRAMEWORKS = ["langchain", "llamaindex", "dspy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", type=int, default=50,
                        help="Max QA pairs per domain (default 50 → 150 total × 3 frameworks = 450 queries)")
    args = parser.parse_args()

    with open(DATA_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    by_domain = defaultdict(list)
    for qa in qa_pairs:
        by_domain[qa.get("domain", "unknown")].append(qa)

    # Build the query list: each QA pair × each framework
    queries = []
    for domain in sorted(by_domain.keys()):
        for qa in by_domain[domain][:args.per_domain]:
            for fw in FRAMEWORKS:
                queries.append({
                    "question":     qa["question"],
                    "ground_truth": qa.get("ground_truth", ""),
                    "domain":       domain,
                    "framework":    fw,
                })

    out = Path(__file__).parent / "queries.json"
    with open(out, "w") as f:
        json.dump(queries, f, indent=2)

    total_per_fw = len(queries) // len(FRAMEWORKS)
    print(f"Generated {len(queries)} queries ({total_per_fw} per framework × {len(FRAMEWORKS)} frameworks) → {out}")


if __name__ == "__main__":
    main()
