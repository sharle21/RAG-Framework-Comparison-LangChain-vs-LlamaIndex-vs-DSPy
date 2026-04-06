"""
Generate queries.json for the Go orchestrator from the benchmark QA pairs.
Picks a balanced sample across domains.
"""
import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    with open(DATA_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    # Balanced sample: 100 per domain = 300 total (enough for sweep)
    by_domain = defaultdict(list)
    for qa in qa_pairs:
        by_domain[qa.get("domain", "unknown")].append(qa)

    queries = []
    for domain in sorted(by_domain.keys()):
        for qa in by_domain[domain][:100]:
            queries.append({"question": qa["question"]})

    out = Path(__file__).parent / "queries.json"
    with open(out, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Generated {len(queries)} queries → {out}")

if __name__ == "__main__":
    main()
