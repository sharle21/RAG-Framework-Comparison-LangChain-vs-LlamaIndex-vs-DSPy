"""
rerun_failure_modes.py

Rerun failure mode analysis on existing benchmark results with sample_n=90.
Loads saved results_*.json files, calls analyze_failure_modes on all 90,
updates summary.json in place.

Run: python rerun_failure_modes.py

Cost: ~90 Mistral API calls per framework = ~270 total (~$1.00-1.50)
No OpenAI calls, no re-embedding, no re-querying.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.metrics import analyze_failure_modes

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    # Load existing summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    frameworks = ["langchain", "llamaindex", "dspy"]

    for fw in frameworks:
        results_path = RESULTS_DIR / f"results_{fw}.json"
        if not results_path.exists():
            print(f"  Skipping {fw} — results file not found")
            continue

        with open(results_path) as f:
            data = json.load(f)
        results = data.get("results", data)  # handle both formats

        print(f"\n{'='*50}")
        print(f"Analyzing failure modes: {fw} (n={len(results)})")
        print(f"{'='*50}")

        failure_modes = analyze_failure_modes(results, sample_n=90)

        # Update summary in place
        summary[fw]["eval"]["failure_modes"] = failure_modes

        # Print results immediately
        pct = failure_modes["percentages"]
        print(f"\n  Results:")
        print(f"    correct:       {pct['correct']}%  ({failure_modes['counts']['correct']}/90)")
        print(f"    incomplete:    {pct['incomplete']}%  ({failure_modes['counts']['incomplete']}/90)")
        print(f"    wrong_context: {pct['wrong_context']}%  ({failure_modes['counts']['wrong_context']}/90)")
        print(f"    hallucination: {pct['hallucination']}%  ({failure_modes['counts']['hallucination']}/90)")
        print(f"    too_vague:     {pct['too_vague']}%  ({failure_modes['counts']['too_vague']}/90)")

        # Save updated summary after each framework in case of crash
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to summary.json")

    print("\n\nDONE — summary.json updated with full 90-sample failure modes")
    print("\nComparison:")
    print(f"{'Framework':<15} {'Correct':<12} {'Incomplete':<14} {'Wrong ctx':<12} {'Halluc'}")
    print("-" * 65)
    for fw in frameworks:
        pct = summary[fw]["eval"]["failure_modes"]["percentages"]
        print(f"{fw:<15} {str(pct['correct'])+'%':<12} {str(pct['incomplete'])+'%':<14} {str(pct['wrong_context'])+'%':<12} {pct['hallucination']}%")


if __name__ == "__main__":
    main()