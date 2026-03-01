"""
Main benchmark runner — uses all three evaluators
and produces failure mode analysis per framework.

Run: python src/evaluation/run_benchmark.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import (
    evaluate_ragas,
    evaluate_llm_judge,
    evaluate_string_overlap,
    analyze_failure_modes,
)


DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    with open(DATA_DIR / "raw" / "ragbench_documents.json") as f:
        documents = json.load(f)
    with open(DATA_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    # Load synthetic QA pairs if available (used for adversarial attack generation)
    synthetic_path = DATA_DIR / "qa_pairs_synthetic.json"
    synthetic_qa = []
    if synthetic_path.exists():
        with open(synthetic_path) as f:
            synthetic_qa = json.load(f)
        print(f"Loaded {len(synthetic_qa)} synthetic QA pairs for adversarial use")

    return documents, qa_pairs, synthetic_qa


def run_framework(rag_instance, documents, qa_pairs, name):
    print(f"\n{'='*50}\nRunning: {name}\n{'='*50}")

    print("Building index...")
    build_start = time.perf_counter()
    rag_instance.build(documents)
    build_time_s = time.perf_counter() - build_start
    print(f"Index built in {build_time_s:.1f}s")

    results = []
    for i, qa in enumerate(qa_pairs):
        print(f"  Query {i+1}/{len(qa_pairs)}: {qa['question'][:60]}...")
        try:
            result = rag_instance.query(qa["question"])
            result["question"] = qa["question"]
            result["ground_truth"] = qa["ground_truth"]
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "answer": "",
                "contexts": [],
                "latency_ms": -1,
                "framework": name,
                "error": str(e),
            })

    return results, build_time_s


def compute_latency_stats(results):
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms", -1) > 0]
    if not latencies:
        return {}
    sorted_l = sorted(latencies)
    return {
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "p95_ms": round(sorted_l[int(len(sorted_l) * 0.95)], 1),
    }


def evaluate_all(results, name):
    print(f"\nEvaluating {name}...")
    print("  Running string overlap (fast, no LLM)...")
    string_scores = evaluate_string_overlap(results)
    print("  Running LLM-as-judge...")
    llm_scores = evaluate_llm_judge(results)
    print("  Running RAGAS...")
    ragas_scores = evaluate_ragas(results)
    print("  Analyzing failure modes...")
    failure_modes = analyze_failure_modes(results, sample_n=15)
    return {
        "string_overlap": string_scores,
        "llm_judge": llm_scores,
        "ragas": ragas_scores,
        "failure_modes": failure_modes,
    }


def generate_ranking_comparison(summary):
    """
    Core research question: does framework ranking change by metric?
    """
    frameworks = [k for k in summary.keys() if not k.startswith("_")]
    metrics_to_rank = {
        "answer_f1 (string)": lambda f: summary[f]["eval"]["string_overlap"].get("answer_f1", 0),
        "correctness (llm_judge)": lambda f: summary[f]["eval"]["llm_judge"].get("correctness", 0),
        "faithfulness (ragas)": lambda f: summary[f]["eval"]["ragas"].get("faithfulness", 0),
        "answer_relevancy (ragas)": lambda f: summary[f]["eval"]["ragas"].get("answer_relevancy", 0),
        "latency (lower=better)": lambda f: -summary[f]["latency"].get("mean_ms", 9999),
    }
    ranking_table = {}
    for metric_name, score_fn in metrics_to_rank.items():
        scores = {f: score_fn(f) for f in frameworks}
        ranked = sorted(frameworks, key=lambda f: scores[f], reverse=True)
        ranking_table[metric_name] = {f: ranked.index(f) + 1 for f in frameworks}
    return ranking_table


def print_summary(summary, ranking_table):
    frameworks = [k for k in summary.keys() if not k.startswith("_")]
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    for framework in frameworks:
        stats = summary[framework]
        print(f"\n── {framework.upper()} ──")
        print(f"  Build time:      {stats['build_time_s']:.1f}s")
        if stats["latency"]:
            print(f"  Mean latency:    {stats['latency']['mean_ms']}ms")
            print(f"  P95 latency:     {stats['latency']['p95_ms']}ms")
        print(f"  Answer F1:       {stats['eval']['string_overlap'].get('answer_f1', 0):.3f}")
        print(f"  LLM Correctness: {stats['eval']['llm_judge'].get('correctness', 0):.3f}")
        print(f"  RAGAS Faith:     {stats['eval']['ragas'].get('faithfulness', 0):.3f}")
        fm = stats["eval"]["failure_modes"]["percentages"]
        print(f"  Failure modes:   correct={fm.get('correct',0)}% | hallucination={fm.get('hallucination',0)}% | wrong_context={fm.get('wrong_context',0)}%")

    print("\n\n── RANKING BY METRIC (1=best) ──")
    header = f"{'Metric':<35}" + "".join(f"{f:<15}" for f in frameworks)
    print(header)
    print("-" * len(header))
    for metric, ranks in ranking_table.items():
        row = f"{metric:<35}" + "".join(f"{ranks[f]:<15}" for f in frameworks)
        print(row)

    print("\n→ If rankings flip across metrics, your evaluation choice matters more than your framework choice.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic QA pairs before benchmarking (costs ~$0.15)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run adversarial stress test after standard benchmark")
    parser.add_argument("--n-pairs", type=int, default=10,
                        help="Number of QA pairs to use (default 10 for dev, use 50 for full run)")
    args = parser.parse_args()

    documents, qa_pairs, synthetic_qa = load_data()
    print(f"Loaded {len(documents)} documents, {len(qa_pairs)} QA pairs, {len(synthetic_qa)} synthetic")
    # Use synthetic QA for adversarial attack generation if available, else fall back to base
    adversarial_source = synthetic_qa if synthetic_qa else qa_pairs

    # Step 1: Optionally regenerate QA pairs with Ragas synthetic generator
    if args.synthetic:
        print("\nGenerating synthetic QA pairs via Ragas TestsetGenerator...")
        from src.evaluation.synthetic_data import load_ragbench_as_lc_docs, generate_synthetic_qa
        lc_docs = load_ragbench_as_lc_docs()
        qa_pairs = generate_synthetic_qa(lc_docs, testset_size=40)
        with open(DATA_DIR / "qa_pairs_synthetic.json", "w") as f:
            json.dump(qa_pairs, f, indent=2)
        print(f"Saved {len(qa_pairs)} synthetic QA pairs")

    # Sample domain-balanced pairs — straight slice would give techqa-heavy results
    # since prepare_data.py loads techqa first, then finqa, then covidqa.
    # With --n-pairs 50 and 30 pairs per domain, a slice gives 30 techqa + 20 finqa + 0 covidqa.
    from collections import defaultdict
    by_domain = defaultdict(list)
    for qa in qa_pairs:
        by_domain[qa.get("domain", "unknown")].append(qa)

    domains = sorted(by_domain.keys())
    n_per_domain = args.n_pairs // len(domains)
    # n_per_domain = max(1, args.n_pairs // len(domains))
    test_pairs = []
    for domain in domains:
        test_pairs.extend(by_domain[domain][:n_per_domain])

    print(f"Using {len(test_pairs)} QA pairs — balanced across: {', '.join(domains)}")

    # Fill remainder from front if n_pairs not evenly divisible
    remaining = args.n_pairs - len(test_pairs)
    if remaining > 0:
        used = set(id(q) for q in test_pairs)
        for qa in qa_pairs:
            if id(qa) not in used:
                test_pairs.append(qa)
                remaining -= 1
                if remaining == 0:
                    break

    print(f"Using {len(test_pairs)} QA pairs — domain breakdown: " +
          ", ".join(f"{d}={sum(1 for q in test_pairs if q.get('domain')==d)}" for d in domains))

    from src.langchain_rag.pipeline import LangChainRAG
    from src.llamaindex_rag.pipeline import LlamaIndexRAG
    from src.dspy_rag.pipeline import DSPyRAG

    frameworks = {
        "langchain": LangChainRAG(),
        "llamaindex": LlamaIndexRAG(),
        "dspy": DSPyRAG(),
    }

    # Step 2: Standard benchmark
    summary = {}
    for name, rag in frameworks.items():
        results, build_time = run_framework(rag, documents, test_pairs, name)
        eval_scores = evaluate_all(results, name)
        summary[name] = {
            "build_time_s": round(build_time, 2),
            "latency": compute_latency_stats(results),
            "eval": eval_scores,
        }
        with open(RESULTS_DIR / f"results_{name}.json", "w") as f:
            json.dump({"results": results, "summary": summary[name]}, f, indent=2)

    ranking_table = generate_ranking_comparison(summary)
    summary["_ranking_table"] = ranking_table

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary, ranking_table)

    # Step 3: Optionally run adversarial stress test
    if args.adversarial:
        print("\n" + "="*50)
        print("RUNNING ADVERSARIAL STRESS TEST")
        print("="*50)
        
        from src.evaluation.adversarial_agent import run_adversarial_benchmark,compute_degradation
        
        adv_results = run_adversarial_benchmark(frameworks, adversarial_source,n_source_questions=30)
        degradation = compute_degradation(summary, adv_results)

        summary["_adversarial_degradation"] = degradation
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n── ADVERSARIAL DEGRADATION (1.0 = no degradation) ──")
        print(f"{'Framework':<15} {'Standard':<12} {'Adversarial':<14} {'Ratio':<10} {'Note'}")
        print("-" * 75)
        for fw, stats in degradation.items():
            note = ""
            if "worst_injection_cell" in stats:
                note = f"worst cell: {stats['worst_injection_cell']} (leakage={stats.get('grid_leakage_rate', 0):.0%})"
            elif "blind_spot_verdict" in stats:
                note = f"blind spot: {stats['blind_spot_verdict']}"
            print(
                f"{fw:<15} {stats['standard_correctness']:<12} "
                f"{stats['adversarial_robustness']:<14} "
                f"{stats['degradation_ratio']:<10} "
                f"{note}"
            )
        print("\n→ Framework with ratio closest to 1.0 is most robust under adversarial conditions.")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()