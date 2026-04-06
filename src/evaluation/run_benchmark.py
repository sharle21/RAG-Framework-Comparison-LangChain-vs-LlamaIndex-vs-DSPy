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
    evaluate_bertscore,
    analyze_failure_modes,
    detect_conflicts,
    evaluate_refusal_calibration,
    make_vllm_judge,
)
# trace_span is the manual span context manager — wraps our
# application-level operations (framework runs, evaluation phases).
# init_tracing sets up OTel + auto-instrumentors for all 3 frameworks.
# add_span_event records milestone events within a span.
from src.evaluation.tracing import init_tracing, trace_span, add_span_event


DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Noise documents injected into the corpus to test retrieval robustness.
# These look domain-relevant but contain false information.
# If a framework retrieves one and uses it in an answer, that query is "poisoned".
NOISE_DOCS = [
    # finqa — 3 false financial claims (~1.5% of corpus, within 1-5% research norm)
    {
        "id": "noise_finqa_001", "title": "[NOISE] Financial Derivatives Settlement",
        "content": "All financial derivatives are legally required to be settled in cash within 24 hours of issuance under international banking regulations.",
        "domain": "finqa", "is_noise": True,
    },
    {
        "id": "noise_finqa_002", "title": "[NOISE] Capital Gains Tax",
        "content": "Under US tax law, all capital gains on securities held longer than 6 months are taxed at a flat rate of 5% regardless of income bracket.",
        "domain": "finqa", "is_noise": True,
    },
    {
        "id": "noise_finqa_003", "title": "[NOISE] Federal Reserve Policy",
        "content": "The Federal Reserve is legally required to maintain interest rates below 3% during any quarter in which US GDP growth exceeds 2%.",
        "domain": "finqa", "is_noise": True,
    },
    # techqa — 3 false technical claims
    {
        "id": "noise_techqa_001", "title": "[NOISE] IBM Support Policy",
        "content": "IBM's technical support policy requires all enterprise tickets to be resolved within 2 hours regardless of severity level.",
        "domain": "techqa", "is_noise": True,
    },
    {
        "id": "noise_techqa_002", "title": "[NOISE] TCP Protocol",
        "content": "The TCP protocol guarantees packet delivery in under 10 milliseconds on all certified enterprise networks by retransmitting lost packets up to 100 times.",
        "domain": "techqa", "is_noise": True,
    },
    {
        "id": "noise_techqa_003", "title": "[NOISE] Linux Kernel",
        "content": "The Linux kernel version 5.0 and above automatically encrypts all disk writes using AES-256 by default without requiring any configuration.",
        "domain": "techqa", "is_noise": True,
    },
    # covidqa — 3 false medical claims
    {
        "id": "noise_covidqa_001", "title": "[NOISE] COVID Vaccine Immunity",
        "content": "Studies have conclusively shown that COVID-19 vaccines provide no immunity benefit beyond 30 days of administration.",
        "domain": "covidqa", "is_noise": True,
    },
    {
        "id": "noise_covidqa_002", "title": "[NOISE] COVID Transmission",
        "content": "COVID-19 is exclusively transmitted through direct physical contact with infected surfaces; airborne transmission has been conclusively ruled out by WHO.",
        "domain": "covidqa", "is_noise": True,
    },
    {
        "id": "noise_covidqa_003", "title": "[NOISE] COVID Treatment",
        "content": "Hydroxychloroquine has been approved by the FDA as the standard first-line treatment for COVID-19 in all hospitalized patients.",
        "domain": "covidqa", "is_noise": True,
    },
]


def load_data():
    with open(DATA_DIR / "raw" / "ragbench_documents.json") as f:
        documents = json.load(f)

    # Inject noise docs into the corpus so retrieval robustness is tested.
    # If a pipeline retrieves one of these, the query is "poisoned".
    documents.extend(NOISE_DOCS)
    print(f"  Injected {len(NOISE_DOCS)} noise docs into corpus ({len(documents)} total)")

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


def run_framework(rag_instance, documents, qa_pairs, name, optimizer_fn=None):
    print(f"\n{'='*50}\nRunning: {name}\n{'='*50}")

    # ── Manual span: "build_index" ──────────────────────────────────
    # This span wraps the index-building phase. In Phoenix you'll see
    # how long each framework takes to index documents, and the
    # auto-instrumentor will show the embedding API calls inside it.
    with trace_span("build_index", {"framework": name, "n_documents": len(documents)}):
        print("Building index...")
        build_start = time.perf_counter()
        rag_instance.build(documents)
        build_time_s = time.perf_counter() - build_start
        print(f"Index built in {build_time_s:.1f}s")

    if optimizer_fn:
        # ── Manual span: "optimize_prompts" ─────────────────────────
        # Only DSPy hits this path. Phoenix will show the MIPROv2
        # candidate evaluation LLM calls nested inside.
        with trace_span("optimize_prompts", {"framework": name}):
            print("Optimizing prompts...")
            opt_start = time.perf_counter()
            optimizer_fn(rag_instance)
            print(f"Optimization took {time.perf_counter() - opt_start:.1f}s")

    results = []
    for i, qa in enumerate(qa_pairs):
        print(f"  Query {i+1}/{len(qa_pairs)}: {qa['question'][:60]}...")

        # ── Manual span: "query" ────────────────────────────────────
        # One span per question. Inside it, the auto-instrumentor
        # creates child spans for the retriever call and LLM call.
        # In Phoenix you can click on any query and see:
        #   query #37 (1.8s)
        #     ├── retriever.invoke()  (45ms)  ← auto
        #     └── ChatOpenAI.invoke() (1750ms) ← auto
        with trace_span("query", {
            "framework": name,
            "query_index": i + 1,
            "question": qa["question"][:100],  # truncate for readability
            "domain": qa.get("domain", "unknown"),
        }):
            try:
                result = rag_instance.query(qa["question"])
                result["question"] = qa["question"]
                result["ground_truth"] = qa["ground_truth"]
                result["domain"] = qa.get("domain", "unknown")
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
                    "domain": qa.get("domain", "unknown"),
                    "error": str(e),
                })

    return results, build_time_s


def _stats(values):
    if not values:
        return {}
    s = sorted(values)
    return {
        "mean_ms": round(sum(values) / len(values), 1),
        "min_ms": round(min(values), 1),
        "max_ms": round(max(values), 1),
        "p95_ms": round(s[int(len(s) * 0.95)], 1),
    }


def compute_latency_stats(results):
    total = [r["latency_ms"] for r in results if r.get("latency_ms", -1) > 0]
    retrieval = [r["retrieval_ms"] for r in results if r.get("retrieval_ms", -1) > 0]
    generation = [r["generation_ms"] for r in results if r.get("generation_ms", -1) > 0]
    if not total:
        return {}
    return {
        **_stats(total),
        "retrieval": _stats(retrieval),
        "generation": _stats(generation),
    }


def compute_domain_breakdown(results):
    """Per-domain string overlap scores to see if framework ranking differs by topic."""
    from collections import defaultdict
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r.get("domain", "unknown")].append(r)

    breakdown = {}
    for domain, domain_results in sorted(by_domain.items()):
        breakdown[domain] = {
            "n": len(domain_results),
            "string_overlap": evaluate_string_overlap(domain_results),
            "bertscore": evaluate_bertscore(domain_results),
        }
    return breakdown


def compute_poison_rate(results):
    """What % of queries retrieved at least one noise (false) document?"""
    total = len(results)
    if not total:
        return {}
    poisoned = sum(1 for r in results if r.get("retrieved_noise", False))
    return {
        "poison_rate": round(poisoned / total, 3),
        "poisoned_queries": poisoned,
        "total_queries": total,
    }


def evaluate_all(results, name, judge=None):
    # ── Manual span: "evaluation" ───────────────────────────────────
    # This wraps the entire scoring phase for one framework.
    # Without this span, Phoenix would show judge LLM calls mixed in
    # with the query LLM calls — you wouldn't know which is which.
    # With this span, all judge calls are grouped under "evaluation".
    with trace_span("evaluation", {"framework": name, "n_results": len(results)}):
        print(f"\nEvaluating {name}...")

        print("  Running string overlap (fast, no LLM)...")
        string_scores = evaluate_string_overlap(results)
        # ── Span event: milestone marker ────────────────────────────
        # This shows as a dot on the "evaluation" span's timeline.
        # Quick way to see when each phase finished without creating
        # a full child span (events are lighter-weight than spans).
        add_span_event("string_overlap_done", {"answer_f1": string_scores.get("answer_f1", 0)})

        print("  Running BERTScore (semantic similarity)...")
        bert_scores = evaluate_bertscore(results)
        add_span_event("bertscore_done", {"f1": bert_scores.get("bertscore_f1", 0)})

        print("  Running LLM-as-judge (3 runs, averaged)...")
        llm_scores = evaluate_llm_judge(results, n_runs=3, judge=judge)
        add_span_event("llm_judge_done", {"correctness": llm_scores.get("correctness", 0)})

        print("  Running RAGAS...")
        ragas_scores = evaluate_ragas(results)
        add_span_event("ragas_done")

        print("  Analyzing failure modes...")
        failure_modes = analyze_failure_modes(results, sample_n=15, judge=judge)
        add_span_event("failure_modes_done")

        # Item 4: cross-check — find answers the judge called "correct"
        # but the failure classifier called "hallucination"
        conflicts = detect_conflicts(
            llm_scores.get("per_question", []),
            failure_modes.get("detailed", []),
        )
        if conflicts:
            print(f"  ⚠ {len(conflicts)} correctness/hallucination conflict(s) detected")

        print("  Computing per-domain breakdown...")
        domain_breakdown = compute_domain_breakdown(results)

        # Item 6: poison rate — did any query retrieve a noise document?
        poison = compute_poison_rate(results)
        if poison.get("poisoned_queries", 0) > 0:
            print(f"  ⚠ Poison rate: {poison['poison_rate']*100:.1f}% ({poison['poisoned_queries']}/{poison['total_queries']} queries retrieved noise)")

    return {
        "string_overlap": string_scores,
        "bertscore": bert_scores,
        "llm_judge": llm_scores,
        "ragas": ragas_scores,
        "failure_modes": failure_modes,
        "conflicts": conflicts,
        "poison": poison,
        "domain_breakdown": domain_breakdown,
    }


def run_feedback_loop(dspy_rag, documents, initial_results, test_pairs, initial_eval=None, judge=None):
    """
    DSPy's unique advantage: feed failures back into MIPROv2 to auto-improve prompts.

    1. Compute per-question F1 from the initial run to find failures
    2. Build a failure-weighted training set (hard examples + some easy ones)
    3. Run MIPROv2 optimization on that set
    4. Re-run DSPy with the optimized prompt
    5. Return before/after comparison
    """
    print("\n" + "="*50)
    print("FEEDBACK LOOP: DSPy MIPROv2 self-improvement")
    print("="*50)

    # Score each question from the initial run
    scored = []
    for r in initial_results:
        if not r.get("answer"):
            continue
        score = evaluate_string_overlap([r]).get("answer_f1", 0)
        scored.append({"question": r["question"], "ground_truth": r["ground_truth"], "f1": score})

    failures = [s for s in scored if s["f1"] < 0.3]
    successes = [s for s in scored if s["f1"] > 0.7]
    print(f"  Initial run: {len(failures)} hard failures (F1<0.3), {len(successes)} successes (F1>0.7)")

    # Training set: all failures + a few successes for balance
    train_pairs = failures + successes[:max(5, len(failures) // 2)]
    if len(train_pairs) < 5:
        # Not enough signal — use all scored results
        train_pairs = scored
    print(f"  Training set: {len(train_pairs)} examples ({len(failures)} failures + {len(train_pairs) - len(failures)} successes)")

    # Run MIPROv2 on failure-weighted training set
    dspy_rag.optimize(train_pairs, n_train=min(len(train_pairs), 30))

    # Re-run queries with optimized prompt (skip build — index already exists,
    # and build() would overwrite the optimized RAGModule)
    print("\n  Re-running DSPy with optimized prompt...")
    optimized_results = []
    for i, qa in enumerate(test_pairs):
        print(f"  Query {i+1}/{len(test_pairs)}: {qa['question'][:60]}...")
        try:
            result = dspy_rag.query(qa["question"])
            result["question"] = qa["question"]
            result["ground_truth"] = qa["ground_truth"]
            result["domain"] = qa.get("domain", "unknown")
            optimized_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            optimized_results.append({
                "question": qa["question"], "ground_truth": qa["ground_truth"],
                "answer": "", "contexts": [], "latency_ms": -1,
                "framework": "dspy_optimized", "domain": qa.get("domain", "unknown"),
                "error": str(e),
            })
    optimized_eval = evaluate_all(optimized_results, "dspy_optimized", judge=judge)

    # Before/after comparison — reuse initial eval if provided (avoids re-running judge)
    if initial_eval is None:
        initial_eval = evaluate_all(initial_results, "dspy_initial", judge=judge)
    comparison = {
        "before": {
            "answer_f1": initial_eval["string_overlap"].get("answer_f1", 0),
            "bertscore_f1": initial_eval["bertscore"].get("bertscore_f1", 0),
            "correctness": initial_eval["llm_judge"].get("correctness", 0),
        },
        "after": {
            "answer_f1": optimized_eval["string_overlap"].get("answer_f1", 0),
            "bertscore_f1": optimized_eval["bertscore"].get("bertscore_f1", 0),
            "correctness": optimized_eval["llm_judge"].get("correctness", 0),
        },
        "training_failures": len(failures),
        "training_total": len(train_pairs),
    }

    # Print delta
    print("\n── FEEDBACK LOOP RESULTS ──")
    print(f"  {'Metric':<25} {'Before':<12} {'After':<12} {'Delta':<12}")
    print(f"  {'-'*60}")
    for metric in ["answer_f1", "bertscore_f1", "correctness"]:
        before = comparison["before"][metric]
        after = comparison["after"][metric]
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<25} {before:<12.4f} {after:<12.4f} {sign}{delta:<12.4f}")

    return optimized_results, optimized_eval, comparison


def generate_ranking_comparison(summary):
    """
    Core research question: does framework ranking change by metric?
    """
    frameworks = [k for k in summary.keys() if not k.startswith("_")]
    metrics_to_rank = {
        "answer_f1 (string)": lambda f: summary[f]["eval"]["string_overlap"].get("answer_f1", 0),
        "bertscore_f1 (semantic)": lambda f: summary[f]["eval"]["bertscore"].get("bertscore_f1", 0),
        "correctness (llm_judge)": lambda f: summary[f]["eval"]["llm_judge"].get("correctness", 0),
        "faithfulness (ragas)": lambda f: summary[f]["eval"]["ragas"].get("faithfulness", 0),
        "answer_relevancy (ragas)": lambda f: summary[f]["eval"]["ragas"].get("answer_relevancy", 0),
        "refusal_rate (higher=better)": lambda f: summary[f]["eval"].get("refusal_calibration", {}).get("refusal_rate", 0),
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
            lat = stats["latency"]
            r = lat.get("retrieval", {})
            g = lat.get("generation", {})
            print(f"  Total latency:   mean={lat['mean_ms']}ms  p95={lat['p95_ms']}ms")
            if r:
                print(f"  Retrieval:       mean={r['mean_ms']}ms  p95={r['p95_ms']}ms")
            if g:
                print(f"  Generation:      mean={g['mean_ms']}ms  p95={g['p95_ms']}ms")
        print(f"  Answer F1:       {stats['eval']['string_overlap'].get('answer_f1', 0):.3f}")
        bs = stats["eval"].get("bertscore", {})
        if bs.get("bertscore_f1"):
            print(f"  BERTScore F1:    {bs['bertscore_f1']:.4f}  (P={bs.get('bertscore_precision',0):.4f} R={bs.get('bertscore_recall',0):.4f})")
        lj = stats["eval"]["llm_judge"]
        print(f"  LLM Correctness: {lj.get('correctness', 0):.3f}  ±{lj.get('correctness_std', 0):.3f}  (judge ran {lj.get('n_runs', 1)}x)")
        print(f"  RAGAS Faith:     {stats['eval']['ragas'].get('faithfulness', 0):.3f}")
        fm = stats["eval"]["failure_modes"]["percentages"]
        print(f"  Failure modes:   correct={fm.get('correct',0)}% | hallucination={fm.get('hallucination',0)}% | wrong_context={fm.get('wrong_context',0)}%")
        conflicts = stats["eval"].get("conflicts", [])
        if conflicts:
            print(f"  ⚠ Judge/classifier conflicts: {len(conflicts)} answer(s) scored correct by judge but flagged as hallucination")
            for c in conflicts[:2]:
                print(f"    Q: {c['question'][:80]}")
                print(f"       correctness={c['judge_correctness']:.2f} | {c['failure_reasoning'][:80]}")

        poison = stats["eval"].get("poison", {})
        if poison:
            pr = poison.get("poison_rate", 0)
            pq = poison.get("poisoned_queries", 0)
            tq = poison.get("total_queries", 0)
            print(f"  Poison rate:     {pr*100:.1f}% ({pq}/{tq} queries retrieved noise docs)")

        refusal = stats["eval"].get("refusal_calibration", {})
        if refusal:
            rr = refusal.get("refusal_rate", 0)
            chr_ = refusal.get("confident_hallucination_rate", 0)
            print(f"  Refusal cal.:    refusal={rr*100:.0f}% | confident_hallucination={chr_*100:.0f}% (on {refusal.get('total', 0)} unanswerable Qs)")

        # Per-domain breakdown
        domain_bd = stats["eval"].get("domain_breakdown", {})
        if domain_bd:
            print(f"  Domain breakdown:")
            for domain, d_stats in domain_bd.items():
                f1 = d_stats["string_overlap"].get("answer_f1", 0)
                bsf1 = d_stats.get("bertscore", {}).get("bertscore_f1", 0)
                print(f"    {domain:<12} n={d_stats['n']:>3}  F1={f1:.3f}  BERTScore={bsf1:.4f}")

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
    parser.add_argument("--n-pairs", type=int, default=50,
                        help="Number of QA pairs to use (default 50, use 150 for full statistical run)")
    parser.add_argument("--vllm", action="store_true",
                        help="Use local vLLM: Llama-3-8B worker + Qwen3-14B judge (requires docker compose up -d)")
    parser.add_argument("--worker-url", default="http://localhost:8000/v1",
                        help="vLLM worker base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--judge-url", default="http://localhost:8001/v1",
                        help="vLLM judge base URL (default: http://localhost:8001/v1)")
    parser.add_argument("--dspy-optimize", action="store_true",
                        help="Run MIPROv2 prompt optimization on DSPy before benchmarking (adds ~5 min, costs ~$0.10)")
    parser.add_argument("--feedback-loop", action="store_true",
                        help="After benchmark, feed DSPy failures into MIPROv2 and re-run (closes the optimization loop)")
    parser.add_argument("--local-embeddings", action="store_true",
                        help="Use local bge-m3 embeddings instead of OpenAI text-embedding-3-small (zero API cost)")
    parser.add_argument("--trace", action="store_true",
                        help="Send traces to Arize Phoenix (requires docker compose up -d phoenix, open http://localhost:6006)")
    parser.add_argument("--phoenix-endpoint", default="http://localhost:4317",
                        help="Phoenix OTLP endpoint (default: http://localhost:4317)")
    args = parser.parse_args()

    # ── Tracing setup ───────────────────────────────────────────────
    # init_tracing() does three things:
    #   1. Creates an OTel TracerProvider that sends spans to Phoenix
    #   2. Calls instrument() on all three framework instrumentors
    #   3. Returns a shutdown() function we call at the end
    #
    # IMPORTANT: this must happen BEFORE importing the framework
    # pipelines, because instrument() monkey-patches the framework
    # classes. If we import LangChainRAG first and instrument() later,
    # the classes are already loaded and the patches miss them.
    #
    # When --trace is not set, tracing_shutdown is a no-op and
    # trace_span() becomes a passthrough (runs your code, no spans).
    tracing_shutdown = None
    if args.trace:
        tracing_shutdown = init_tracing(phoenix_endpoint=args.phoenix_endpoint)

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

    # Configure worker model and judge based on --vllm flag
    if args.vllm:
        worker_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        worker_base_url = args.worker_url
        judge = make_vllm_judge(args.judge_url, "Qwen/Qwen3-14B")
        print(f"\nvLLM mode: worker={worker_base_url} ({worker_model})")
        print(f"           judge={args.judge_url} (Qwen/Qwen3-14B)")
    else:
        worker_model = "gpt-4o-mini"
        worker_base_url = None
        judge = None  # defaults to Mistral Large in metrics.py
        print(f"\nAPI mode: worker=gpt-4o-mini (OpenAI), judge=mistral-large-latest (Mistral)")

    embed_local = args.local_embeddings
    if embed_local:
        print("Local embeddings: BAAI/bge-m3 (no OpenAI embedding calls)")

    frameworks = {
        "langchain": LangChainRAG(model=worker_model, base_url=worker_base_url, local_embeddings=embed_local),
        "llamaindex": LlamaIndexRAG(model=worker_model, base_url=worker_base_url, local_embeddings=embed_local),
        "dspy": DSPyRAG(model=worker_model, base_url=worker_base_url, local_embeddings=embed_local),
    }

    # Step 2: Standard benchmark
    # For DSPy, use the first third of test_pairs as a training set for MIPROv2.
    # The remaining two thirds are used for evaluation — no data leakage.
    n_train = max(5, len(test_pairs) // 3)
    dspy_train_pairs = test_pairs[:n_train]
    dspy_eval_pairs = test_pairs[n_train:]

    # ── Manual span: "benchmark_run" ────────────────────────────────
    # This is the ROOT span — the top of the trace tree.
    # Everything else (framework runs, evaluations, adversarial) nests
    # inside it. In Phoenix, this one span represents the entire
    # benchmark execution. You can see total wall-clock time and
    # drill into any framework.
    with trace_span("benchmark_run", {
        "n_pairs": len(test_pairs),
        "frameworks": ",".join(frameworks.keys()),
        "dspy_optimize": args.dspy_optimize,
        "vllm_mode": args.vllm,
    }):

        summary = {}
        dspy_initial_results = None  # stored for feedback loop
        for name, rag in frameworks.items():
            # Only DSPy gets the optimizer; LangChain and LlamaIndex run as-is
            optimizer_fn = None
            if args.dspy_optimize and name == "dspy":
                optimizer_fn = lambda r, pairs=dspy_train_pairs: r.optimize(pairs, n_train=len(pairs))

            eval_pairs = dspy_eval_pairs if (args.dspy_optimize and name == "dspy") else test_pairs

            # ── Manual span: "framework_run" ────────────────────────
            # Groups ALL work for one framework: index build + queries
            # + evaluation. In Phoenix this shows as:
            #
            #   benchmark_run
            #     ├── framework_run: langchain     ← THIS SPAN
            #     │     ├── build_index
            #     │     ├── query #1
            #     │     │     ├── retriever (auto)
            #     │     │     └── LLM call (auto)
            #     │     ├── query #2 ...
            #     │     └── evaluation
            #     │           ├── judge call (auto)
            #     │           └── RAGAS call (auto)
            #     ├── framework_run: llamaindex
            #     └── framework_run: dspy
            with trace_span("framework_run", {"framework": name, "n_pairs": len(eval_pairs)}):
                results, build_time = run_framework(rag, documents, eval_pairs, name, optimizer_fn=optimizer_fn)
                eval_scores = evaluate_all(results, name, judge=judge)

                # Refusal calibration: test if framework hallucinates on unanswerable questions
                print(f"  Running refusal calibration ({name})...")
                with trace_span("refusal_calibration", {"framework": name}):
                    refusal_scores = evaluate_refusal_calibration(rag, judge=judge)
                eval_scores["refusal_calibration"] = refusal_scores
                print(f"  Refusal rate: {refusal_scores['refusal_rate']*100:.0f}% | "
                      f"Confident hallucination: {refusal_scores['confident_hallucination_rate']*100:.0f}%")

            summary[name] = {
                "build_time_s": round(build_time, 2),
                "latency": compute_latency_stats(results),
                "eval": eval_scores,
            }
            if name == "dspy":
                dspy_initial_results = results
            with open(RESULTS_DIR / f"results_{name}.json", "w") as f:
                json.dump({"results": results, "summary": summary[name]}, f, indent=2)

        ranking_table = generate_ranking_comparison(summary)
        summary["_ranking_table"] = ranking_table

        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print_summary(summary, ranking_table)

        # Step 3: Optionally run feedback loop on DSPy
        if args.feedback_loop and dspy_initial_results:
            with trace_span("feedback_loop", {"framework": "dspy"}):
                opt_results, opt_eval, comparison = run_feedback_loop(
                    frameworks["dspy"], documents, dspy_initial_results, test_pairs,
                    initial_eval=summary["dspy"]["eval"], judge=judge,
                )
            summary["dspy_optimized"] = {
                "build_time_s": summary["dspy"]["build_time_s"],
                "latency": compute_latency_stats(opt_results),
                "eval": opt_eval,
            }
            summary["_feedback_loop"] = comparison
            with open(RESULTS_DIR / "results_dspy_optimized.json", "w") as f:
                json.dump({"results": opt_results, "summary": summary["dspy_optimized"]}, f, indent=2)
            with open(RESULTS_DIR / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        # Step 4: Optionally run adversarial stress test
        if args.adversarial:
            print("\n" + "="*50)
            print("RUNNING ADVERSARIAL STRESS TEST")
            print("="*50)

            from src.evaluation.adversarial_agent import run_adversarial_benchmark,compute_degradation

            # ── Manual span: "adversarial_benchmark" ────────────────
            # Groups all adversarial work: query generation + running
            # all frameworks on hard queries + robustness evaluation.
            with trace_span("adversarial_benchmark", {"n_source_questions": 30}):
                adv_results = run_adversarial_benchmark(frameworks, adversarial_source,n_source_questions=30)
                degradation = compute_degradation(summary, adv_results)

            summary["_adversarial_degradation"] = degradation
            with open(RESULTS_DIR / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print("\n── ADVERSARIAL DEGRADATION (OOD excluded from ratio) ──")
            print(f"{'Framework':<15} {'Standard':<12} {'Non-OOD Rob.':<14} {'Ratio':<10} {'OOD Refusal':<14} {'Top Failure'}")
            print("-" * 85)
            for fw, stats in degradation.items():
                print(
                    f"{fw:<15} {stats['standard_correctness']:<12} "
                    f"{stats['non_ood_robustness']:<14} "
                    f"{stats['degradation_ratio']:<10} "
                    f"{stats['ood_refusal_rate']:<14} "
                    f"{stats['top_failure_mode']}"
                )
            print("\n→ Ratio < 1.0 = framework degrades under pressure (expected).")
            print("→ OOD Refusal Rate: higher = better at saying 'I don't know' (hallucination resistance).")

    # ── Flush remaining spans to Phoenix ────────────────────────────
    # This must happen OUTSIDE the benchmark_run span (after it ends)
    # so the root span itself gets flushed too.
    if tracing_shutdown:
        tracing_shutdown()

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()