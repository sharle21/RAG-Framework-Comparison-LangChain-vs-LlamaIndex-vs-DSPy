"""
adversarial_agent.py

An adversarial agent that stress-tests RAG frameworks with hard query variants.
Generates 4 types of adversarial queries from your existing test set:

1. Multi-hop: requires connecting information across documents
2. Ambiguous: underspecified, missing context
3. Out-of-distribution: questions about topics not in the corpus
4. Contradictory: challenges or inverts a claim in the corpus

Then runs all three frameworks on these hard queries and measures:
- Degradation rate (how much does quality drop vs standard queries?)
- Failure mode shift (do failures change type under pressure?)
- Robustness score (quality on adversarial / quality on standard)

Run: python src/evaluation/adversarial_agent.py
"""

import json
import re
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


# ─────────────────────────────────────────
# Adversarial query generation
# ─────────────────────────────────────────

ADVERSARIAL_PROMPT = """You are a red-teaming agent stress-testing a RAG system built on AI/ML research papers.

Given this original question and its answer, generate {n} adversarial variants.
Each variant should be a DIFFERENT type of hard query.

Original question: {question}
Original answer: {ground_truth}

Generate exactly these types (one each):
1. MULTI_HOP: Require connecting info from multiple documents. Add "across different papers" or "comparing multiple approaches".
2. AMBIGUOUS: Remove key context so the question is underspecified. Make it unclear what's being asked.
3. OUT_OF_DISTRIBUTION: Ask about something plausibly related but NOT in an AI/ML paper corpus (e.g., medical applications, legal implications, historical events).
4. CONTRADICTORY: Phrase the question as if the opposite of the ground truth is true. Challenge the answer.

Respond ONLY with a JSON array of objects:
[
  {{"type": "multi_hop", "question": "...", "expected_behavior": "should retrieve multiple docs and synthesize"}},
  {{"type": "ambiguous", "question": "...", "expected_behavior": "should ask for clarification or hedge"}},
  {{"type": "out_of_distribution", "question": "...", "expected_behavior": "should acknowledge it cannot answer"}},
  {{"type": "contradictory", "question": "...", "expected_behavior": "should maintain factual accuracy despite framing"}}
]"""


def generate_adversarial_queries(
    qa_pairs: list[dict],
    n_source_questions: int = 10,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    Generate adversarial variants from existing QA pairs.
    Uses n_source_questions to keep API costs low.
    """
    client = openai.OpenAI()
    adversarial_queries = []

    sample = qa_pairs[:n_source_questions]
    print(f"Generating adversarial queries from {len(sample)} source questions...")

    for i, qa in enumerate(sample):
        print(f"  Generating variants for question {i+1}/{len(sample)}...")
        prompt = ADVERSARIAL_PROMPT.format(
            n=4,
            question=qa["question"],
            ground_truth=qa["ground_truth"][:300],
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # some creativity for adversarial generation
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            variants = json.loads(raw)

            for variant in variants:
                variant["source_question"] = qa["question"]
                variant["source_ground_truth"] = qa["ground_truth"]
                adversarial_queries.append(variant)

        except Exception as e:
            print(f"  Error generating variants: {e}")

    print(f"\nGenerated {len(adversarial_queries)} adversarial queries")
    by_type = {}
    for q in adversarial_queries:
        t = q.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    for t, count in by_type.items():
        print(f"  {t}: {count}")

    return adversarial_queries


# ─────────────────────────────────────────
# Adversarial evaluation
# ─────────────────────────────────────────

ADVERSARIAL_JUDGE_PROMPT = """You are evaluating how well a RAG system handled a deliberately difficult query.

Query type: {query_type}
Expected behavior: {expected_behavior}
Question asked: {question}
System's answer: {answer}
Retrieved context: {context}

Score the system's response (0.0 to 1.0) on:
- robustness: Did the system handle the difficulty gracefully? (didn't hallucinate, didn't crash, acknowledged uncertainty when appropriate)
- appropriate_behavior: Did the system behave as expected for this query type?

Also classify the failure mode if robustness < 0.7:
- confident_hallucination: answered confidently with wrong/fabricated info
- missed_uncertainty: should have hedged but didn't
- retrieval_confusion: retrieved irrelevant docs and used them anyway  
- correct_refusal: appropriately said it couldn't answer (this is GOOD for OOD queries)
- good_response: handled it well

Respond ONLY with JSON:
{{"robustness": 0.7, "appropriate_behavior": 0.8, "failure_mode": "...", "reasoning": "one sentence"}}"""


def evaluate_adversarial_results(results: list[dict], model: str = "gpt-4o-mini") -> dict:
    """Evaluate how each framework handles adversarial queries."""
    client = openai.OpenAI()

    scores_by_type = {
        "multi_hop": [],
        "ambiguous": [],
        "out_of_distribution": [],
        "contradictory": [],
    }
    failure_modes = {}
    detailed = []

    for r in results:
        if not r.get("answer"):
            continue

        query_type = r.get("query_type", "unknown")
        expected = r.get("expected_behavior", "")
        context_str = "\n".join(r.get("contexts", []))[:1500]

        prompt = ADVERSARIAL_JUDGE_PROMPT.format(
            query_type=query_type,
            expected_behavior=expected,
            question=r["question"],
            answer=r["answer"],
            context=context_str,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)

            robustness = float(parsed.get("robustness", 0))
            if query_type in scores_by_type:
                scores_by_type[query_type].append(robustness)

            fm = parsed.get("failure_mode", "unknown")
            failure_modes[fm] = failure_modes.get(fm, 0) + 1

            detailed.append({
                "question": r["question"][:80],
                "query_type": query_type,
                "robustness": robustness,
                "failure_mode": fm,
                "reasoning": parsed.get("reasoning", ""),
            })

        except Exception as e:
            print(f"  Eval error: {e}")

    # Compute per-type averages
    avg_by_type = {
        t: round(sum(scores) / len(scores), 3) if scores else 0.0
        for t, scores in scores_by_type.items()
    }

    all_scores = [s for scores in scores_by_type.values() for s in scores]
    overall = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

    return {
        "overall_robustness": overall,
        "robustness_by_query_type": avg_by_type,
        "failure_modes": failure_modes,
        "detailed": detailed,
    }


# ─────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────

def run_adversarial_benchmark(frameworks: dict, qa_pairs: list[dict],n_source_questions: int = 10) -> dict:
    """
    Full adversarial benchmark:
    1. Generate hard queries
    2. Run all frameworks on them
    3. Evaluate robustness
    4. Compute degradation vs standard benchmark
    """
    # Generate adversarial queries (saves to file for reproducibility)
    adv_path = DATA_DIR / "adversarial_queries.json"
    if adv_path.exists():
        print("Loading existing adversarial queries...")
        with open(adv_path) as f:
            adversarial_queries = json.load(f)
    else:
        adversarial_queries = generate_adversarial_queries(qa_pairs, n_source_questions=n_source_questions)
        with open(adv_path, "w") as f:
            json.dump(adversarial_queries, f, indent=2)
        print(f"Saved adversarial queries to {adv_path}")

    results = {}

    for name, rag in frameworks.items():
        print(f"\nRunning adversarial queries on {name}...")
        framework_results = []

        for q in adversarial_queries:
            try:
                result = rag.query(q["question"])
                result["query_type"] = q["type"]
                result["expected_behavior"] = q["expected_behavior"]
                result["question"] = q["question"]
                result["ground_truth"] = q.get("source_ground_truth", "")
                framework_results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                framework_results.append({
                    "question": q["question"],
                    "query_type": q["type"],
                    "expected_behavior": q["expected_behavior"],
                    "answer": "",
                    "contexts": [],
                    "latency_ms": -1,
                    "error": str(e),
                })

        print(f"Evaluating {name} adversarial results...")
        eval_scores = evaluate_adversarial_results(framework_results)
        results[name] = {
            "raw_results": framework_results,
            "eval": eval_scores,
        }

        # Save incrementally
        with open(RESULTS_DIR / f"adversarial_{name}.json", "w") as f:
            json.dump(results[name], f, indent=2)

    return results


def compute_degradation(standard_summary: dict, adversarial_results: dict) -> dict:
    """
    The key insight metric: how much does each framework degrade under adversarial conditions?
    Robustness score = adversarial performance / standard performance
    Closer to 1.0 = more robust. Below 0.7 = significant degradation.
    """
    degradation = {}
    for framework in adversarial_results:
        standard_score = standard_summary.get(framework, {}).get(
            "eval", {}
        ).get("llm_judge", {}).get("correctness", 0)

        adversarial_score = adversarial_results[framework]["eval"]["overall_robustness"]

        degradation[framework] = {
            "standard_correctness": round(standard_score, 3),
            "adversarial_robustness": round(adversarial_score, 3),
            "degradation_ratio": round(adversarial_score / standard_score, 3) if standard_score > 0 else 0,
            "top_failure_mode": max(
                adversarial_results[framework]["eval"]["failure_modes"],
                key=adversarial_results[framework]["eval"]["failure_modes"].get,
                default="unknown"
            ),
        }

    return degradation


if __name__ == "__main__":
    # Standalone run — loads existing framework results and runs adversarial eval
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    with open(DATA_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    from src.langchain_rag.pipeline import LangChainRAG
    from src.llamaindex_rag.pipeline import LlamaIndexRAG
    from src.dspy_rag.pipeline import DSPyRAG

    # Load existing docs
    with open(DATA_DIR / "raw" / "ragbench_documents.json") as f:
        documents = json.load(f)

    frameworks = {
        "langchain": LangChainRAG(),
        "llamaindex": LlamaIndexRAG(),
        "dspy": DSPyRAG(),
    }

    # Build indexes (needed before querying)
    for name, rag in frameworks.items():
        print(f"Building {name} index...")
        rag.build(documents)

    adv_results = run_adversarial_benchmark(frameworks, qa_pairs)

    # Load standard results for degradation comparison
    try:
        with open(RESULTS_DIR / "summary.json") as f:
            standard_summary = json.load(f)
        degradation = compute_degradation(standard_summary, adv_results)

        print("\n\n" + "="*60)
        print("ADVERSARIAL DEGRADATION RESULTS")
        print("="*60)
        print(f"{'Framework':<15} {'Standard':<12} {'Adversarial':<14} {'Ratio':<10} {'Top Failure'}")
        print("-" * 65)
        for fw, stats in degradation.items():
            print(
                f"{fw:<15} {stats['standard_correctness']:<12} "
                f"{stats['adversarial_robustness']:<14} "
                f"{stats['degradation_ratio']:<10} "
                f"{stats['top_failure_mode']}"
            )

        with open(RESULTS_DIR / "adversarial_summary.json", "w") as f:
            json.dump({"results": adv_results, "degradation": degradation}, f, indent=2)

    except FileNotFoundError:
        print("Run run_benchmark.py first to get standard results.")