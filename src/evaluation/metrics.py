"""
evaluation/metrics.py

Three evaluation approaches run on the same results:
1. String overlap (no LLM — fast reproducible baseline)
2. Custom LLM-as-judge (Mistral Large — cross-family judge)
3. RAGAS (standard RAG evaluation suite)

Judge model: mistral-large-latest
  Worker model is gpt-4o-mini (OpenAI family).
  Using Mistral as judge avoids same-family self-evaluation bias —
  different training lineage, different RLHF, different company.
  RAGAS stays on OpenAI because it's tightly coupled to the OpenAI SDK.

The key research question: does the ranking of frameworks change
depending on which metric you use?
"""

import re
import json
import random
import statistics
from typing import Any

RANDOM_SEED = 42

# ─────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────

def compute_bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(scores)
    if n < 2:
        v = scores[0] if scores else 0.0
        return v, v
    means = sorted(
        sum(rng.choices(scores, k=n)) / n for _ in range(n_bootstrap)
    )
    lo = int((1 - ci) / 2 * n_bootstrap)
    hi = int((1 + ci) / 2 * n_bootstrap)
    return means[lo], means[hi]


def compute_stats(scores: list[float]) -> dict:
    """Mean, std, n, analytical 95% CI, bootstrap 95% CI."""
    n = len(scores)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0,
                "ci95_lower": 0.0, "ci95_upper": 0.0,
                "boot_ci95_lower": 0.0, "boot_ci95_upper": 0.0}
    mean = sum(scores) / n
    std = statistics.stdev(scores) if n > 1 else 0.0
    margin = 1.96 * std / (n ** 0.5)
    boot_lo, boot_hi = compute_bootstrap_ci(scores)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "n": n,
        "ci95_lower": round(mean - margin, 4),
        "ci95_upper": round(mean + margin, 4),
        "boot_ci95_lower": round(boot_lo, 4),
        "boot_ci95_upper": round(boot_hi, 4),
    }


# Cross-family judge — lazy-initialized so we don't pay API init cost if overridden
_default_judge = None


def _get_default_judge():
    from langchain_mistralai import ChatMistralAI
    global _default_judge
    if _default_judge is None:
        _default_judge = ChatMistralAI(model="mistral-large-latest", temperature=0)
    return _default_judge


def make_vllm_judge(base_url: str, model: str = "Qwen/Qwen3-14B"):
    """
    OpenAI-compatible judge pointing at a vLLM endpoint.
    Use this instead of Mistral when running on Lambda with local models.
    Different model family from Llama worker = no same-family bias.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, base_url=base_url, api_key="none", temperature=0)


# ─────────────────────────────────────────
# 1. String overlap (no LLM baseline)
# ─────────────────────────────────────────

def tokenize(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return set(text.split())


def f1_score(pred: str, ground_truth: str) -> float:
    pred_tokens = tokenize(pred)
    gt_tokens = tokenize(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def context_coverage(contexts: list[str], ground_truth: str) -> float:
    """What fraction of ground truth tokens appear in retrieved contexts?"""
    all_context = " ".join(contexts)
    gt_tokens = tokenize(ground_truth)
    ctx_tokens = tokenize(all_context)
    if not gt_tokens:
        return 0.0
    return len(gt_tokens & ctx_tokens) / len(gt_tokens)


def evaluate_string_overlap(results: list[dict]) -> dict:
    """Baseline evaluation using token F1 and context coverage. No LLM."""
    answer_f1s = []
    ctx_coverages = []

    for r in results:
        if not r.get("answer"):
            continue
        answer_f1s.append(f1_score(r["answer"], r["ground_truth"]))
        ctx_coverages.append(context_coverage(r.get("contexts", []), r["ground_truth"]))

    f1_stats = compute_stats(answer_f1s)
    ctx_stats = compute_stats(ctx_coverages)
    return {
        "answer_f1": f1_stats["mean"],
        "answer_f1_std": f1_stats["std"],
        "answer_f1_n": f1_stats["n"],
        "answer_f1_ci95": [f1_stats["ci95_lower"], f1_stats["ci95_upper"]],
        "answer_f1_boot_ci95": [f1_stats["boot_ci95_lower"], f1_stats["boot_ci95_upper"]],
        "context_coverage": ctx_stats["mean"],
        "context_coverage_ci95": [ctx_stats["ci95_lower"], ctx_stats["ci95_upper"]],
    }


# ─────────────────────────────────────────
# 1b. BERTScore (semantic similarity)
# ─────────────────────────────────────────

def evaluate_bertscore(results: list[dict]) -> dict:
    """
    Semantic similarity using BERTScore.
    Unlike string F1, understands that 'Q4' and 'fourth quarter' are the same.
    Uses distilbert-base-uncased for speed; scores are precision/recall/F1.
    """
    from bert_score import score as bert_score_fn

    valid = [(r["answer"], r["ground_truth"]) for r in results if r.get("answer")]
    if not valid:
        return {}

    predictions, references = zip(*valid)
    try:
        P, R, F1 = bert_score_fn(
            list(predictions), list(references),
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        return {
            "bertscore_precision": round(P.mean().item(), 4),
            "bertscore_recall": round(R.mean().item(), 4),
            "bertscore_f1": round(F1.mean().item(), 4),
        }
    except Exception as e:
        print(f"  BERTScore error: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────
# 2. Custom LLM-as-judge (Mistral Large)
# ─────────────────────────────────────────

LLM_JUDGE_PROMPT = """You are evaluating the quality of an AI-generated answer.
The answer was produced by a different AI system — you are an independent judge.

Question: {question}
Ground truth answer: {ground_truth}
Generated answer: {answer}
Retrieved context: {context}

Score the generated answer on these dimensions (each 0.0 to 1.0):
- correctness: Is the answer factually correct compared to ground truth?
- faithfulness: Is the answer grounded in the retrieved context (no hallucination)?
- completeness: Does the answer cover the key points from the ground truth?
- confidence: How confident are you in your correctness score? (0.0=very uncertain, 1.0=very certain)

Respond ONLY with a JSON object like:
{{"correctness": 0.8, "faithfulness": 0.9, "completeness": 0.7, "confidence": 0.85, "reasoning": "brief explanation"}}"""


def compute_ece(confidences: list[float], correctness: list[float], n_bins: int = 10) -> float:
    """Expected Calibration Error — how well judge confidence predicts correctness."""
    if not confidences or len(confidences) != len(correctness):
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for conf, corr in zip(confidences, correctness):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, corr))
    n = len(confidences)
    ece = 0.0
    for bin_items in bins:
        if not bin_items:
            continue
        mean_conf = sum(c for c, _ in bin_items) / len(bin_items)
        mean_corr = sum(r for _, r in bin_items) / len(bin_items)
        ece += (len(bin_items) / n) * abs(mean_conf - mean_corr)
    return round(ece, 4)


def evaluate_llm_judge(results: list[dict], n_runs: int = 3, judge=None) -> dict:
    """
    Custom LLM-as-judge. Defaults to Mistral Large (cross-family vs GPT-4o-mini worker).
    Pass judge=make_vllm_judge(...) to use local Qwen on vLLM instead.

    Runs the judge n_runs times per question and averages the scores.
    Reports std dev to flag high-variance / unreliable judgements.
    Also captures judge confidence for ECE (Expected Calibration Error) computation.
    """
    if judge is None:
        judge = _get_default_judge()

    judge_model_name = getattr(judge, "model_name", None) or getattr(judge, "model", "unknown")
    dims = ["correctness", "faithfulness", "completeness"]
    per_question_means = {d: [] for d in dims}
    per_question_stds = {d: [] for d in dims}
    per_question_scores = []  # one entry per question, used for conflict detection
    confidence_scores = []    # per-question mean confidence, for ECE
    errors = 0

    for r in results:
        if not r.get("answer"):
            errors += 1
            continue

        context_str = "\n".join(r.get("contexts", []))[:2000]
        prompt = LLM_JUDGE_PROMPT.format(
            question=r["question"],
            ground_truth=r["ground_truth"][:500],
            answer=r["answer"],
            context=context_str,
        )

        run_scores = {d: [] for d in dims}
        run_confidences = []
        for _ in range(n_runs):
            try:
                response = judge.invoke(prompt)
                raw = response.content.strip()
                # Strip Qwen3 <think>...</think> blocks before parsing
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                raw = re.sub(r"```json|```", "", raw).strip()
                # Extract first JSON object if there's surrounding text
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    raw = m.group(0)
                parsed = json.loads(raw, strict=False)
                for dim in dims:
                    if dim in parsed:
                        run_scores[dim].append(float(parsed[dim]))
                if "confidence" in parsed:
                    run_confidences.append(float(parsed["confidence"]))
            except Exception as e:
                errors += 1
                print(f"  Judge error: {e}")

        q_scores = {"question": r["question"]}
        for dim in dims:
            vals = run_scores[dim]
            if vals:
                mean_val = sum(vals) / len(vals)
                per_question_means[dim].append(mean_val)
                q_scores[dim] = round(mean_val, 4)
                if len(vals) > 1:
                    per_question_stds[dim].append(statistics.stdev(vals))
            else:
                q_scores[dim] = 0.0
        if run_confidences:
            q_scores["confidence"] = round(sum(run_confidences) / len(run_confidences), 4)
            confidence_scores.append(q_scores["confidence"])
        per_question_scores.append(q_scores)

    result = {}
    for dim in dims:
        s = compute_stats(per_question_means[dim])
        result[dim] = s["mean"]
        result[f"{dim}_std"] = s["std"]
        result[f"{dim}_ci95"] = [s["ci95_lower"], s["ci95_upper"]]
        result[f"{dim}_boot_ci95"] = [s["boot_ci95_lower"], s["boot_ci95_upper"]]
        result[f"{dim}_n"] = s["n"]

    # ECE: compare judge confidence to actual correctness scores
    if confidence_scores and per_question_means["correctness"]:
        result["ece"] = compute_ece(confidence_scores, per_question_means["correctness"])
        result["mean_confidence"] = round(sum(confidence_scores) / len(confidence_scores), 4)

    result["errors"] = errors
    result["judge_model"] = judge_model_name
    result["n_runs"] = n_runs
    result["per_question"] = per_question_scores
    return result


# ─────────────────────────────────────────
# 3. RAGAS
# ─────────────────────────────────────────

def evaluate_ragas(results: list[dict]) -> dict:
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings

    # 1. Clean data to ensure no weird characters remain
    valid = []
    for r in results:
        if r.get("answer") and r.get("contexts"):
            # RAGAS 0.4.x can choke on literal newlines in its internal JSON prompts
            clean_answer = r["answer"].replace("\n", " ").replace("\t", " ").strip()
            clean_contexts = [c.replace("\n", " ").replace("\t", " ").strip() for c in r["contexts"]]
            valid.append({
                "question": r["question"],
                "answer": clean_answer,
                "ground_truth": r["ground_truth"],
                "contexts": clean_contexts
            })

    if not valid:
        return {"error": "No valid results"}

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            reference=r["ground_truth"],
            retrieved_contexts=r["contexts"],
        )
        for r in valid
    ]
    dataset = EvaluationDataset(samples=samples)

    # 2. Setup LLM and Embeddings
    # Using a higher temperature (0.1) can sometimes help the LLM 
    # follow the RAGAS formatting instructions better than 0.0
    base_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    ragas_llm = LangchainLLMWrapper(base_model)
    
    # CRITICAL: RAGAS 0.4.x internal batching fix
    ragas_llm.bypass_n = True 

    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    try:
        # 3. The Execution Fix
        # We set run_config to be extremely conservative
        from ragas.run_config import RunConfig
        
        config = RunConfig(
            timeout=60,
            max_retries=3,
            max_wait=60,
            max_workers=1 # FORCE serial execution to prevent the "grouped results" bug
        )

        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=config  # Use the restricted config
        )
        
        # Convert to dict safely
        df = scores.to_pandas()
        numeric_cols = df.select_dtypes(include=['number'])
        return numeric_cols.mean().to_dict()

    except Exception as e:
        print(f"  RAGAS evaluation error: {e}")
        # If it fails, try to return whatever was captured
        return {"error": str(e)}


# ─────────────────────────────────────────
# 4. Failure mode analysis (Mistral judge)
# ─────────────────────────────────────────

FAILURE_CATEGORIES = {
    "unsupported_claim": "Answer contains a claim not present in retrieved context",
    "wrong_retrieval": "Retrieved documents are off-topic or irrelevant to the question",
    "no_retrieval": "No useful context was retrieved; answer is pure hallucination",
    "prompt_failure": "Model misread or ignored the question",
    "reasoning_failure": "Context was relevant but model reasoning was incorrect",
    "formatting_failure": "Answer format is wrong (wrong units, wrong structure, truncated)",
    "partial_answer": "Correct but misses key information from ground truth",
    "missing_citation": "Answer is plausible but not grounded in any retrieved passage",
}

FAILURE_CLASSIFIER_PROMPT = """You are analyzing why an AI RAG system answer is wrong or suboptimal.
The answer was produced by a different AI system — you are an independent analyst.

Question: {question}
Ground truth: {ground_truth}
Generated answer: {answer}
Retrieved context: {context}

Classify this answer into EXACTLY ONE failure category:
- unsupported_claim: Answer asserts something not in the retrieved context
- wrong_retrieval: Retrieved documents are off-topic or don't address the question
- no_retrieval: No useful context retrieved; answer appears entirely fabricated
- prompt_failure: Model misread, ignored, or answered a different question
- reasoning_failure: Context was relevant but the reasoning chain was wrong
- formatting_failure: Answer is wrong format, wrong units, or truncated mid-thought
- partial_answer: Answer is correct but misses key points from the ground truth
- missing_citation: Answer is plausible-sounding but has no grounding in retrieved passages

Note: use "partial_answer" if the answer is mostly correct. Use "unsupported_claim" only if there's
an active wrong claim (not just missing info). Use "correct" if the answer is good (add as fallback).

Respond ONLY with a JSON object:
{{"category": "<category>", "reasoning": "<one sentence>"}}"""


def analyze_failure_modes(results: list[dict], sample_n: int = 15, judge=None) -> dict:
    """
    Classify a sample of answers into failure categories.
    Defaults to Mistral Large; pass judge=make_vllm_judge(...) for local Qwen.
    """
    if judge is None:
        judge = _get_default_judge()

    judge_model_name = getattr(judge, "model_name", None) or getattr(judge, "model", "unknown")
    sample = results[:sample_n]
    categories = {cat: 0 for cat in FAILURE_CATEGORIES}
    detailed = []

    for r in sample:
        if not r.get("answer"):
            categories["no_retrieval"] += 1
            continue

        context_str = "\n".join(r.get("contexts", []))[:1500]
        prompt = FAILURE_CLASSIFIER_PROMPT.format(
            question=r["question"],
            ground_truth=r["ground_truth"][:400],
            answer=r["answer"],
            context=context_str,
        )

        try:
            response = judge.invoke(prompt)
            raw = response.content.strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            parsed = json.loads(raw, strict=False)
            cat = parsed.get("category", "partial_answer")
            if cat not in categories:
                cat = "partial_answer"
            categories[cat] += 1
            detailed.append({
                "question": r["question"],  # full question for conflict detection
                "category": cat,
                "reasoning": parsed.get("reasoning", ""),
            })
        except Exception as e:
            print(f"  Failure classifier error: {e}")

    total = sum(categories.values())
    percentages = {
        k: round(v / total * 100, 1) if total else 0
        for k, v in categories.items()
    }

    return {
        "counts": categories,
        "percentages": percentages,
        "sample_size": total,
        "detailed": detailed,
        "judge_model": judge_model_name,
    }


# ─────────────────────────────────────────
# 5. Conflict detection (Item 4)
# ─────────────────────────────────────────

def detect_conflicts(
    judge_per_question: list[dict],
    failure_detailed: list[dict],
    correctness_threshold: float = 0.7,
) -> list[dict]:
    """
    Find answers where the LLM judge scored correctness >= threshold
    BUT the failure classifier labelled them as hallucination.

    These are suspicious: the judge is rewarding a fluent, confident answer
    even though the failure classifier caught that the answer isn't grounded
    in the retrieved context. Known as judge sycophancy.

    Returns a list of flagged questions with their scores and reasoning.
    """
    # Build a lookup: full question text → judge scores for that question
    judge_by_question = {item["question"]: item for item in judge_per_question}

    conflicts = []
    for entry in failure_detailed:
        if entry["category"] != "hallucination":
            continue
        q = entry["question"]
        if q not in judge_by_question:
            continue
        correctness = judge_by_question[q].get("correctness", 0.0)
        if correctness >= correctness_threshold:
            conflicts.append({
                "question": q[:120],
                "judge_correctness": correctness,
                "failure_reasoning": entry["reasoning"],
            })

    return conflicts


# ─────────────────────────────────────────
# 6. Refusal calibration
# ─────────────────────────────────────────
#
# Why: LLMs are RLHF-trained to be helpful, so "I don't know" is penalized
# during training. This makes confident hallucination the default behavior
# on unanswerable questions. (See: "Why Do Language Models Hallucinate?")
#
# A well-calibrated RAG system should REFUSE to answer when the retrieved
# context doesn't contain the answer, rather than guessing confidently.

# Regex patterns that indicate the model is refusing / hedging
REFUSAL_PATTERNS = [
    r"(?i)i don'?t know",
    r"(?i)i cannot (determine|answer|find|provide)",
    r"(?i)not enough information",
    r"(?i)no (relevant |sufficient )?information (is )?(available|provided|found)",
    r"(?i)the (context|documents?|passages?) (does|do) not (contain|mention|provide|include)",
    r"(?i)cannot be (determined|answered|found)",
    r"(?i)there is no (information|evidence|mention)",
    r"(?i)based on the (provided |given )?(context|information), .{0,30}(not|no|cannot|unable)",
    r"(?i)unable to (answer|determine|find|provide)",
    r"(?i)this question cannot be answered",
    r"(?i)outside the scope",
    r"(?i)not (mentioned|discussed|covered|addressed) in",
]

# Questions that LOOK domain-relevant but have no answer in the RAGBench corpus.
# A well-calibrated system should refuse these; a sycophantic one will hallucinate.
UNANSWERABLE_QUESTIONS = [
    # finqa — plausible financial questions not in corpus
    {"question": "What was the total revenue impact of the 2024 Basel IV implementation on US regional banks?",
     "domain": "finqa", "expected": "refuse"},
    {"question": "How does the SOFR transition affect legacy LIBOR-based swap contracts under ISDA 2021 fallback protocol?",
     "domain": "finqa", "expected": "refuse"},
    {"question": "What percentage of Fortune 500 companies adopted real-time treasury management systems by Q3 2024?",
     "domain": "finqa", "expected": "refuse"},
    {"question": "What is the average cost basis adjustment for SPAC de-SPAC transactions under ASC 805?",
     "domain": "finqa", "expected": "refuse"},
    {"question": "How did the EU Carbon Border Adjustment Mechanism affect steel industry forward contracts in 2024?",
     "domain": "finqa", "expected": "refuse"},
    # techqa — plausible tech questions not in corpus
    {"question": "What is the recommended kernel parameter tuning for IBM WebSphere on RHEL 9 with SELinux enforcing?",
     "domain": "techqa", "expected": "refuse"},
    {"question": "How do you configure IBM MQ Advanced Message Security with a PKCS#11 hardware security module?",
     "domain": "techqa", "expected": "refuse"},
    {"question": "What is the maximum supported cluster size for IBM Db2 pureScale on PowerVM with GPFS 5.1?",
     "domain": "techqa", "expected": "refuse"},
    {"question": "How does the IBM Z16 Telum AI accelerator integrate with CICS Transaction Server for real-time fraud scoring?",
     "domain": "techqa", "expected": "refuse"},
    {"question": "What are the known compatibility issues between IBM DataPower Gateway and Kubernetes Istio service mesh?",
     "domain": "techqa", "expected": "refuse"},
    # covidqa — plausible medical questions not in corpus
    {"question": "What is the mechanism by which the XBB.1.16 Arcturus variant evades bivalent booster immunity?",
     "domain": "covidqa", "expected": "refuse"},
    {"question": "How does long COVID affect mitochondrial function in cardiac tissue based on 2024 autopsy studies?",
     "domain": "covidqa", "expected": "refuse"},
    {"question": "What is the recommended antiviral protocol for immunocompromised patients with JN.1 variant breakthrough infections?",
     "domain": "covidqa", "expected": "refuse"},
    {"question": "How does the updated WHO severity classification for COVID-19 pneumonia differ from the original 2020 criteria?",
     "domain": "covidqa", "expected": "refuse"},
    {"question": "What percentage of healthcare workers in the RECOVER-VITAL trial showed persistent T-cell exhaustion after third booster?",
     "domain": "covidqa", "expected": "refuse"},
]


def detect_refusal(answer: str) -> bool:
    """Check if an answer contains refusal/hedging patterns."""
    if not answer:
        return True  # empty answer = implicit refusal
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, answer):
            return True
    return False


def evaluate_refusal_calibration(rag_instance, judge=None) -> dict:
    """
    Run unanswerable questions through a RAG pipeline and measure refusal rate.

    A well-calibrated system should refuse (say "I don't know") on questions
    where the corpus has no answer. A sycophantic system will hallucinate.

    Returns refusal rate (higher = better calibrated) and per-question details.
    """
    if judge is None:
        judge = _get_default_judge()

    results = []
    for q in UNANSWERABLE_QUESTIONS:
        try:
            result = rag_instance.query(q["question"])
            answer = result.get("answer", "")
        except Exception as e:
            answer = ""

        refused = detect_refusal(answer)

        # Also ask the judge if the answer is a hallucination
        is_hallucination = False
        if not refused and answer:
            try:
                prompt = f"""An AI was asked a question that CANNOT be answered from its knowledge base.
Question: {q['question']}
AI's answer: {answer}

Did the AI hallucinate (make up an answer) or did it correctly refuse/hedge?
Respond ONLY with JSON: {{"hallucinated": true/false, "reasoning": "brief explanation"}}"""
                response = judge.invoke(prompt)
                raw = response.content.strip()
                raw = re.sub(r"```json|```", "", raw).strip()
                parsed = json.loads(raw, strict=False)
                is_hallucination = parsed.get("hallucinated", False)
            except Exception:
                pass

        results.append({
            "question": q["question"],
            "domain": q["domain"],
            "answer": answer[:200],
            "refused": refused,
            "hallucinated": is_hallucination,
        })

    total = len(results)
    refused_count = sum(1 for r in results if r["refused"])
    hallucinated_count = sum(1 for r in results if r["hallucinated"])
    confident_wrong = sum(1 for r in results if not r["refused"] and r["hallucinated"])

    return {
        "refusal_rate": round(refused_count / total, 3) if total else 0,
        "hallucination_rate": round(hallucinated_count / total, 3) if total else 0,
        "confident_hallucination_rate": round(confident_wrong / total, 3) if total else 0,
        "refused": refused_count,
        "hallucinated": hallucinated_count,
        "total": total,
        "details": results,
    }