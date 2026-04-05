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
import statistics
from typing import Any

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

# Cross-family judge — lazy-initialized so we don't pay API init cost if overridden
_default_judge = None


def _get_default_judge():
    global _default_judge
    if _default_judge is None:
        _default_judge = ChatMistralAI(model="mistral-large-latest", temperature=0)
    return _default_judge


def make_vllm_judge(base_url: str, model: str = "Qwen/Qwen2.5-14B-Instruct"):
    """
    OpenAI-compatible judge pointing at a vLLM endpoint.
    Use this instead of Mistral when running on Lambda with local models.
    Different model family from Llama worker = no same-family bias.
    """
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

    return {
        "answer_f1": sum(answer_f1s) / len(answer_f1s) if answer_f1s else 0.0,
        "context_coverage": sum(ctx_coverages) / len(ctx_coverages) if ctx_coverages else 0.0,
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

Respond ONLY with a JSON object like:
{{"correctness": 0.8, "faithfulness": 0.9, "completeness": 0.7, "reasoning": "brief explanation"}}"""


def evaluate_llm_judge(results: list[dict], n_runs: int = 3, judge=None) -> dict:
    """
    Custom LLM-as-judge. Defaults to Mistral Large (cross-family vs GPT-4o-mini worker).
    Pass judge=make_vllm_judge(...) to use local Qwen on vLLM instead.

    Runs the judge n_runs times per question and averages the scores.
    Reports std dev to flag high-variance / unreliable judgements.
    """
    if judge is None:
        judge = _get_default_judge()

    judge_model_name = getattr(judge, "model_name", None) or getattr(judge, "model", "unknown")
    dims = ["correctness", "faithfulness", "completeness"]
    per_question_means = {d: [] for d in dims}
    per_question_stds = {d: [] for d in dims}
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
        for _ in range(n_runs):
            try:
                response = judge.invoke(prompt)
                raw = response.content.strip()
                raw = re.sub(r"```json|```", "", raw).strip()
                parsed = json.loads(raw, strict=False)
                for dim in dims:
                    if dim in parsed:
                        run_scores[dim].append(float(parsed[dim]))
            except Exception as e:
                errors += 1
                print(f"  Judge error: {e}")

        for dim in dims:
            vals = run_scores[dim]
            if vals:
                per_question_means[dim].append(sum(vals) / len(vals))
                if len(vals) > 1:
                    per_question_stds[dim].append(statistics.stdev(vals))

    result = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in per_question_means.items()}
    for dim in dims:
        std_vals = per_question_stds[dim]
        result[f"{dim}_std"] = round(sum(std_vals) / len(std_vals), 4) if std_vals else 0.0
    result["errors"] = errors
    result["judge_model"] = judge_model_name
    result["n_runs"] = n_runs
    return result


# ─────────────────────────────────────────
# 3. RAGAS
# ─────────────────────────────────────────

def evaluate_ragas(results: list[dict]) -> dict:
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
    "wrong_context": "Right question, wrong documents retrieved",
    "hallucination": "Right context retrieved but answer not grounded in it",
    "too_vague": "Answer is technically correct but too generic to be useful",
    "incomplete": "Answer misses key points present in ground truth",
    "correct": "Answer is good",
}

FAILURE_CLASSIFIER_PROMPT = """You are analyzing why an AI RAG system answer is wrong or suboptimal.
The answer was produced by a different AI system — you are an independent analyst.

Question: {question}
Ground truth: {ground_truth}
Generated answer: {answer}
Retrieved context: {context}

Classify this answer into EXACTLY ONE failure category:
- wrong_context: The retrieved context didn't contain relevant information
- hallucination: The answer contains claims not supported by the retrieved context
- too_vague: The answer is too generic or non-committal to be useful
- incomplete: The answer is partially correct but misses important information
- correct: The answer is good quality

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
            categories["wrong_context"] += 1
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
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw,strict=False)
            cat = parsed.get("category", "incomplete")
            if cat not in categories:
                cat = "incomplete"
            categories[cat] += 1
            detailed.append({
                "question": r["question"][:80],
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