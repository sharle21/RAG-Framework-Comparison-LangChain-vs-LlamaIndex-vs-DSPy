"""
Approximate retrieval-recall signal without stored doc IDs (CPU-only, no GPU).

For each query, checks whether the framework's retrieved `contexts` text
overlaps with the ground-truth `relevant_doc_ids` passages from RAGBench.
Since only passage text was persisted in go_results (not doc IDs), matching
is done via normalized word-overlap containment rather than exact ID lookup.

This is NOT Recall@k/NDCG (no ranking, no ID-level ground truth match) —
it reports a "reference-context overlap rate": the proportion of queries
where at least one retrieved passage substantially overlaps a relevant
source document.

Usage:
    python run_retrieval_overlap.py results/go_results_20260408_013644.json
"""

import json
import re
import sys
from collections import defaultdict

QA_PAIRS_PATH = "data/qa_pairs.json"
DOCUMENTS_PATH = "data/raw/ragbench_documents.json"
OVERLAP_THRESHOLD = 0.6  # match if EITHER direction's containment ratio clears this
# (retrieved chunk mostly inside the relevant doc, or vice versa — needed because
# frameworks chunk very differently: LangChain's techqa chunks average ~745 chars
# vs LlamaIndex/DSPy's ~2900-3250 chars, so a one-directional "doc mostly in chunk"
# test unfairly penalizes small-chunk frameworks even on correct retrievals)


def normalize_words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap_ratio(doc_words: set, context_words: set) -> float:
    if not doc_words or not context_words:
        return 0.0
    shared = len(doc_words & context_words)
    forward = shared / len(doc_words)       # how much of the doc is covered by the chunk
    backward = shared / len(context_words)  # how much of the chunk is covered by the doc
    return max(forward, backward)


def load_documents(path: str) -> dict:
    with open(path) as f:
        docs = json.load(f)
    return {d["id"]: d["content"] for d in docs}


def load_relevant_doc_ids(path: str) -> dict:
    with open(path) as f:
        qa_pairs = json.load(f)
    return {qa["question"]: qa.get("relevant_doc_ids", []) for qa in qa_pairs}


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <go_results.json>")
        sys.exit(1)

    results_path = sys.argv[1]

    with open(results_path) as f:
        results = json.load(f)

    documents = load_documents(DOCUMENTS_PATH)
    relevant_by_question = load_relevant_doc_ids(QA_PAIRS_PATH)

    hits = defaultdict(int)
    totals = defaultdict(int)
    hits_by_domain = defaultdict(lambda: defaultdict(int))
    totals_by_domain = defaultdict(lambda: defaultdict(int))
    no_ground_truth = 0

    for item in results:
        framework = item["framework"]
        domain = item["domain"]
        question = item["question"]
        contexts = item.get("contexts") or []

        relevant_ids = relevant_by_question.get(question, [])
        if not relevant_ids:
            no_ground_truth += 1
            continue

        relevant_word_sets = [
            normalize_words(documents[doc_id])
            for doc_id in relevant_ids
            if doc_id in documents
        ]
        if not relevant_word_sets:
            no_ground_truth += 1
            continue

        context_word_sets = [normalize_words(c) for c in contexts]

        matched = any(
            overlap_ratio(doc_words, ctx_words) >= OVERLAP_THRESHOLD
            for doc_words in relevant_word_sets
            for ctx_words in context_word_sets
        )

        totals[framework] += 1
        totals_by_domain[framework][domain] += 1
        if matched:
            hits[framework] += 1
            hits_by_domain[framework][domain] += 1

    print(f"Reference-context overlap rate (threshold={OVERLAP_THRESHOLD})")
    print(f"Queries skipped (no ground-truth doc IDs found): {no_ground_truth}\n")

    print("Overall by framework:")
    for framework in sorted(totals):
        rate = hits[framework] / totals[framework] if totals[framework] else 0.0
        print(f"  {framework:12s} {rate:.3f}  ({hits[framework]}/{totals[framework]})")

    print("\nPer-domain:")
    for framework in sorted(totals_by_domain):
        print(f"  {framework}:")
        for domain in sorted(totals_by_domain[framework]):
            t = totals_by_domain[framework][domain]
            h = hits_by_domain[framework][domain]
            rate = h / t if t else 0.0
            print(f"    {domain:10s} {rate:.3f}  ({h}/{t})")


if __name__ == "__main__":
    main()
