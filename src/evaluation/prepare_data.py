"""
prepare_data.py

Loads RAGBench from HuggingFace.

Actual RAGBench schema (confirmed from dataset card):
  - id: string
  - question: string
  - documents: sequence of strings  ← this is the context passages
  - response: string                ← this is the answer

Run: python src/evaluation/prepare_data.py
Requires: pip install datasets
"""

import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)

# Three domains for variety across technical, finance, and science.
# Available in RAGBench HuggingFace (test+train combined):
#   techqa:  ~1,506 pairs
#   finqa:   ~14,796 pairs
#   covidqa: ~1,498 pairs
#
# Default: load all available data. run_benchmark.py handles balancing via --n-pairs.
RAGBENCH_SUBSETS = {
    "techqa":  99999,  # IBM tech support docs — load all (~1,506)
    "finqa":   99999,  # Financial documents — load all (~14,796)
    "covidqa": 99999,  # Scientific/medical — load all (~1,498)
}


def load_ragbench_subset(subset: str, n: int) -> tuple[list[dict], list[dict]]:
    """
    Load documents and QA pairs from one RAGBench subset.
    Combines test + train splits to maximize available data.

    Each row contains:
      - question: the query
      - documents: list of context passages (these become our retrieval corpus)
      - response: the ground truth answer
    """
    print(f"  Loading {subset} (up to {n})...")

    # Combine test + train splits for maximum data
    rows = []
    for split in ["test", "train"]:
        try:
            ds = load_dataset("rungalileo/ragbench", subset, split=split)
            rows.extend(ds)
            print(f"    {split}: {len(ds)} rows")
        except Exception as e:
            print(f"    {split}: not available ({e})")

    if not rows:
        print(f"  Could not load {subset}")
        return [], []

    documents = []
    qa_pairs = []
    seen = set()

    for i, row in enumerate(rows):
        if i >= n:
            break

        question = row.get("question", "").strip()
        response = row.get("response", "").strip()
        passages = row.get("documents", [])  # ← correct field name

        if not question or not response or not passages:
            continue

        # Each passage in 'documents' becomes a retrievable document
        for j, passage in enumerate(passages):
            if not passage or len(passage) < 50:
                continue
            key = passage[:80]
            if key in seen:
                continue
            seen.add(key)
            documents.append({
                "id": f"{subset}_{i}_{j}",
                "title": f"{subset.upper()} | Q{i} passage {j}",
                "content": passage,
                "domain": subset,
                "source_question_idx": i,
            })

        qa_pairs.append({
            "question": question,
            "ground_truth": response,
            "domain": subset,
            "source": "ragbench",
            "relevant_doc_ids": [
                f"{subset}_{i}_{j}" for j in range(len(passages))
            ],
        })

    print(f"  → {len(documents)} passages, {len(qa_pairs)} QA pairs")
    return documents, qa_pairs


def generate_synthetic_for_domain(documents: list[dict], domain: str, n_generate: int) -> list[dict]:
    """
    Generate synthetic QA pairs for a specific domain using an LLM.
    Uses existing documents as context to create question-answer pairs.
    """
    from langchain_openai import ChatOpenAI
    import random

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    domain_docs = [d for d in documents if d.get("domain") == domain]
    if not domain_docs:
        return []

    synthetic_pairs = []
    batch_size = 5  # generate 5 QA pairs per LLM call

    for i in range(0, n_generate, batch_size):
        n_this_batch = min(batch_size, n_generate - i)
        # Pick random docs as context
        sample_docs = random.sample(domain_docs, min(3, len(domain_docs)))
        context = "\n\n".join(
            f"Document: {d['title']}\n{d['content'][:500]}" for d in sample_docs
        )

        prompt = f"""Based on the following {domain} documents, generate exactly {n_this_batch} question-answer pairs.
Each question should be answerable from the documents. Each answer should be concise and factual.

Documents:
{context}

Respond with a JSON array of objects, each with "question" and "answer" keys.
Example: [{{"question": "What is X?", "answer": "X is..."}}]"""

        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()
            # Strip markdown code fences if present
            import re
            raw = re.sub(r"```json|```", "", raw).strip()
            pairs = json.loads(raw)
            for p in pairs:
                if p.get("question") and p.get("answer"):
                    synthetic_pairs.append({
                        "question": p["question"],
                        "ground_truth": p["answer"],
                        "domain": domain,
                        "source": "synthetic_expansion",
                    })
        except Exception as e:
            print(f"    Synthetic generation error: {e}")

        if len(synthetic_pairs) >= n_generate:
            break

    return synthetic_pairs[:n_generate]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=0,
                        help="Target total QA pairs. If set, generates synthetic pairs for underrepresented domains to reach this target with balanced distribution.")
    args = parser.parse_args()

    all_documents = []
    all_qa_pairs = []

    print("Loading RAGBench subsets from HuggingFace...")
    for subset, n in RAGBENCH_SUBSETS.items():
        docs, qas = load_ragbench_subset(subset, n=n)
        all_documents.extend(docs)
        all_qa_pairs.extend(qas)

    from collections import Counter
    domain_counts = Counter(qa["domain"] for qa in all_qa_pairs)
    doc_counts = Counter(d["domain"] for d in all_documents)
    print(f"\nTotal: {len(all_documents)} document passages, {len(all_qa_pairs)} QA pairs")
    print(f"QA pairs by domain:  {dict(domain_counts)}")
    print(f"Documents by domain: {dict(doc_counts)}")

    # Synthetic expansion: generate QA pairs for underrepresented domains
    if args.target > len(all_qa_pairs):
        domains = sorted(domain_counts.keys())
        target_per_domain = args.target // len(domains)
        print(f"\n── Synthetic expansion to {args.target} pairs ({target_per_domain}/domain) ──")

        for domain in domains:
            current = domain_counts[domain]
            deficit = target_per_domain - current
            if deficit > 0:
                print(f"  {domain}: {current} real → generating {deficit} synthetic...")
                synthetic = generate_synthetic_for_domain(all_documents, domain, deficit)
                all_qa_pairs.extend(synthetic)
                print(f"  {domain}: +{len(synthetic)} synthetic = {current + len(synthetic)} total")
            else:
                print(f"  {domain}: {current} real (already >= {target_per_domain}, no expansion needed)")

        # Recount
        domain_counts = Counter(qa["domain"] for qa in all_qa_pairs)
        print(f"\nAfter expansion: {len(all_qa_pairs)} QA pairs")
        print(f"QA pairs by domain: {dict(domain_counts)}")

    # Save
    docs_path = DATA_DIR / "raw" / "ragbench_documents.json"
    with open(docs_path, "w") as f:
        json.dump(all_documents, f, indent=2)
    print(f"Saved documents → {docs_path}")

    qa_path = DATA_DIR / "qa_pairs.json"
    with open(qa_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)
    print(f"Saved QA pairs  → {qa_path}")

    print("\nSample questions per domain:")
    seen_domains = set()
    for qa in all_qa_pairs:
        if qa["domain"] not in seen_domains:
            print(f"  [{qa['domain']}] {qa['question'][:80]}...")
            seen_domains.add(qa["domain"])


if __name__ == "__main__":
    main()