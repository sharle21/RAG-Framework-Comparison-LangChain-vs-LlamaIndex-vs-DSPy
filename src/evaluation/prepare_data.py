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

# Three domains for variety across technical, finance, and science
# This matters for adversarial attacks — OOD queries and trojans are
# more realistic when the corpus has genuine topic diversity
RAGBENCH_SUBSETS = {
    "techqa":  30,   # IBM tech support docs — longer passages, real-world
    "finqa":   30,   # Financial documents — numerical, precise language
    "covidqa": 30,   # Scientific/medical — domain-specific vocabulary
}


def load_ragbench_subset(subset: str, n: int) -> tuple[list[dict], list[dict]]:
    """
    Load documents and QA pairs from one RAGBench subset.
    
    Each row contains:
      - question: the query
      - documents: list of context passages (these become our retrieval corpus)
      - response: the ground truth answer
    """
    print(f"  Loading {subset}...")
    try:
        ds = load_dataset("rungalileo/ragbench", subset, split="test")
    except Exception:
        try:
            ds = load_dataset("rungalileo/ragbench", subset, split="train")
        except Exception as e:
            print(f"  Could not load {subset}: {e}")
            return [], []

    documents = []
    qa_pairs = []
    seen = set()

    for i, row in enumerate(ds):
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


def main():
    all_documents = []
    all_qa_pairs = []

    print("Loading RAGBench subsets from HuggingFace...")
    for subset, n in RAGBENCH_SUBSETS.items():
        docs, qas = load_ragbench_subset(subset, n=n)
        all_documents.extend(docs)
        all_qa_pairs.extend(qas)

    print(f"\nTotal: {len(all_documents)} document passages, {len(all_qa_pairs)} QA pairs")
    domains = set(d["domain"] for d in all_documents)
    print(f"Domains: {domains}")

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