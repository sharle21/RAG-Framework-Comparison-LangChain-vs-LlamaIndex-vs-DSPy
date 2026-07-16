"""
run_semantic_sim.py — semantic similarity via bge-m3 embeddings (L13).

Cosine similarity between answer and ground truth using the same embedding
model used for retrieval. Captures semantic equivalence beyond token overlap.

Requires GPU. Run on Lambda after benchmark:
    python run_semantic_sim.py
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ubuntu/rag-bench")

RESULTS_FILE = Path("/home/ubuntu/rag-bench/results/go_results_20260408_013644.json")
OUT_FILE = Path("/home/ubuntu/rag-bench/results/semantic_sim_results.json")


def main():
    print("Loading bge-m3...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    valid = [r for r in data if r.get("answer") and r.get("ground_truth") and not r.get("error")]
    print(f"Valid results: {len(valid)}")

    answers = [r["answer"] for r in valid]
    ground_truths = [r["ground_truth"] for r in valid]

    print("Encoding answers...")
    ans_embs = model.encode(answers, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print("Encoding ground truths...")
    gt_embs = model.encode(ground_truths, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # Cosine sim = dot product since embeddings are L2-normalized
    sims = (ans_embs * gt_embs).sum(axis=1)

    fw_sims = defaultdict(list)
    fw_domain_sims = defaultdict(lambda: defaultdict(list))
    for r, sim in zip(valid, sims):
        fw_sims[r["framework"]].append(float(sim))
        fw_domain_sims[r["framework"]][r.get("domain", "unknown")].append(float(sim))

    print("\n=== SEMANTIC SIMILARITY (bge-m3 cosine) ===")
    out = {}
    for fw in sorted(fw_sims.keys()):
        scores = fw_sims[fw]
        mean = sum(scores) / len(scores)
        p50 = statistics.median(scores)
        print(f"  {fw}: mean={mean:.4f}  median={p50:.4f}  n={len(scores)}")
        by_domain = {}
        for domain, d_scores in fw_domain_sims[fw].items():
            by_domain[domain] = round(sum(d_scores) / len(d_scores), 4)
        print(f"    by domain: {by_domain}")
        out[fw] = {
            "mean": round(mean, 4),
            "median": round(p50, 4),
            "n": len(scores),
            "by_domain": by_domain,
        }

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
