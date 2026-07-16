"""
run_serial_latency.py — serial (no-concurrency) latency benchmark (L6).

Fires one query at a time to each framework so each query gets the full GPU.
Gives clean per-framework latency without concurrent GPU pressure.

Run on Lambda after all servers are up:
    python run_serial_latency.py
"""
import json
import statistics
import time
from pathlib import Path

import requests

RESULTS_DIR = Path("/home/ubuntu/rag-bench/results")
DATA_DIR = Path("/home/ubuntu/rag-bench/data")

ENDPOINTS = {
    "langchain":  "http://localhost:8100/query",
    "llamaindex": "http://localhost:8101/query",
    "dspy":       "http://localhost:8102/query",
}
N_PER_FRAMEWORK = 50


def main():
    with open(DATA_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    sample = qa_pairs[:N_PER_FRAMEWORK]
    all_results = []

    for fw, url in ENDPOINTS.items():
        print(f"\n=== {fw} ({len(sample)} queries, serial) ===")
        fw_results = []

        for i, qa in enumerate(sample):
            payload = {
                "question": qa["question"],
                "domain": qa.get("domain", "unknown"),
                "ground_truth": qa.get("ground_truth", ""),
            }
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                data["wall_ms"] = (time.perf_counter() - t0) * 1000
                fw_results.append(data)
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(sample)} — gen: {data['generation_ms']:.0f}ms  ret: {data['retrieval_ms']:.0f}ms")
            except Exception as e:
                print(f"  Error on q{i}: {e}")
                fw_results.append({"framework": fw, "error": str(e), "wall_ms": 0,
                                   "retrieval_ms": 0, "generation_ms": 0})

        all_results.extend(fw_results)

        gen_times = [r["generation_ms"] for r in fw_results if not r.get("error") and r["generation_ms"] > 0]
        ret_times = [r["retrieval_ms"] for r in fw_results if not r.get("error") and r["retrieval_ms"] > 0]
        if gen_times:
            gen_sorted = sorted(gen_times)
            p95_idx = int(0.95 * len(gen_sorted))
            print(f"  Generation — median: {statistics.median(gen_times):.0f}ms  "
                  f"p95: {gen_sorted[p95_idx]:.0f}ms  mean: {sum(gen_times)/len(gen_times):.0f}ms")
        if ret_times:
            print(f"  Retrieval  — median: {statistics.median(ret_times):.0f}ms")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"serial_latency_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
