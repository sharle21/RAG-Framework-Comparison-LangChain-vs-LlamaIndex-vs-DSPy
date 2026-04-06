#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Lambda Cloud Setup Script
#
# Run this on any Lambda H100/A100 instance to set up the full benchmark stack.
# Usage: bash setup_lambda.sh <huggingface_token>
# ─────────────────────────────────────────────────────────────────────────────
set -e

HF_TOKEN="${1:?Usage: bash setup_lambda.sh <huggingface_token>}"

echo "=== Creating Python venv (isolate from system TensorFlow) ==="
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate

echo "=== Installing vLLM ==="
pip install vllm

echo "=== Installing benchmark dependencies ==="
pip install langchain langchain-openai langchain-community langchain-mistralai \
    langchain-huggingface langchain-text-splitters \
    llama-index llama-index-embeddings-openai llama-index-embeddings-huggingface \
    llama-index-llms-openai \
    dspy faiss-cpu chromadb ragas python-dotenv bert-score \
    sentence-transformers datasets

echo "=== HuggingFace login ==="
huggingface-cli login --token "$HF_TOKEN"

echo "=== Cloning repo ==="
cd ~
git clone https://github.com/sharle21/RAG-Framework-Comparison-LangChain-vs-LlamaIndex-vs-DSPy.git rag-bench

echo "=== Preparing data ==="
cd ~/rag-bench
python3 src/evaluation/prepare_data.py

echo "=== Starting vLLM servers ==="
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 --gpu-memory-utilization 0.45 \
    > /tmp/vllm_worker.log 2>&1 &

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --port 8001 --gpu-memory-utilization 0.45 --max-model-len 4096 \
    > /tmp/vllm_judge.log 2>&1 &

echo "=== Waiting for servers to be ready ==="
for port in 8000 8001; do
    echo -n "Waiting for :$port"
    until curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; do
        echo -n "."
        sleep 10
    done
    echo " ready!"
done

echo ""
echo "=== Setup complete! ==="
echo "Run the benchmark:"
echo "  source ~/vllm_env/bin/activate"
echo "  cd ~/rag-bench"
echo "  PYTHONUNBUFFERED=1 python3 src/evaluation/run_benchmark.py --n-pairs 5000 --vllm --local-embeddings"
echo ""
echo "Or run everything (benchmark + sweep + push):"
echo "  nohup bash orchestrator/run_all.sh > /tmp/pipeline.log 2>&1 &"
