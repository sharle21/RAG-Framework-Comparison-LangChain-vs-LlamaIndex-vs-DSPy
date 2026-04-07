#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Lambda Cloud Setup Script (GH200 / A100 / H100)
#
# Run this once on a fresh Lambda instance. It installs everything, pulls
# models, starts vLLM, and prints the command to run the benchmark.
#
# Usage: bash setup_lambda.sh <huggingface_token>
# ─────────────────────────────────────────────────────────────────────────────
set -e

HF_TOKEN="${1:?Usage: bash setup_lambda.sh <huggingface_token>}"

echo "=== Creating Python venv (inherits system PyTorch + CUDA) ==="
python3 -m venv --system-site-packages ~/vllm_env
source ~/vllm_env/bin/activate

echo "=== Installing vLLM ==="
pip install vllm

echo "=== Installing benchmark dependencies ==="
pip install \
    langchain langchain-openai langchain-community langchain-mistralai \
    langchain-huggingface langchain-text-splitters \
    llama-index llama-index-embeddings-openai llama-index-embeddings-huggingface \
    llama-index-llms-openai \
    dspy faiss-cpu chromadb ragas python-dotenv bert-score \
    sentence-transformers datasets \
    fastapi uvicorn

echo "=== Installing Go ==="
# GH200 has an ARM CPU (Grace) — need arm64 binary
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    GO_ARCH="arm64"
else
    GO_ARCH="amd64"
fi
GO_VERSION="1.23.4"
curl -sL "https://go.dev/dl/go${GO_VERSION}.linux-${GO_ARCH}.tar.gz" | sudo tar -C /usr/local -xzf -
export PATH="/usr/local/go/bin:$PATH"
echo 'export PATH="/usr/local/go/bin:$PATH"' >> ~/.bashrc
go version

echo "=== HuggingFace login ==="
huggingface-cli login --token "$HF_TOKEN"

echo "=== Cloning repo ==="
cd ~
git clone https://github.com/sharle21/RAG-Framework-Comparison-LangChain-vs-LlamaIndex-vs-DSPy.git rag-bench

echo "=== Preparing data ==="
cd ~/rag-bench
python3 src/evaluation/prepare_data.py

echo "=== Building Go orchestrator ==="
cd ~/rag-bench/orchestrator
go build -o rag-orchestrator .
echo "Go orchestrator built"
cd ~/rag-bench

echo "=== Starting vLLM servers ==="
# GH200 has 96GB HBM3 — give each model 0.45 = ~43GB each, plenty of KV cache room.
# On A100 80GB, lower to 0.40 each.
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 --gpu-memory-utilization 0.45 \
    > /tmp/vllm_worker.log 2>&1 &

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --port 8001 --gpu-memory-utilization 0.45 --max-model-len 4096 \
    > /tmp/vllm_judge.log 2>&1 &

echo "=== Waiting for vLLM servers to be ready ==="
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
echo ""
echo "Run the benchmark (Go orchestrator + Python RAG servers):"
echo "  source ~/vllm_env/bin/activate"
echo "  cd ~/rag-bench"
echo "  bash orchestrator/run_servers.sh"
echo ""
echo "Or run everything (benchmark + sweep + push to GitHub):"
echo "  nohup bash orchestrator/run_all.sh > /tmp/pipeline.log 2>&1 &"
echo "  tail -f /tmp/pipeline.log"
