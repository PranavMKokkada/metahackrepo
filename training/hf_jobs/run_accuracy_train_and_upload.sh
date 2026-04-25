#!/usr/bin/env bash
set -euo pipefail

# Required env vars (passed by hf jobs run):
# - MODEL_REPO (e.g. teletubbies/autonomous-sre-lora)
# - DATASET_REPO (e.g. teletubbies/autonomous-sre-logs)
# - HF_TOKEN (secret)

apt-get update >/dev/null
apt-get install -y git wget >/dev/null

git clone https://github.com/PranavMKokkada/metahackrepo.git /workspace/metahackrepo
cd /workspace/metahackrepo

mkdir -p training
wget -q -O training/sft_data.jsonl \
  https://huggingface.co/datasets/teletubbies/autonomous-sre-logs/resolve/main/training/sft_data.jsonl

pip install --upgrade pip >/dev/null
pip install -r requirements-training.txt >/dev/null
pip install -U huggingface_hub >/dev/null

python training/grpo_train.py \
  --mode grpo \
  --run-gpu \
  --model-id unsloth/llama-3-8b-Instruct-bnb-4bit \
  --dataset-path training/sft_data.jsonl \
  --output-dir results/grpo_run_accuracy \
  --max-seq-length 2048 \
  --max-steps 300 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --warmup-steps 30 \
  --logging-steps 10 \
  --save-steps 100 \
  --lora-r 32 \
  --lora-alpha 32 \
  --seed 42

# Upload trained adapter and metrics back to the user's HF repos.
hf upload "$MODEL_REPO" results/grpo_run_accuracy/lora_adapter lora_adapter \
  --repo-type model \
  --token "$HF_TOKEN" \
  --commit-message "Upload trained LoRA adapter from HF Job"

if [[ -f results/grpo_gpu_run_result.json ]]; then
  hf upload "$DATASET_REPO" results/grpo_gpu_run_result.json training/grpo_gpu_run_result.json \
    --repo-type dataset \
    --token "$HF_TOKEN" \
    --commit-message "Upload training metrics from HF Job"
fi

if [[ -f results/training_run_summary.json ]]; then
  hf upload "$DATASET_REPO" results/training_run_summary.json training/training_run_summary.json \
    --repo-type dataset \
    --token "$HF_TOKEN" \
    --commit-message "Upload training metrics from HF Job"
fi
