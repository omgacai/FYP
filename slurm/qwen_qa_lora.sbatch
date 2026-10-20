#!/usr/bin/env bash
# Qwen image-only QA LoRA launch.  The historical trainer module supports both
# graph and QA chat records; this launcher keeps QA outputs/provenance separate.
#SBATCH --job-name=qwen-qa-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/qwen-qa-lora-%j.out
#SBATCH --error=slurm/logs/qwen-qa-lora-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
CUBICASA_ROOT="${CUBICASA_ROOT:?Set CUBICASA_ROOT to the CubiCasa corpus root}"
QA_SFT_DATASET_DIR="${QA_SFT_DATASET_DIR:?Set QA_SFT_DATASET_DIR to the generated QA SFT directory}"
QA_RUN_NAME="${QA_RUN_NAME:?Set QA_RUN_NAME to a fresh descriptive name}"
QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-$PROJECT_DIR/outputs/qwen_qa_lora/$QA_RUN_NAME}"
QA_EPOCHS="${QA_EPOCHS:-3}"
QA_MAX_STEPS="${QA_MAX_STEPS:--1}"
QA_PER_DEVICE_TRAIN_BATCH_SIZE="${QA_PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
QA_GRADIENT_ACCUMULATION_STEPS="${QA_GRADIENT_ACCUMULATION_STEPS:-1}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
[[ -f "$QA_SFT_DATASET_DIR/train.jsonl" && -f "$QA_SFT_DATASET_DIR/val.jsonl" ]] || { echo "Missing QA SFT train/val JSONL" >&2; exit 2; }
[[ ! -e "$QA_OUTPUT_DIR" ]] || { echo "Output already exists: $QA_OUTPUT_DIR; choose a new QA_RUN_NAME" >&2; exit 2; }

mkdir -p "$REPO_DIR/slurm/logs" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface" TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

python -m retrieval_app.scripts.train_qwen_graph_lora \
  --dataset-dir "$QA_SFT_DATASET_DIR" \
  --corpus-root "$CUBICASA_ROOT" \
  --output-dir "$QA_OUTPUT_DIR" \
  --task-name qa_image_only_cubigraph_v1 \
  --epochs "$QA_EPOCHS" \
  --max-steps "$QA_MAX_STEPS" \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --per-device-train-batch-size "$QA_PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$QA_GRADIENT_ACCUMULATION_STEPS" \
  --eval-strategy epoch \
  --save-strategy epoch \
  --eval-steps 1 \
  --save-steps 1 \
  --max-pixels 501760
