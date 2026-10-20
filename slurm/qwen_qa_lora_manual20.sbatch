#!/usr/bin/env bash
# Sharded Manual20 image-only QA evaluation for a trained Qwen LoRA adapter.
#SBATCH --job-name=qwen-qa-lora-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/qwen-qa-lora-eval-%A_%a.out
#SBATCH --error=slurm/logs/qwen-qa-lora-eval-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
: "${QA_QUESTIONS:?Set QA_QUESTIONS to the frozen Manual20 question CSV}"
: "${QA_IMAGES_DIR:?Set QA_IMAGES_DIR to Manual20 images}"
: "${QA_ADAPTER:?Set QA_ADAPTER to the trained adapter directory}"
: "${QA_OUTPUT_DIR:?Set QA_OUTPUT_DIR to a new evaluation folder}"
QA_NUM_SHARDS="${QA_NUM_SHARDS:-4}"
QA_SHARD_INDEX="${QA_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
mkdir -p "$REPO_DIR/slurm/logs" "$QA_OUTPUT_DIR" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface" TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
python -m retrieval_app.scripts.run_qwen_qa_lora \
  --questions "$QA_QUESTIONS" --images-dir "$QA_IMAGES_DIR" --adapter "$QA_ADAPTER" \
  --output "$QA_OUTPUT_DIR/image_only_lora.shard${QA_SHARD_INDEX}.jsonl" \
  --num-shards "$QA_NUM_SHARDS" --shard-index "$QA_SHARD_INDEX" --max-pixels 1048576 --max-new-tokens 512
