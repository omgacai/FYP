#!/usr/bin/env bash
# Teacher-forced category breakdown on the held-out QA SFT validation split.
# It never reads Manual20.
#SBATCH --job-name=qwen-qa-valcat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/qwen-qa-valcat-%j.out
#SBATCH --error=slurm/logs/qwen-qa-valcat-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
CUBICASA_ROOT="${CUBICASA_ROOT:?Set CUBICASA_ROOT}"
QA_SFT_DATASET_DIR="${QA_SFT_DATASET_DIR:?Set QA_SFT_DATASET_DIR}"
QA_ADAPTER_DIR="${QA_ADAPTER_DIR:?Set QA_ADAPTER_DIR to the completed adapter directory}"
QA_VAL_CATEGORY_EVAL_OUT="${QA_VAL_CATEGORY_EVAL_OUT:?Set QA_VAL_CATEGORY_EVAL_OUT to a fresh output directory}"
QA_MAX_PIXELS="${QA_MAX_PIXELS:-501760}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
[[ -f "$QA_SFT_DATASET_DIR/val.jsonl" ]] || { echo "Missing validation JSONL" >&2; exit 2; }
[[ -f "$QA_ADAPTER_DIR/adapter_config.json" ]] || { echo "Missing adapter" >&2; exit 2; }
[[ ! -e "$QA_VAL_CATEGORY_EVAL_OUT" ]] || { echo "Output already exists: $QA_VAL_CATEGORY_EVAL_OUT" >&2; exit 2; }

mkdir -p "$REPO_DIR/slurm/logs" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface" TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

python -m retrieval_app.scripts.evaluate_qwen_qa_validation_by_category \
  --dataset-dir "$QA_SFT_DATASET_DIR" \
  --corpus-root "$CUBICASA_ROOT" \
  --adapter-dir "$QA_ADAPTER_DIR" \
  --output-dir "$QA_VAL_CATEGORY_EVAL_OUT" \
  --max-pixels "$QA_MAX_PIXELS"
