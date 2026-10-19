#!/usr/bin/env bash
# Evaluate a saved Qwen graph LoRA adapter on filtered CubiCasa validation.
#SBATCH --job-name=qwen-graph-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/qwen-graph-eval-%j.out
#SBATCH --error=slurm/logs/qwen-graph-eval-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
CUBICASA_ROOT="${CUBICASA_ROOT:?Set CUBICASA_ROOT}"
SFT_DATASET_DIR="${SFT_DATASET_DIR:?Set SFT_DATASET_DIR}"
SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:?Set SFT_ADAPTER_DIR to the saved adapter directory}"
SFT_EVAL_OUTPUT_DIR="${SFT_EVAL_OUTPUT_DIR:?Set SFT_EVAL_OUTPUT_DIR to a fresh output directory}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
[[ -f "$SFT_DATASET_DIR/val.jsonl" ]] || { echo "Missing validation data: $SFT_DATASET_DIR/val.jsonl" >&2; exit 2; }
[[ -f "$SFT_ADAPTER_DIR/adapter_config.json" ]] || { echo "Missing adapter: $SFT_ADAPTER_DIR" >&2; exit 2; }
[[ ! -e "$SFT_EVAL_OUTPUT_DIR" ]] || { echo "Evaluation output already exists: $SFT_EVAL_OUTPUT_DIR" >&2; exit 2; }

mkdir -p "$REPO_DIR/slurm/logs" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface" TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

python -m retrieval_app.scripts.evaluate_qwen_graph_lora \
  --dataset-dir "$SFT_DATASET_DIR" \
  --corpus-root "$CUBICASA_ROOT" \
  --adapter-dir "$SFT_ADAPTER_DIR" \
  --output-dir "$SFT_EVAL_OUTPUT_DIR"
