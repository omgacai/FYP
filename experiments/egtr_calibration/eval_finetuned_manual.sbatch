#!/usr/bin/env bash
# Image-only fine-tuned EGTR inference followed by manual-20 evaluation.
# The model never receives manual graph annotations; the Node evaluator reads
# them only after predictions have been written.
#SBATCH --job-name=egtr-manual-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=xgph[0-9]
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
: "${EGTR_RUN:?Set EGTR_RUN to a newly initialized manual-20 run}"
: "${FINETUNED_CHECKPOINT:?Set FINETUNED_CHECKPOINT to best.pt}"
: "${FINETUNED_CONFIG:?Set FINETUNED_CONFIG to the sibling model_config directory}"
OBJECT_THRESHOLD="${OBJECT_THRESHOLD:-0.30}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-0.01}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Requires x86_64 Torch" >&2; exit 2; }
[[ -f "$FINETUNED_CHECKPOINT" && -f "$FINETUNED_CONFIG/config.json" ]] || { echo "Missing checkpoint or model config" >&2; exit 2; }
mkdir -p "$EGTR_RUN/logs"
export PYTHONPATH="$HOME/aigc-storage/fyp-envs/egtr-deps-py312-v1"
cd "$REPO_DIR"

"$VENV_DIR/bin/python" -u experiments/egtr_calibration/infer_finetuned.py \
  --root "$EGTR_RUN" --checkpoint "$FINETUNED_CHECKPOINT" --model-config "$FINETUNED_CONFIG" \
  --object-threshold "$OBJECT_THRESHOLD" --relation-threshold "$RELATION_THRESHOLD"
node experiments/egtr_calibration/evaluate.mjs "$EGTR_RUN" --prediction-dir predictions/egtr_finetuned
"$VENV_DIR/bin/python" experiments/egtr_calibration/preview.py --root "$EGTR_RUN" --prediction-dir predictions/egtr_finetuned
"$VENV_DIR/bin/python" experiments/egtr_calibration/report.py --root "$EGTR_RUN" --prediction-dir predictions/egtr_finetuned
