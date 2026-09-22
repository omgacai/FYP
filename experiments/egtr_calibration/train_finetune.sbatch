#!/usr/bin/env bash
# Submit with variables shown in the EGTR calibration README / task instructions.
# This job trains solely on the silver train/validation adapter; manual gold is absent.
#SBATCH --job-name=cubicasa-egtr
#SBATCH --partition=gpu
# The shared PyTorch installation is x86-64. xgph0–xgph9 are the SOC
# x86-64 A100-80 nodes, so leave Slurm ten compatible choices without risking
# an ARM allocation (for example, xgpj0).
#SBATCH --nodelist=xgph[0-9]
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail

: "${EGTR_PYTHON:?Set EGTR_PYTHON}"
: "${EGTR_CONFIG:?Set EGTR_CONFIG to released config.json}"
: "${EGTR_CHECKPOINT:?Set EGTR_CHECKPOINT to released checkpoint}"
: "${TRAIN_OUT:?Set a new output directory}"
EGTR_DATA="${EGTR_DATA:-$HOME/vlm/data/derived/egtr_finetune_v1/cubicasa_visual_genome_v1}"
EPOCHS="${EPOCHS:-1}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
export PYTHONPATH="$HOME/aigc-storage/fyp-envs/egtr-deps-py312-v1"
cd "$HOME/vlm/code/FYP"
extra_args=()
if [[ "$MAX_TRAIN_BATCHES" != 0 ]]; then extra_args+=(--max-train-batches "$MAX_TRAIN_BATCHES"); fi
if [[ "$MAX_VAL_BATCHES" != 0 ]]; then extra_args+=(--max-val-batches "$MAX_VAL_BATCHES"); fi
"$EGTR_PYTHON" -u experiments/egtr_calibration/train_finetune.py \
  --data "$EGTR_DATA" --config "$EGTR_CONFIG" --checkpoint "$EGTR_CHECKPOINT" \
  --output "$TRAIN_OUT" --epochs "$EPOCHS" --batch-size 1 --accumulate 4 \
  --num-workers 2 --log-every 25 "${extra_args[@]}"
