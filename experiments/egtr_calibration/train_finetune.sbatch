#!/usr/bin/env bash
# Submit with variables shown in the EGTR calibration README / task instructions.
# This job trains solely on the silver train/validation adapter; manual gold is absent.
#SBATCH --job-name=cubicasa-egtr
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph1
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
export PYTHONPATH="$HOME/aigc-storage/fyp-envs/egtr-deps-py312-v1"
cd "$HOME/vlm/code/FYP"
"$EGTR_PYTHON" -u experiments/egtr_calibration/train_finetune.py \
  --data "$EGTR_DATA" --config "$EGTR_CONFIG" --checkpoint "$EGTR_CHECKPOINT" \
  --output "$TRAIN_OUT" --epochs "$EPOCHS" --batch-size 1 --accumulate 4 \
  --num-workers 2 --log-every 25
