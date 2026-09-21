#!/bin/bash
#SBATCH --job-name=egtr-calibration
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
# Submit from the FYP root. Use a dedicated upstream-compatible EGTR environment.
: "${EGTR_PYTHON:?Set EGTR_PYTHON to the EGTR environment python executable}"
: "${EGTR_ARTIFACT:?Set EGTR_ARTIFACT to the checkpoint configuration directory}"
: "${EGTR_CHECKPOINT:?Set EGTR_CHECKPOINT to the explicit full EGTR checkpoint}"
: "${EGTR_LABELS:?Set EGTR_LABELS to verified objects/relations vocabulary JSON}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$PWD/cubicasa_eval/calibration/egtr_v1}"
"$EGTR_PYTHON" experiments/egtr_calibration/infer.py \
  --root "$CALIBRATION_ROOT" --artifact "$EGTR_ARTIFACT" \
  --checkpoint "$EGTR_CHECKPOINT" --labels "$EGTR_LABELS"
