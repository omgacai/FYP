#!/usr/bin/env bash
# Controlled Qwen prompt experiment: spatial zero-shot baseline versus verified
# CubiCasa few-shot prompt. Submit from ~/vlm/code/FYP.
# Required: VLM_MANIFEST and VLM_SUPPORT (two adjudicated calibration examples).
# Example:
# sbatch --export=ALL,VLM_MANIFEST=/path/frozen_audit.jsonl,VLM_SUPPORT=/path/verified_support_spatial.jsonl,VLM_LIMIT=20 slurm/qwen3vl_prompt_ablation.sbatch
#SBATCH --job-name=qwen3vl-prompt-ablation
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/qwen3vl-prompt-ablation-%j.out
#SBATCH --error=slurm/logs/qwen3vl-prompt-ablation-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
VLM_MANIFEST="${VLM_MANIFEST:?Set VLM_MANIFEST to the frozen audit/calibration manifest}"
VLM_SUPPORT="${VLM_SUPPORT:?Set VLM_SUPPORT to one or two adjudicated spatial support examples}"
VLM_LIMIT="${VLM_LIMIT:-20}"
VLM_RUN_PREFIX="${VLM_RUN_PREFIX:-qwen3vl8b_spatial_prompt_ablation}"

[[ -f "$VLM_MANIFEST" ]] || { echo "Missing VLM_MANIFEST: $VLM_MANIFEST" >&2; exit 2; }
[[ -f "$VLM_SUPPORT" ]] || { echo "Missing VLM_SUPPORT: $VLM_SUPPORT" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready environment: $VENV_DIR/.ready" >&2; exit 2; }

mkdir -p "$REPO_DIR/slurm/logs" "$PROJECT_DIR/outputs/vlm_graph" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

run_condition() {
  local name="$1"; shift
  python -m retrieval_app.scripts.run_qwen_graph \
    --manifest "$VLM_MANIFEST" --corpus-root "$PROJECT_DIR/data/cubicasa5k" \
    --output "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_PREFIX}_${name}.jsonl" \
    --representation spatial --limit "$VLM_LIMIT" "$@"
  python -m retrieval_app.scripts.evaluate_vlm_graph \
    --predictions "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_PREFIX}_${name}.jsonl" \
    --manifest "$VLM_MANIFEST" --corpus-root "$PROJECT_DIR/data/cubicasa5k" \
    --output "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_PREFIX}_${name}_silver_metrics.json"
}

run_condition baseline --mode zero --prompt-version baseline
run_condition cubicasa_fewshot_v1 --mode few --prompt-version cubicasa_fewshot_v1 --support-manifest "$VLM_SUPPORT"

echo "Completed controlled baseline and few-shot runs. Select a prompt only on calibration metrics; keep the audit/test split unseen."
