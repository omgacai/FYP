#!/usr/bin/env bash
# Build plan-disjoint QA SFT pairs from the CubiGraph silver corpus.
#SBATCH --job-name=qwen-qa-data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/qwen-qa-data-%j.out
#SBATCH --error=slurm/logs/qwen-qa-data-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CANONICAL_CORPUS="${CANONICAL_CORPUS:-$PROJECT_DIR/data/cubicasa5k/derived_full_v1/canonical.jsonl}"
EXCLUSIONS="${EXCLUSIONS:-$PROJECT_DIR/data/cubicasa5k/manual20_exclusion_manifest.json}"
QA_SFT_OUT="${QA_SFT_OUT:-$PROJECT_DIR/data/qwen_qa_sft/cubigraph_v1_filtered_5q}"
QA_INPUT_CONDITION="${QA_INPUT_CONDITION:-image_only}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
[[ -f "$CANONICAL_CORPUS" && -f "$EXCLUSIONS" ]] || { echo "Missing canonical corpus or Manual20 exclusion manifest" >&2; exit 2; }
[[ ! -e "$QA_SFT_OUT" ]] || { echo "Output already exists: $QA_SFT_OUT; choose a fresh QA_SFT_OUT" >&2; exit 2; }

mkdir -p "$REPO_DIR/slurm/logs"
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
python -m retrieval_app.scripts.build_qwen_qa_sft_dataset \
  --canonical-corpus "$CANONICAL_CORPUS" \
  --exclude-identities "$EXCLUSIONS" \
  --output-dir "$QA_SFT_OUT" \
  --input-condition "$QA_INPUT_CONDITION"
