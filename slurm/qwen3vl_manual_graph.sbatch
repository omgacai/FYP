#!/usr/bin/env bash
# Direct image-only Qwen3-VL spatial graph prediction for the held-out manual
# benchmark. Submit as a Slurm array; each task writes a disjoint JSONL shard.
# The runner stores the complete prompt, raw response, and validated graph.
#SBATCH --job-name=qwen3vl-manual
#SBATCH --partition=gpu
#SBATCH --nodes=1
# xgph0–xgph9 are the verified x86-64 A100-80 candidate pool.
#SBATCH --nodelist=xgph[0-9]
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/qwen3vl-manual-%A_%a.out
#SBATCH --error=slurm/logs/qwen3vl-manual-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
: "${VLM_MANIFEST:?Set VLM_MANIFEST to prepared manual Qwen JSONL}"
: "${VLM_MANUAL_ROOT:?Set VLM_MANUAL_ROOT to packaged manual-20 directory}"
: "${VLM_OUTPUT_DIR:?Set VLM_OUTPUT_DIR to a new output directory}"
VLM_NUM_SHARDS="${VLM_NUM_SHARDS:-4}"
VLM_SHARD_INDEX="${VLM_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
VLM_RUN_NAME="${VLM_RUN_NAME:-qwen3vl8b_manual20_direct_spatial}"
VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-1048576}"
VLM_PROMPT_VERSION="${VLM_PROMPT_VERSION:-baseline}"
VLM_JSON_REPAIR_ATTEMPTS="${VLM_JSON_REPAIR_ATTEMPTS:-1}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "This Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing Qwen ready marker: $VENV_DIR/.ready" >&2; exit 2; }
[[ -f "$VLM_MANIFEST" && -d "$VLM_MANUAL_ROOT" ]] || { echo "Missing manual input paths" >&2; exit 2; }
mkdir -p "$REPO_DIR/slurm/logs" "$VLM_OUTPUT_DIR" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "Host: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m retrieval_app.scripts.run_qwen_graph \
  --manifest "$VLM_MANIFEST" --corpus-root "$VLM_MANUAL_ROOT" \
  --output "$VLM_OUTPUT_DIR/${VLM_RUN_NAME}.shard${VLM_SHARD_INDEX}.jsonl" \
  --model Qwen/Qwen3-VL-8B-Instruct --mode zero --representation spatial \
  --prompt-version "$VLM_PROMPT_VERSION" --graph-context none --limit 0 \
  --num-shards "$VLM_NUM_SHARDS" --shard-index "$VLM_SHARD_INDEX" \
  --max-pixels "$VLM_MAX_PIXELS" --max-new-tokens 1024 --json-repair-attempts "$VLM_JSON_REPAIR_ATTEMPTS"
