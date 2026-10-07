#!/usr/bin/env bash
# Submit one array per input condition. Every task writes a disjoint QA shard.
#SBATCH --job-name=qwen3vl-qa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=xgph[0-9]
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/qwen3vl-qa-%A_%a.out
#SBATCH --error=slurm/logs/qwen3vl-qa-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
: "${QA_QUESTIONS:?Set QA_QUESTIONS to manual20_qa_v1.csv}"
: "${QA_ANNOTATIONS_DIR:?Set QA_ANNOTATIONS_DIR to Manual20 annotations/}"
: "${QA_IMAGES_DIR:?Set QA_IMAGES_DIR to Manual20 images/}"
: "${QA_OUTPUT_DIR:?Set QA_OUTPUT_DIR to a new output directory}"
: "${QA_CONDITION:?Set QA_CONDITION to image_only, graph_only, or image_graph}"
QA_GRAPH_DIR="${QA_GRAPH_DIR:-$QA_ANNOTATIONS_DIR}"
QA_GRAPH_SOURCE="${QA_GRAPH_SOURCE:-gold_manual}"
QA_NUM_SHARDS="${QA_NUM_SHARDS:-4}"
QA_SHARD_INDEX="${QA_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
mkdir -p "$REPO_DIR/slurm/logs" "$QA_OUTPUT_DIR" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface" TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

OUTPUT_PATH="$QA_OUTPUT_DIR/${QA_CONDITION}.shard${QA_SHARD_INDEX}.jsonl"
if [[ "$QA_CONDITION" != "image_only" ]]; then
  OUTPUT_PATH="$QA_OUTPUT_DIR/${QA_CONDITION}_${QA_GRAPH_SOURCE}.shard${QA_SHARD_INDEX}.jsonl"
fi
args=(--questions "$QA_QUESTIONS" --annotations-dir "$QA_ANNOTATIONS_DIR" --images-dir "$QA_IMAGES_DIR"
  --output "$OUTPUT_PATH"
  --condition "$QA_CONDITION" --model Qwen/Qwen3-VL-8B-Instruct --num-shards "$QA_NUM_SHARDS"
  --shard-index "$QA_SHARD_INDEX" --max-pixels 1048576 --max-new-tokens 512)
if [[ "$QA_CONDITION" != "image_only" ]]; then
  args+=(--graph-dir "$QA_GRAPH_DIR" --graph-source "$QA_GRAPH_SOURCE")
fi
python -m retrieval_app.scripts.run_qwen_qa "${args[@]}"
