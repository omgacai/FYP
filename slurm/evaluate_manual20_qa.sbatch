#!/usr/bin/env bash
# Score a completed, sharded Manual20 QA run. Submit only after every QA array succeeds.
#SBATCH --job-name=manual20-qa-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/manual20-qa-eval-%j.out
#SBATCH --error=slurm/logs/manual20-qa-eval-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
: "${QA_QUESTIONS:?Set QA_QUESTIONS to the frozen Manual20 question CSV}"
: "${QA_OUTPUT_DIR:?Set QA_OUTPUT_DIR to the QA run directory}"
QA_RESULTS_DIR="${QA_RESULTS_DIR:-$QA_OUTPUT_DIR/results}"
QA_NUM_SHARDS="${QA_NUM_SHARDS:-4}"
QA_PREDICTION_SUFFIX="${QA_PREDICTION_SUFFIX:-}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

shopt -s nullglob
image_only=("$QA_OUTPUT_DIR"/image_only"${QA_PREDICTION_SUFFIX}".shard*.jsonl)
graph_only=("$QA_OUTPUT_DIR"/graph_only_gold_manual"${QA_PREDICTION_SUFFIX}".shard*.jsonl)
image_graph=("$QA_OUTPUT_DIR"/image_graph_gold_manual"${QA_PREDICTION_SUFFIX}".shard*.jsonl)
[[ ${#image_only[@]} -eq "$QA_NUM_SHARDS" ]] || { echo "Expected $QA_NUM_SHARDS image_only shards; found ${#image_only[@]}" >&2; exit 2; }
[[ ${#graph_only[@]} -eq "$QA_NUM_SHARDS" ]] || { echo "Expected $QA_NUM_SHARDS graph_only shards; found ${#graph_only[@]}" >&2; exit 2; }
[[ ${#image_graph[@]} -eq "$QA_NUM_SHARDS" ]] || { echo "Expected $QA_NUM_SHARDS image_graph shards; found ${#image_graph[@]}" >&2; exit 2; }

python -m retrieval_app.scripts.evaluate_qa \
  --questions "$QA_QUESTIONS" \
  --predictions "${image_only[@]}" \
  --predictions "${graph_only[@]}" \
  --predictions "${image_graph[@]}" \
  --output-dir "$QA_RESULTS_DIR"
