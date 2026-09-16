#!/usr/bin/env bash
# End-to-end CubiCasa -> CubiGraph -> Qwen experiment. It first builds a small
# source manifest and CubiGraph silver graph, then compares direct Qwen graph
# extraction, CubiCasa few-shot extraction, and few-shot silver-graph correction.
# Submit from ~/vlm/code/FYP. VLM_SUPPORT contains one or two support examples.
# Example:
# sbatch --export=ALL,VLM_SUPPORT=/path/support.jsonl,VLM_LIMIT=20 slurm/qwen3vl_prompt_ablation.sbatch
#SBATCH --job-name=qwen3vl-prompt-ablation
#SBATCH --partition=gpu
# Let Slurm choose any eligible GPU node; specific hosts may move partitions.
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# The 8B smoke test fits on 40 GB; this avoids waiting for the busy 80 GB pool.
#SBATCH --gres=gpu:a100-40:1
#SBATCH --output=slurm/logs/qwen3vl-prompt-ablation-%j.out
#SBATCH --error=slurm/logs/qwen3vl-prompt-ablation-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
VLM_SUPPORT="${VLM_SUPPORT:?Set VLM_SUPPORT to one or two spatial support examples}"
VLM_LIMIT="${VLM_LIMIT:-20}"
VLM_RUN_PREFIX="${VLM_RUN_PREFIX:-qwen3vl8b_spatial_prompt_ablation}"
CUBICASA_ROOT="${CUBICASA_ROOT:-$PROJECT_DIR/data/cubicasa5k}"
CUBIGRAPH_REPO="${CUBIGRAPH_REPO:-$REPO_DIR/third_party/CubiGraph5K}"
VLM_SOURCE_MANIFEST="${VLM_SOURCE_MANIFEST:-$PROJECT_DIR/data/cubicasa5k/manifests/${VLM_RUN_PREFIX}_source.jsonl}"
VLM_MANIFEST="${VLM_MANIFEST:-$PROJECT_DIR/data/cubicasa5k/manifests/${VLM_RUN_PREFIX}_cubigraph.jsonl}"
CUBIGRAPH_GRAPH_DIR="${CUBIGRAPH_GRAPH_DIR:-$PROJECT_DIR/outputs/cubigraph/${VLM_RUN_PREFIX}}"

[[ -f "$VLM_SUPPORT" ]] || { echo "Missing VLM_SUPPORT: $VLM_SUPPORT" >&2; exit 2; }
[[ -d "$CUBICASA_ROOT" ]] || { echo "Missing CUBICASA_ROOT: $CUBICASA_ROOT" >&2; exit 2; }
[[ -d "$CUBIGRAPH_REPO" ]] || { echo "Missing CUBIGRAPH_REPO: $CUBIGRAPH_REPO" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready environment: $VENV_DIR/.ready" >&2; exit 2; }

mkdir -p "$REPO_DIR/slurm/logs" "$PROJECT_DIR/outputs/vlm_graph" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

# The image/SVG source manifest and silver graph are generated here, so every
# Qwen condition receives the same CubiCasa/CubiGraph pipeline output.
python -m retrieval_app.scripts.build_cubicasa_manifest \
  --cubicasa-root "$CUBICASA_ROOT" --output "$VLM_SOURCE_MANIFEST" --limit "$VLM_LIMIT"
python -m retrieval_app.scripts.index_cubicasa_graphs \
  --manifest "$VLM_SOURCE_MANIFEST" --corpus-root "$CUBICASA_ROOT" \
  --cubigraph-repo "$CUBIGRAPH_REPO" --graph-dir "$CUBIGRAPH_GRAPH_DIR" \
  --output "$VLM_MANIFEST" --limit "$VLM_LIMIT"

mapfile -t support_plan_ids < <(python -c 'import json,sys; [print(json.loads(line)["plan_id"]) for line in open(sys.argv[1]) if line.strip()]' "$VLM_SUPPORT")
exclude_args=()
for plan_id in "${support_plan_ids[@]}"; do
  exclude_args+=(--exclude-plan-id "$plan_id")
done

run_condition() {
  local name="$1"; shift
  python -m retrieval_app.scripts.run_qwen_graph \
    --manifest "$VLM_MANIFEST" --corpus-root "$CUBICASA_ROOT" \
    --output "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_PREFIX}_${name}.jsonl" \
    --representation spatial --limit "$VLM_LIMIT" "${exclude_args[@]}" "$@"
  python -m retrieval_app.scripts.evaluate_vlm_graph \
    --predictions "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_PREFIX}_${name}.jsonl" \
    --manifest "$VLM_MANIFEST" --corpus-root "$CUBICASA_ROOT" \
    --output "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_PREFIX}_${name}_silver_metrics.json"
  local review_dir="$PROJECT_DIR/outputs/vlm_graph/reviews/${VLM_RUN_PREFIX}_${name}"
  python -m retrieval_app.scripts.build_vlm_review \
    --predictions "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_PREFIX}_${name}.jsonl" \
    --manifest "$VLM_MANIFEST" --corpus-root "$CUBICASA_ROOT" \
    --output-dir "$review_dir" --limit "$VLM_LIMIT"
  python -m retrieval_app.scripts.render_qwen_review_graphs --review-dir "$review_dir"
  echo "HTML review bundle ($name): $review_dir/index.html"
}

run_condition direct_baseline --mode zero --prompt-version baseline --graph-context none
run_condition direct_cubicasa_fewshot --mode few --prompt-version cubicasa_fewshot_v1 --support-manifest "$VLM_SUPPORT" --graph-context none
run_condition silver_corrected_fewshot --mode few --prompt-version cubicasa_fewshot_v1 --support-manifest "$VLM_SUPPORT" --graph-context silver

echo "Completed CubiCasa/CubiGraph pipeline and three Qwen conditions. The HTML bundles compare raw images, silver relation SVGs, Qwen JSON, and rendered Qwen graphs."
