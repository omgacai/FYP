#!/usr/bin/env bash
# Direct Qwen floorplan-image-to-graph experiment. Submit from ~/vlm/code/FYP.
# Override VLM_MANIFEST, VLM_MODE, VLM_REPRESENTATION, VLM_PROMPT_VERSION,
# VLM_SUPPORT, VLM_LIMIT, VLM_RUN_NAME, VLM_SHARD_INDEX, and VLM_NUM_SHARDS
# with sbatch --export.
#SBATCH --job-name=qwen3vl-graph
#SBATCH --partition=gpu
# Let Slurm choose any eligible GPU node; specific hosts may move partitions.
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/qwen3vl-graph-%j.out
#SBATCH --error=slurm/logs/qwen3vl-graph-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
VLM_MODE="${VLM_MODE:-zero}"
VLM_REPRESENTATION="${VLM_REPRESENTATION:-semantic}"
VLM_PROMPT_VERSION="${VLM_PROMPT_VERSION:-baseline}"
VLM_LIMIT="${VLM_LIMIT:-20}"
VLM_RUN_NAME="${VLM_RUN_NAME:-qwen3vl8b_${VLM_MODE}_${VLM_LIMIT}}"
VLM_MANIFEST="${VLM_MANIFEST:-$PROJECT_DIR/data/cubicasa5k/manifests/cubicasa_cubigraph_v1_20.jsonl}"
VLM_SHARD_INDEX="${VLM_SHARD_INDEX:-0}"
VLM_NUM_SHARDS="${VLM_NUM_SHARDS:-1}"

if [[ "$VLM_MODE" != "zero" && "$VLM_MODE" != "few" ]]; then
  echo "VLM_MODE must be zero or few" >&2
  exit 2
fi
if [[ "$VLM_REPRESENTATION" != "semantic" && "$VLM_REPRESENTATION" != "spatial" ]]; then
  echo "VLM_REPRESENTATION must be semantic or spatial" >&2
  exit 2
fi
if [[ "$VLM_MODE" == "few" && -z "${VLM_SUPPORT:-}" ]]; then
  echo "Few-shot mode requires VLM_SUPPORT=/absolute/path/to/support.jsonl" >&2
  exit 2
fi
if [[ "$VLM_PROMPT_VERSION" == "cubicasa_fewshot_v1" && "$VLM_MODE" != "few" ]]; then
  echo "cubicasa_fewshot_v1 requires VLM_MODE=few and verified VLM_SUPPORT" >&2
  exit 2
fi
if [[ ! -f "$VLM_MANIFEST" ]]; then
  echo "Missing VLM_MANIFEST: $VLM_MANIFEST" >&2
  exit 2
fi

mkdir -p "$REPO_DIR/slurm/logs" "$PROJECT_DIR/outputs/vlm_graph" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$VENV_DIR/.ready" ]]; then
  echo "Missing ready A100 environment: $VENV_DIR/.ready" >&2
  echo "Submit slurm/bootstrap_qwen_a100_env.sbatch first." >&2
  exit 2
fi

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not visible in this job"
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

args=(
  --manifest "$VLM_MANIFEST"
  --corpus-root "$PROJECT_DIR/data/cubicasa5k"
  --output "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_NAME}.jsonl"
  --mode "$VLM_MODE"
  --representation "$VLM_REPRESENTATION"
  --prompt-version "$VLM_PROMPT_VERSION"
  --limit "$VLM_LIMIT"
  --shard-index "$VLM_SHARD_INDEX"
  --num-shards "$VLM_NUM_SHARDS"
)
if [[ "$VLM_MODE" == "few" ]]; then
  args+=(--support-manifest "$VLM_SUPPORT")
fi

python -m retrieval_app.scripts.run_qwen_graph "${args[@]}"
python -m retrieval_app.scripts.evaluate_vlm_graph \
  --predictions "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_NAME}.jsonl" \
  --manifest "$VLM_MANIFEST" \
  --corpus-root "$PROJECT_DIR/data/cubicasa5k" \
  --output "$PROJECT_DIR/outputs/vlm_graph/${VLM_RUN_NAME}_silver_metrics.json"
