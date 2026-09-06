#!/usr/bin/env bash
# Build the Qwen A100 Python environment once, on an allocated GPU node.
# It deliberately does not use ~/.venv or the login node's limited quota.
#SBATCH --job-name=bootstrap-qwen-a100
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/bootstrap-qwen-a100-%j.out
#SBATCH --error=slurm/logs/bootstrap-qwen-a100-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
READY_FILE="$VENV_DIR/.ready"

mkdir -p "$REPO_DIR/slurm/logs" "$CACHE_ROOT/pip"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"

if [[ -f "$READY_FILE" ]]; then
  echo "Ready environment already exists: $VENV_DIR"
  exit 0
fi
if [[ -e "$VENV_DIR" ]]; then
  echo "Environment exists but is not marked ready: $VENV_DIR" >&2
  echo "Inspect it before removing or retrying; refusing to overwrite it." >&2
  exit 2
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Qwen's current Transformers processor requires PyTorch >= 2.5.  Keep the
# CUDA 12.1 pair explicit so the persistent A100 environment is reproducible.
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1

cd "$REPO_DIR"
python -m pip install --no-cache-dir -r requirements_vlm_graph.txt
python - <<'PY'
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
assert torch.cuda.is_available(), "CUDA is not available"
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("Transformers/Qwen utilities import successfully")
PY
touch "$READY_FILE"
echo "READY: $VENV_DIR"
