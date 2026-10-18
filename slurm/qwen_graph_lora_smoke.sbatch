#!/usr/bin/env bash
# One short LoRA smoke run. It validates the full raster -> graph SFT path.
#SBATCH --job-name=qwen-graph-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/qwen-graph-lora-%j.out
#SBATCH --error=slurm/logs/qwen-graph-lora-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/aigc-storage/fyp-model-cache}"
CUBICASA_ROOT="${CUBICASA_ROOT:?Set CUBICASA_ROOT to the CubiCasa corpus root}"
SFT_DATASET_DIR="${SFT_DATASET_DIR:-$PROJECT_DIR/data/qwen_graph_sft/smoke100}"
SFT_RUN_NAME="${SFT_RUN_NAME:-smoke100_lora_v1}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-$PROJECT_DIR/outputs/qwen_graph_lora/$SFT_RUN_NAME}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-20}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-5}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-5}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
SFT_EVAL_STRATEGY="${SFT_EVAL_STRATEGY:-steps}"
SFT_SAVE_STRATEGY="${SFT_SAVE_STRATEGY:-steps}"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
[[ -f "$SFT_DATASET_DIR/train.jsonl" && -f "$SFT_DATASET_DIR/val.jsonl" ]] || { echo "Missing SFT train/val data under $SFT_DATASET_DIR" >&2; exit 2; }
[[ ! -e "$SFT_OUTPUT_DIR" ]] || { echo "Output already exists: $SFT_OUTPUT_DIR; choose a new SFT_RUN_NAME" >&2; exit 2; }

mkdir -p "$REPO_DIR/slurm/logs" "$CACHE_ROOT/huggingface"
export HF_HOME="$CACHE_ROOT/huggingface" TOKENIZERS_PARALLELISM=false
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

python - <<'PY'
import datasets, peft, tensorboard, torch
import transformers
assert torch.cuda.is_available(), "CUDA is not visible"
print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__, "peft:", peft.__version__, "datasets:", datasets.__version__, "tensorboard:", tensorboard.__version__)
PY

python -m retrieval_app.scripts.train_qwen_graph_lora \
  --dataset-dir "$SFT_DATASET_DIR" \
  --corpus-root "$CUBICASA_ROOT" \
  --output-dir "$SFT_OUTPUT_DIR" \
  --epochs "$SFT_EPOCHS" \
  --max-steps "$SFT_MAX_STEPS" \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --gradient-accumulation-steps "$SFT_GRADIENT_ACCUMULATION_STEPS" \
  --eval-strategy "$SFT_EVAL_STRATEGY" \
  --save-strategy "$SFT_SAVE_STRATEGY" \
  --eval-steps "$SFT_EVAL_STEPS" \
  --save-steps "$SFT_SAVE_STEPS" \
  --max-pixels 501760
