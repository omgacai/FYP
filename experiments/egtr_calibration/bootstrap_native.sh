#!/usr/bin/env bash
# Candidate compatibility environment for SOC CUDA 12; run on allocated A100.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SLURM_JOB_ID:?Run inside a Slurm GPU allocation}"
EGTR_STORAGE="${EGTR_STORAGE:-$HOME/aigc-storage}"
EGTR_ENV="$EGTR_STORAGE/fyp-envs/egtr-py310-cu121"
DOWNLOAD_ENV="$EGTR_STORAGE/fyp-envs/download-tools"
export UV_PYTHON_INSTALL_DIR="$EGTR_STORAGE/fyp-envs/managed-python"
export UV_CACHE_DIR="$EGTR_STORAGE/fyp-model-cache/uv"
export HF_HOME="$EGTR_STORAGE/fyp-model-cache/huggingface"
export PIP_CACHE_DIR="$EGTR_STORAGE/fyp-model-cache/pip"
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
mkdir -p "$EGTR_STORAGE/fyp-model-cache/egtr/setup-logs"
LOG="$EGTR_STORAGE/fyp-model-cache/egtr/setup-logs/native-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
printf 'Setup log: %s\n' "$LOG"
nvidia-smi --query-gpu=name,driver_version --format=csv
nvcc --version
"$CXX" --version
bash experiments/egtr_calibration/fetch_sources.sh
"$DOWNLOAD_ENV/bin/python" -m pip install --no-cache-dir uv
"$DOWNLOAD_ENV/bin/uv" python install 3.10
if [[ ! -d "$EGTR_ENV" ]]; then
  "$DOWNLOAD_ENV/bin/uv" venv --python 3.10 --seed "$EGTR_ENV"
fi
"$EGTR_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
"$EGTR_ENV/bin/python" -m pip install --no-cache-dir \
  'numpy==1.26.4' 'setuptools<81' wheel
"$EGTR_ENV/bin/python" -m pip install --no-cache-dir \
  torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
"$EGTR_ENV/bin/python" -m pip install --no-cache-dir \
  transformers==4.18.0 tokenizers==0.12.1 huggingface-hub==0.25.2 \
  timm==0.5.4 scipy==1.11.4 pillow==9.5.0 ninja==1.11.1.1 \
  beautifulsoup4 lxml shapely
"$EGTR_ENV/bin/python" -m pip check
"$EGTR_ENV/bin/python" -m pip freeze > "$EGTR_ENV/requirements-resolved.txt"
cat > "$EGTR_ENV/runtime.sh" <<'ENV'
export EGTR_PYTHON="$HOME/aigc-storage/fyp-envs/egtr-py310-cu121/bin/python"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.0
export HF_HOME="$HOME/aigc-storage/fyp-model-cache/huggingface"
ENV
printf 'export EGTR_PYTHON=%q\nexport HF_HOME=%q\n' "$EGTR_ENV/bin/python" "$HF_HOME" >> "$EGTR_ENV/runtime.sh"
"$EGTR_ENV/bin/python" - <<'PY'
import sys
import torch
import transformers
from torch.utils.cpp_extension import CUDA_HOME
print('Python:', sys.version)
print('Torch:', torch.__version__, 'runtime CUDA:', torch.version.cuda)
print('Transformers:', transformers.__version__, 'CUDA_HOME:', CUDA_HOME)
assert torch.cuda.is_available(), 'CUDA unavailable in this allocation'
print('GPU:', torch.cuda.get_device_name(0))
sys.path.insert(0, 'third_party/egtr')
from model import deformable_detr
from model.egtr import DetrForSceneGraphGeneration
assert deformable_detr.MultiScaleDeformableAttention is not None, 'EGTR custom CUDA extension failed to load; inspect compiler log'
print('EGTR imports and custom CUDA extension: PASS')
PY
printf 'Environment checks passed. Full checkpoint forward pass remains to be tested.\n'
printf 'For later commands: source "%s/runtime.sh"\n' "$EGTR_ENV"
