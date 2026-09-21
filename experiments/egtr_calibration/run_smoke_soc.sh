#!/usr/bin/env bash
# Run inside an allocated x86_64 SOC GPU shell. Never installs packages.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$HOME/aigc-storage/fyp-envs/egtr-deps-py312-v1"
export PYTHONNOUSERSITE=1
mkdir -p "$HOME/vlm/outputs/egtr/logs"
smoke_log=$(mktemp "$HOME/vlm/outputs/egtr/logs/smoke.XXXXXXXX.log")
echo "Live log: $smoke_log"
"$HOME/aigc-storage/fyp-envs/qwen-a100-cu121/bin/python" -u \
  experiments/egtr_calibration/smoke_checkpoint.py "$@" 2>&1 | tee "$smoke_log"
