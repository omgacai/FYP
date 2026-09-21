#!/usr/bin/env bash
# Run inside an allocated x86_64 SOC GPU shell. Never installs packages.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cat >&2 <<'EOF'
This script must run inside a Slurm GPU allocation, not on an xlogin node.
First allocate the verified x86_64 A100 node:
  srun --partition=gpu --nodelist=xgph1 --gres=gpu:a100-80:1 --cpus-per-task=4 --mem=24G --time=01:00:00 --pty bash -l
Then run this script again from that allocated shell.
EOF
  exit 2
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  echo 'No NVIDIA GPU is visible in this Slurm allocation; do not run EGTR here.' >&2
  exit 2
fi
export PYTHONPATH="$HOME/aigc-storage/fyp-envs/egtr-deps-py312-v1"
export PYTHONNOUSERSITE=1
mkdir -p "$HOME/vlm/outputs/egtr/logs"
smoke_log=$(mktemp "$HOME/vlm/outputs/egtr/logs/smoke.XXXXXXXX.log")
echo "Live log: $smoke_log"
"$HOME/aigc-storage/fyp-envs/qwen-a100-cu121/bin/python" -u \
  experiments/egtr_calibration/smoke_checkpoint.py "$@" 2>&1 | tee "$smoke_log"
