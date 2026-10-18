#!/usr/bin/env bash
# Resumable CubiCasa source-SVG/CubiGraph silver-label build for Qwen SFT.
# Run on a verified x86 GPU node because the existing shared Python environment
# is x86-only. This script does not use the GPU for preprocessing.
#SBATCH --job-name=qwen-graph-data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/qwen-graph-data-%j.out
#SBATCH --error=slurm/logs/qwen-graph-data-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vlm}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/code/FYP}"
VENV_DIR="${VENV_DIR:-$HOME/aigc-storage/fyp-envs/qwen-a100-cu121}"
CUBICASA_ROOT="${CUBICASA_ROOT:-$PROJECT_DIR/data/cubicasa5k}"
CUBIGRAPH_REPO="${CUBIGRAPH_REPO:-$REPO_DIR/third_party/CubiGraph5K}"
# This user-owned, gitignored file may contain portable IDs or the older
# absolute local CubiCasa paths; the builders normalise either form.
EXCLUSIONS="${EXCLUSIONS:-$PROJECT_DIR/data/cubicasa5k/manual20_exclusion_manifest.json}"
DERIVED_ROOT="${DERIVED_ROOT:-$PROJECT_DIR/data/cubicasa5k/derived_full_v1}"
SFT_OUT="${SFT_OUT:-$PROJECT_DIR/data/qwen_graph_sft/full_v1}"

SOURCE_MANIFEST="$DERIVED_ROOT/source_manifest.jsonl"
ENRICHED_MANIFEST="$DERIVED_ROOT/cubigraph_manifest.jsonl"
GRAPH_DIR="$DERIVED_ROOT/cubigraph"
CANONICAL_CORPUS="$DERIVED_ROOT/canonical.jsonl"

[[ "$(uname -m)" == "x86_64" ]] || { echo "Qwen environment requires x86_64" >&2; exit 2; }
[[ -f "$VENV_DIR/.ready" ]] || { echo "Missing ready marker: $VENV_DIR/.ready" >&2; exit 2; }
[[ -d "$CUBICASA_ROOT" ]] || { echo "Missing CubiCasa root: $CUBICASA_ROOT" >&2; exit 2; }
[[ -d "$CUBIGRAPH_REPO" ]] || { echo "Missing CubiGraph repository: $CUBIGRAPH_REPO" >&2; exit 2; }
[[ -f "$EXCLUSIONS" ]] || { echo "Missing Manual20 exclusions: $EXCLUSIONS" >&2; exit 2; }

mkdir -p "$REPO_DIR/slurm/logs" "$DERIVED_ROOT"
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  python -m retrieval_app.scripts.build_cubicasa_manifest \
    --cubicasa-root "$CUBICASA_ROOT" --output "$SOURCE_MANIFEST"
fi

# The indexer resumes from cubigraph_manifest.jsonl.partial.jsonl if interrupted.
if [[ ! -f "$ENRICHED_MANIFEST" ]]; then
  python -m retrieval_app.scripts.index_cubicasa_graphs \
    --manifest "$SOURCE_MANIFEST" --corpus-root "$CUBICASA_ROOT" \
    --cubigraph-repo "$CUBIGRAPH_REPO" --graph-dir "$GRAPH_DIR" \
    --output "$ENRICHED_MANIFEST"
fi

if [[ ! -f "$CANONICAL_CORPUS" ]]; then
  python -m retrieval_app.scripts.build_cubicasa_egtr_corpus \
    --manifest "$ENRICHED_MANIFEST" --corpus-root "$CUBICASA_ROOT" \
    --cubigraph-repo "$CUBIGRAPH_REPO" --output "$CANONICAL_CORPUS" \
    --errors "$DERIVED_ROOT/canonical_errors.jsonl" \
    --exclude-identities "$EXCLUSIONS"
fi

if [[ -e "$SFT_OUT" && ! -f "$SFT_OUT/dataset_manifest.json" ]]; then
  echo "Incomplete SFT output exists: $SFT_OUT; inspect or choose another SFT_OUT" >&2
  exit 2
fi
if [[ ! -f "$SFT_OUT/dataset_manifest.json" ]]; then
  python -m retrieval_app.scripts.build_qwen_graph_sft_dataset \
    --canonical-corpus "$CANONICAL_CORPUS" --exclude-identities "$EXCLUSIONS" \
    --output-dir "$SFT_OUT"
fi

cat "$SFT_OUT/dataset_manifest.json"
