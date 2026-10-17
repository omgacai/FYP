# Qwen raster-to-graph LoRA: silver-label preparation

This experiment fine-tunes a Qwen vision-language model to emit a compact room
graph from a **raster floor-plan image only**. Source `model.svg` and CubiGraph
rules provide silver labels; they are never model inputs and must never be used
as predictions on Manual20.

## Data contract

1. `build_cubicasa_egtr_corpus.py` creates identity-split, source-SVG room
   boxes and CubiGraph silver edges.
2. `build_qwen_graph_sft_dataset.py` converts that corpus to deterministic Qwen
   chat messages. It emits `train.jsonl`, `val.jsonl`, error/exclusion records,
   and a reproducibility manifest.
3. Every identity in `cubicasa_eval/exclusion_manifest.json` is excluded from
   train and validation. Manual20 is therefore a held-out evaluation set.

The current labels are silver: room boxes/types come from source SVG; graph
edges come from rules. In particular, open connections may be missing and a
missing edge is not a verified negative.

## SOC smoke-test commands

Run these on the SOC login node only for a 20-record validation of the data
contract; use a batch job for the full 5K corpus.

```bash
cd ~/vlm/code/FYP
source ~/aigc-storage/fyp-envs/qwen-a100-cu121/bin/activate

export DERIVED_ROOT=~/vlm/data/cubicasa5k/derived
export CANONICAL_CORPUS="$DERIVED_ROOT/cubicasa_egtr_canonical.jsonl"
export EXCLUSIONS=~/vlm/code/FYP/cubicasa_eval/exclusion_manifest.json
export SFT_OUT=~/vlm/data/qwen_graph_sft/smoke20

python -m retrieval_app.scripts.build_qwen_graph_sft_dataset \
  --canonical-corpus "$CANONICAL_CORPUS" \
  --exclude-identities "$EXCLUSIONS" \
  --output-dir "$SFT_OUT" \
  --limit 20

cat "$SFT_OUT/dataset_manifest.json"
sed -n '1p' "$SFT_OUT/train.jsonl"
```

Do not start LoRA until this smoke dataset has nonzero `train` and `val` rows,
zero errors, and manually inspected targets. The next implementation step is a
trainer adapter for the selected Qwen fine-tuning framework, after this dataset
contract is frozen.
