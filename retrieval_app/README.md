# Floorplan Retrieval Lab

Separate Streamlit dashboard for the FYP retrieval experiment. It does not modify the existing parser inspector.

```text
buyer query
  -> CLIP visual score (optional)
  + typed room-graph score
  + geometry/count score
  -> transparent fused ranking with per-plan evidence
```

## Install and run

```bash
cd /Users/numkoos/Downloads/FYP
python3 -m pip install -r requirements_retrieval.txt
python3 -m streamlit run retrieval_app/app.py
```

## Build a CubiCasa image/SVG manifest

The corpus builder only creates source image/SVG records. It does not silently invent graph labels.

```bash
python3 -m retrieval_app.scripts.build_cubicasa_manifest \
  --cubicasa-root /path/to/cubicasa5k \
  --output /path/to/cubicasa5k/manifests/cubicasa_image_svg.jsonl \
  --limit 20
```

After running CubiGraph, add these fields to each record:

```json
{
  "plan_id": "cubicasa-example",
  "image_path": "colorful/example/F1_scaled.png",
  "svg_path": "colorful/example/model.svg",
  "graph_path": "derived/cubigraph_v1/cubicasa-example.json",
  "graph_provenance": "silver",
  "room_counts": {"bedroom": 3, "bathroom": 2},
  "geometry": {"aspect_ratio": 1.23}
}
```

`gold` graph provenance is reserved for a manually verified/dataset-verified topology graph. A CubiGraph result from SVG, parser, or VLM output is `silver`.

## Derive a versioned CubiGraph corpus

The graph indexer reads the source manifest, writes graph JSON/SVG files under `derived/`, and emits a **new** enriched manifest. Start with 20 plans to inspect the output before processing all 5K.

```bash
python3 -m retrieval_app.scripts.index_cubicasa_graphs \
  --manifest /path/to/cubicasa5k/manifests/cubicasa_image_svg.jsonl \
  --corpus-root /path/to/cubicasa5k \
  --cubigraph-repo /path/to/FYP/third_party/CubiGraph5K \
  --graph-dir /path/to/cubicasa5k/derived/cubigraph_v1 \
  --output /path/to/cubicasa5k/manifests/cubicasa_cubigraph_v1_20.jsonl \
  --limit 20
```

The enriched manifest labels its graphs as `silver`: they are derived from the official SVG using deterministic CubiGraph rules, not independently human-verified graph annotations.

## Current and future components

| Component | Current baseline | Replace later with |
|---|---|---|
| Visual | OpenCLIP ViT-B/32 image-text cosine score | domain-fine-tuned ViT |
| Graph | Typed relation constraint matching | R-GCN/GAT graph encoder |
| Geometry | Room-count constraint match | geometry MLP / learned embedding |
| Fusion | User-controlled score weights | query-conditioned learned gate |
| Query parser | Inspectable regex baseline | evaluated LLM-to-JSON parser |

Train a learned component only after the corresponding unimodal and score-fusion baselines are measured on a frozen validation split.

## Qwen zero/few-shot image-to-graph experiment

This is a distinct parser experiment, not the buyer-query retrieval model:

```text
floorplan image -> Qwen -> small semantic room graph JSON
floorplan image -> parser/SVG -> CubiGraph -> silver reference graph JSON
```

The direct Qwen condition intentionally produces **JSON, not SVG**. It separates
semantic/topological reasoning from vectorisation quality. Do not use a test image
as a few-shot support example.

### Output schema

```json
{
  "rooms": [{"id": "bedroom_1", "type": "bedroom"}],
  "edges": [{"source": "bedroom_1", "target": "corridor_1", "type": "connected_by_door", "confidence": 0.86}]
}
```

Supported relations are `adjacent_to` and `connected_by_door`. The runner saves
the raw model output alongside the validated JSON; malformed answers are retained
and count against valid-JSON rate.

### Cluster install (Qwen3-VL 8B)

First inspect storage. The model is large, so keep Hugging Face files out of the
login-home quota. Replace `~/aigc-storage` only if your quota check confirms it is
the larger filesystem.

```bash
df -h ~/vlm ~/aigc-storage
mkdir -p ~/aigc-storage/fyp-model-cache/huggingface ~/vlm/outputs/vlm_graph
export HF_HOME=~/aigc-storage/fyp-model-cache/huggingface
export TOKENIZERS_PARALLELISM=false

cd ~/vlm/code/FYP
source ~/vlm/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_vlm_graph.txt
```

Sync this code from the Mac before running the commands below. Use a GPU allocation
on the cluster; do **not** run the 8B model on `xlogin`.

### Zero-shot smoke test

```bash
cd ~/vlm/code/FYP
source ~/vlm/.venv/bin/activate
export HF_HOME=~/aigc-storage/fyp-model-cache/huggingface

# Run this inside an allocated GPU shell, with the same model and settings for every plan.
python -m retrieval_app.scripts.run_qwen_graph \
  --manifest ~/vlm/data/cubicasa5k/manifests/cubicasa_cubigraph_v1_20.jsonl \
  --corpus-root ~/vlm/data/cubicasa5k \
  --output ~/vlm/outputs/vlm_graph/qwen3vl8b_zero_20.jsonl \
  --mode zero \
  --limit 20
```

The first run downloads `Qwen/Qwen3-VL-8B-Instruct` into `$HF_HOME`. The runner
resumes automatically if the output JSONL already exists; add `--overwrite` only
when deliberately replacing a condition.

For the SoC A100-80 environment, an equivalent Slurm job is included at
`slurm/qwen3vl_graph.sbatch`. Validate the GPU request, then submit it:

```bash
cd ~/vlm/code/FYP
mkdir -p slurm/logs
bash -n slurm/qwen3vl_graph.sbatch
sbatch --test-only slurm/qwen3vl_graph.sbatch
sbatch --export=ALL,VLM_MODE=zero,VLM_LIMIT=20,VLM_RUN_NAME=qwen3vl8b_zero_20 \
  slurm/qwen3vl_graph.sbatch
```

Monitor the submitted ID (replace `JOB_ID`):

```bash
squeue -j JOB_ID
tail -f slurm/logs/qwen3vl-graph-JOB_ID.out
```

### Few-shot condition

Create a **separate, manually verified** support JSONL using the template
`retrieval_app/data/vlm_support_example.jsonl`. Use one or two support plans not
among the 20 evaluation plans. Then run the same test images and decoding settings:

```bash
python -m retrieval_app.scripts.run_qwen_graph \
  --manifest ~/vlm/data/cubicasa5k/manifests/cubicasa_cubigraph_v1_20.jsonl \
  --corpus-root ~/vlm/data/cubicasa5k \
  --output ~/vlm/outputs/vlm_graph/qwen3vl8b_few2_20.jsonl \
  --mode few \
  --support-manifest ~/vlm/data/cubicasa5k/manifests/vlm_support_2_gold.jsonl \
  --limit 20
```

### Evaluate against CubiGraph candidates

```bash
python -m retrieval_app.scripts.evaluate_vlm_graph \
  --predictions ~/vlm/outputs/vlm_graph/qwen3vl8b_zero_20.jsonl \
  --manifest ~/vlm/data/cubicasa5k/manifests/cubicasa_cubigraph_v1_20.jsonl \
  --corpus-root ~/vlm/data/cubicasa5k \
  --output ~/vlm/outputs/vlm_graph/qwen3vl8b_zero_20_metrics.json
```

This reports valid JSON rate, room-count error, and typed edge precision/recall/F1.
Because CubiGraph labels are rule-derived, call these **silver-label metrics** until
you manually verify a held-out subset.
