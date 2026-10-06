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

## Generate deterministic QA candidates from manual graphs

The QA generator reads a directory of `floorplan-manual-graph/2` annotations and
writes a CSV of review-ready question/answer candidates. Answers are computed
from the reviewed graph and geometry; an LLM is not used. Each row includes the
supporting node IDs, edge IDs, and path so that it can be checked against the
floor-plan image before the QA set is frozen.

```bash
python3 -m retrieval_app.scripts.generate_manual_graph_qa \
  --input-dir cubicasa_benchmark/annotations \
  --output cubicasa_benchmark/questions/manual20_qa_v1.csv \
  --per-plan 5
```

The default target is five diverse questions per plan: D0 count, D1 direct
relation, D2 relative position, D3 compound access, and D4 shortest access
path. It does **not** turn absent edges into `no`, because the current Manual20
records have `all_pairs_reviewed: false`. Add `--include-unknown` only when you
want explicit incomplete-evidence examples. Every generated row has
`needs_human_review=true`; review and freeze the CSV before running QA models.

## Run and evaluate the Manual20 QA pilot

`run_qwen_qa.py` is separate from the image-to-graph experiment. It makes no
answer-repair calls: malformed output is saved and scored as invalid/wrong.
Keep Qwen model, prompt, pixel limit, decoding, and the frozen CSV identical
for every condition.

After syncing `cubicasa_benchmark/` to SOC, submit a four-way array for each
condition. This example assumes the bundle is at `~/vlm/data/manual20`:

```bash
cd ~/vlm/code/FYP
export QA_QUESTIONS=~/vlm/data/manual20/questions/manual20_qa_v1.csv
export QA_ANNOTATIONS_DIR=~/vlm/data/manual20/annotations
export QA_IMAGES_DIR=~/vlm/data/manual20/images
export QA_OUTPUT_DIR=~/vlm/outputs/manual20_qa/image_only
export QA_CONDITION=image_only
sbatch --array=0-3 slurm/qwen3vl_qa.sbatch
```

For the gold-graph upper bound, change only the input condition and output
location:

```bash
export QA_OUTPUT_DIR=~/vlm/outputs/manual20_qa/image_gold
export QA_CONDITION=image_graph
export QA_GRAPH_DIR="$QA_ANNOTATIONS_DIR"
export QA_GRAPH_SOURCE=gold_manual
sbatch --array=0-3 slurm/qwen3vl_qa.sbatch
```

For Qwen/CubiGraph/EGTR graph conditions, set `QA_GRAPH_DIR` to a directory of
canonical `<plan_id>.graph.json` files and change `QA_GRAPH_SOURCE`, then submit
the same array. The evaluator accepts shards directly:

```bash
python -m retrieval_app.scripts.evaluate_qa \
  --questions "$QA_QUESTIONS" \
  --predictions ~/vlm/outputs/manual20_qa/image_only/*.jsonl \
  --predictions ~/vlm/outputs/manual20_qa/image_gold/*.jsonl \
  --output-dir ~/vlm/outputs/manual20_qa/results
```

It writes `qa_metrics.json` and a per-question CSV. Its headline is plan-macro
exact accuracy; condition deltas use a paired bootstrap over plan identity.
`unknown` metrics are intentionally marked not estimable until reviewed
gold-unknown questions are present in the frozen CSV.

Evaluate graph extraction separately; do not describe these node/edge scores as
QA. The existing evaluator reports matched-room node precision/recall/F1,
room-type accuracy, `direct_access` and `adjacent_to` edge metrics, excluded
uncertain pairs, and aligned graph edit cost:

```bash
node review_react/scripts/evaluate-benchmark.mjs \
  --root ~/vlm/data/manual20 \
  --relation-mode access \
  --output ~/vlm/outputs/manual20_qa/graph_metrics.json
```

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
mkdir -p slurm/logs

# Never run `pip install torch` on xlogin. Build the CUDA PyTorch environment once
# inside the proven A100-80 Slurm allocation.
sbatch slurm/bootstrap_qwen_a100_env.sbatch
```

Wait for the bootstrap job to finish, then confirm this file exists:

```bash
test -f ~/aigc-storage/fyp-envs/qwen-a100-cu121/.ready && echo "Qwen environment ready"
```

Sync this code from the Mac before running the commands below. Use a GPU allocation
on the cluster; do **not** run the 8B model on `xlogin`.

### Zero-shot smoke test

```bash
cd ~/vlm/code/FYP
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
`slurm/qwen3vl_graph.sbatch`; it uses the ready environment created above.
Validate the GPU request, then submit it:

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

## Experiment 2: zero-shot spatial room graph

Experiment 1 asks Qwen for room types and typed edges only. Experiment 2 uses the
same `Qwen/Qwen3-VL-8B-Instruct` model and deterministic decoding, but requires a
normalised 1000×1000 canvas, approximate room bounding boxes, and centroids. This
makes a visual overlay possible; it is **not** a claim of precise SVG/vector output.

```bash
sbatch --nodelist=xgph1 --gres=gpu:a100-80:1 \
  --export=ALL,VLM_MODE=zero,VLM_REPRESENTATION=spatial,VLM_LIMIT=5,VLM_RUN_NAME=qwen3vl8b_spatial_zero_5 \
  slurm/qwen3vl_graph.sbatch
```

Use the same evaluator for room and relation agreement. Spatial-box accuracy needs
manual visual review or an explicitly designed SVG-derived geometry metric; do not
interpret the CubiGraph edge score as a coordinate-accuracy score.

## Build a manual Qwen-versus-CubiGraph review page

Generate a five-plan review bundle before treating either system as ground truth.
It displays the source floorplan, CubiGraph relation SVG, Qwen graph JSON, and raw
model answer side-by-side. It also creates a separate `review_notes.jsonl` template
for your manual gold decisions.

```bash
python -m retrieval_app.scripts.build_vlm_review \
  --predictions ~/vlm/outputs/vlm_graph/qwen3vl8b_zero_20.jsonl \
  --manifest ~/vlm/data/cubicasa5k/manifests/cubicasa_cubigraph_v1_20.jsonl \
  --corpus-root ~/vlm/data/cubicasa5k \
  --output-dir ~/vlm/outputs/vlm_graph/review_zero_5 \
  --limit 5
```

For Experiment 2, point `--predictions` at `qwen3vl8b_spatial_zero_5.jsonl`.
The review bundle adds a fourth panel: Qwen's predicted boxes/centroids and graph
edges overlaid directly on the original floorplan image.

To inspect locally, download the generated review directory or serve it through a
cluster Jupyter/HTTP tunnel. The bundle copies only the five selected images and
relation SVGs, so it is small and portable.
