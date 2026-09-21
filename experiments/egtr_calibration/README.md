# EGTR experiments: frozen transfer first

**Current runtime:** reuse the existing Qwen PyTorch environment on a verified x86-64 GPU node (confirmed xgph1). Do not reinstall Torch. See the confirmed fix below; earlier bootstrap guidance is superseded.

Feed CubiCasa raster images to a released **full EGTR checkpoint**, inspect its native objects/relations, and map only defensible labels to the floor-plan ontology. Run three calibration plans first; freeze settings before a held-out benchmark run. The released model was trained on Visual Genome/Open Images, not room-access graphs. Unsupported ontology is an experimental outcome, not an ordinary empty-graph prediction. Floor-plan fine-tuning is a separate later experiment.

Primary relations: **direct_access** (door OR open passage) and **adjacent_to** (boundary without direct access). Raw door/open annotations stay unchanged. The graph evaluator and QA serializer apply the same collapse to references and predictions. No separate open-passage prediction head is required.

## Current data and execution status

The local calibration snapshot is `cubicasa_eval/calibration/egtr_v2`:

| Source plan | Rooms | Silver edges |
|---|---:|---:|
| high_quality/14343 | 12 | 12 |
| high_quality/1900 | 10 | 8 |
| high_quality/1097 | 8 | 7 |

The v2 reference geometry uses SVG polygon coordinates directly in F1_scaled pixel space, matching the CubiCasa loader. The earlier v1 snapshot incorrectly rescaled from the SVG viewport and is superseded; do not score against v1.

These plans were selected deterministically outside the 20 user-specified excluded folders. Their images and source SVGs are saved locally. References use source-SVG rooms plus CubiGraph silver edges, not CNN predictions or manually verified topology. The source rules miss open passages; merged access labels do not fix those missing positives. Unreviewed absent pairs remain excluded from negative scoring.

One frozen EGTR checkpoint forward pass succeeded on the first manual raster on SOC xgph1 (A100 80GB): logits `(1, 200, 150)`, boxes `(1, 200, 4)`, relations `(1, 200, 200, 50)` and connectivity `(1, 200, 200, 1)`. This validates checkpoint loading and raw tensor generation only; it is not a graph result. Full 20-plan inference and vocabulary inspection remain next. The local Mac has no CUDA. SOC access currently requires the user's terminal.

## One directory per run

```text
egtr_runs/
  experiments.csv                         # regenerated cross-run comparison
  20260921T..._frozen_vg_001/
    run.json                              # intent, dataset identity, input hashes
    status.json                           # per-stage status and exit codes
    manifest.json
    images/                               # exact raster snapshot
    annotations/                          # original reference labels
    questions.jsonl                       # frozen questions, if available
    raw/<plan>.pt                         # complete logits/boxes/relations/connectivity
    raw/<plan>.json                       # objects, top 100 triplets, provenance, time/errors
    predictions/egtr_frozen/<plan>.json    # adapted graph or explicit unavailable outcome
    overlays/                             # separate reference/raw/adapted visualizations
    results/graph_metrics.json            # per-plan + aggregate graph metrics and coverage
    results/qa.jsonl                       # paired QA answers/prompts, when run
    logs/<stage>.log                       # console output, streamed live and saved
    provenance/<stage>/                   # exact commands, code copies/hashes, package list
    summary.json                          # settings, coverage, metrics, paired QA deltas
```

Run data are gitignored. Back up the run folders separately; Git stores the pipeline, not your experiment results. Weights remain in one external checkpoint directory; runs record their SHA-256 instead of duplicating large checkpoints.

New run IDs are unique even with the same name. Initialization copies inputs and verifies image/reference hashes. Every stage rechecks the input snapshot. Stages are single-attempt to prevent mixing configurations or overwriting results: use a new run for changed parameters or retries. Different runs can execute independently; a `.stage-lock` prevents simultaneous writes within one run. If a scheduler forcibly kills a process, the saved state may remain running; inspect the Slurm outcome and preserve that interrupted run before starting a new one.

## Prepare the cluster (no Apptainer required)

Use a dedicated native Python/Conda environment for legacy EGTR; keep it separate from modern Qwen. Inspect Python, CUDA toolkit (`nvcc`), compiler and modules before choosing installation commands. A CUDA driver alone does not provide the compiler required by the deformable-attention extension. Upstream targets PyTorch 1.12.1/CUDA 11.3 and Transformers 4.18.0; this cluster setup is not validated yet. Install/build on an allocated compute node, not xlogin.

Fetch pinned upstream repositories from the FYP root:

```bash
bash experiments/egtr_calibration/fetch_sources.sh
```

Download the **full trained EGTR checkpoint** and matching `config.json` from the [official repository](https://github.com/naver-ai/egtr). `labels.json` must contain `objects` and `relations` arrays in exact model output order, without background entries. For Visual Genome, object indices are COCO category IDs minus one, and predicates are `rel_categories[1:]`. The runner checks dimensions; correct semantic ordering still requires the matching original vocabulary.

To regenerate calibration data, set `CUBICASA_DATASET` to the directory containing `colorful/` and `high_quality/`. Install Pillow, numpy, beautifulsoup4, lxml and shapely in the preparation environment, then:

```bash
python experiments/egtr_calibration/prepare.py \
  --dataset "$CUBICASA_DATASET" \
  --exclusions experiments/egtr_calibration/exclusions.json \
  --output cubicasa_eval/calibration/egtr_v2
```

The exclusion file contains portable category/plan identities, covering every image variant. A different candidate inventory can produce a different deterministic selection: check `selection.json` against the table. Keep all selected calibration identities outside future final testing. `git pull` does not transfer datasets.

## Create and run an experiment

On the allocated xgph1 shell, first extract the checkpoint's verified VG vocabulary and initialize an immutable manual-20 run. This is the next execution step after the smoke test:

```bash
cd ~/vlm/code/FYP
export EGTR_PYTHON="$HOME/aigc-storage/fyp-envs/qwen-a100-cu121/bin/python"
export PYTHONPATH="$HOME/aigc-storage/fyp-envs/egtr-deps-py312-v1"
export EGTR_CONFIG=$(find "$HOME/aigc-storage/fyp-model-cache/egtr/vg/artifact" -name config.json -print -quit)
export EGTR_ARTIFACT=$(dirname "$EGTR_CONFIG")
export EGTR_CHECKPOINT=$(find "$HOME/aigc-storage/fyp-model-cache/egtr/vg/artifact" -name '*.ckpt' -print -quit)
export EGTR_LABELS="$HOME/aigc-storage/fyp-model-cache/egtr/vg/labels.json"
"$EGTR_PYTHON" experiments/egtr_calibration/extract_vg_labels.py \
  --archive "$HOME/aigc-storage/fyp-model-cache/egtr/vg/metadata/vg.zip" --output "$EGTR_LABELS"
export EGTR_RUN=$(python3 experiments/egtr_calibration/runs.py init \
  --data cubicasa_eval/manual20_v1 --runs "$HOME/vlm/outputs/egtr/runs" \
  --name manual20_frozen_vg --note "Frozen VG EGTR; exploratory manual-20 transfer run")
"$EGTR_PYTHON" experiments/egtr_calibration/runs.py stage infer --run "$EGTR_RUN" \
  --artifact "$EGTR_ARTIFACT" --checkpoint "$EGTR_CHECKPOINT" --labels "$EGTR_LABELS"
```

The stage streams its output, writes the same output to `$EGTR_RUN/logs/infer.log`, saves every raw model tensor, and creates one JSON record per plan. It does not yet map Visual Genome concepts to rooms or report a graph score.

From the FYP root (initialization requires only standard Python):

```bash
export EGTR_RUN=$(python3 experiments/egtr_calibration/runs.py init \
  --data cubicasa_eval/calibration/egtr_v2 \
  --name frozen_vg_001 \
  --note "Released VG checkpoint; three-plan transfer feasibility")
```

Review questions against the raster **before** creating a run if QA is planned. Generated questions are drafts; change `review_status` to `human_reviewed` only after review. Nine count questions alone do not test topology sensitivity. Add relation/compound questions with checked evidence. A changed question set needs a new run.

Set `EGTR_PYTHON`, `EGTR_ARTIFACT`, `EGTR_CHECKPOINT`, `EGTR_LABELS` to actual paths in your SOC setup; do not paste placeholder paths. Then submit:

```bash
sbatch --output="$EGTR_RUN/logs/slurm-%j.out" \
  --error="$EGTR_RUN/logs/slurm-%j.err" \
  experiments/egtr_calibration/run_egtr.sbatch
```

The job calls the tracked inference stage. Native Python and Slurm are sufficient; resource requests may need adapting to your allocation. Watch the scheduler log or, after inference starts:

```bash
tail -f "$EGTR_RUN/logs/infer.log"
```

`Ctrl-C` stops following the log, not the Slurm job. Check the queue with `squeue -u "$USER"`.

## Inspect, map, evaluate

Inspect `raw/*.json` and checkpoint vocabularies before populating the mapping. The shipped mapping is intentionally empty. Entries take `{ "target": "direct_access", "justification": "Evidence supporting this mapping" }` under an actual source predicate name, with analogous room mappings under `objects`. Never map `near` to access or a furniture object to a room. If no defensible mapping exists, record unsupported coverage and do not interpret a graph score as floor-plan ability.

Run adaptation in the EGTR environment after inference:

```bash
"$EGTR_PYTHON" experiments/egtr_calibration/runs.py stage adapt \
  --run "$EGTR_RUN" \
  --mapping experiments/egtr_calibration/ontology_mapping.json \
  --object-threshold 0.3 --relation-threshold 0.01

python3 experiments/egtr_calibration/runs.py stage evaluate \
  --run "$EGTR_RUN" --min-iou 0.3

"$EGTR_PYTHON" experiments/egtr_calibration/runs.py stage preview \
  --run "$EGTR_RUN"
```

Evaluation requires Node.js. Thresholds above are provisional calibration values. Inspect class-blind spatial matching. Relation scores multiply subject/object confidence, relation probability and auxiliary connectivity; the highest supported score across directions/types wins for each unordered pair. The auxiliary connectivity output does not itself mean physical access.

Metrics include node/typed-edge precision, recall, F1, room type accuracy per plan, per-relation scores, aligned edit cost, excluded pairs and explicit missing/invalid/unavailable coverage. Aggregate metrics cover evaluated plans only; report coverage with them. Source silver labels measure agreement with the pipeline, not manual-gold accuracy. Reference kind stays explicit.

## QA and comparing runs

Use the fixed Qwen3-VL checkpoint in its separate environment, on an allocated GPU:

```bash
python experiments/egtr_calibration/runs.py stage qa \
  --run "$EGTR_RUN" --model YOUR_QWEN3_VL_MODEL --revision YOUR_PINNED_REVISION
```

The three arms are `image_only`, `reference_graph`, `egtr_graph`. Prompts never include accepted answers. Missing/unsupported EGTR graphs produce unavailable records, not image-only substitutes. Summaries report arm coverage and QA gain/gap only on question IDs successfully answered in all three arms. Pipeline references are never silently renamed manual gold.

Regenerate summaries and the comparison CSV:

```bash
python3 experiments/egtr_calibration/runs.py list
```

The CSV includes checkpoint/input hashes, thresholds, reference kind, stage status, graph coverage and graph metrics. Detailed QA comparisons remain in each `summary.json`. Compare runs on the same inputs and reference protocol. Three plans establish feasibility, not reliable generalization.

## Software checks

```bash
python -m unittest experiments.egtr_calibration.test_pipeline experiments.egtr_calibration.test_runs
node --test review_react/src/graphEvaluation.test.js
```

These use synthetic test fixtures; no scores from them are EGTR benchmark results.

### Earlier SOC native setup candidate — superseded; do not rerun

User verified xgpj0 has an A100 80GB, driver 580.178.04, nvcc 12.0, GCC 13 plus GCC/G++ 12, Python 3.12, and no older Python on PATH. The checkpoint archive is downloaded; its config uses placeholder class names. `bootstrap_native.sh` creates a separate managed Python 3.10 environment with PyTorch 2.1.2/cu121, torchvision 0.16.2, Transformers 4.18.0 and GCC 12. This differs from the original EGTR environment and is a compatibility candidate, not a validated reproduction. The script logs installation, checks CUDA availability and requires the custom extension to load successfully. A full checkpoint forward pass remains required after it passes.

Inside the allocated A100 shell, from the FYP root:

```bash
python3 experiments/egtr_calibration/extract_vg_labels.py \
  --archive "$HOME/aigc-storage/fyp-model-cache/egtr/vg/metadata/vg.zip" \
  --output "$HOME/aigc-storage/fyp-model-cache/egtr/vg/labels.json"
bash experiments/egtr_calibration/bootstrap_native.sh
```

The label extractor reads the actual official annotation archive, checks index ranges and drops only its background predicate. The downloader environment supplies uv to install Python 3.10 into persistent storage without sudo. Bootstrap logs are saved under `fyp-model-cache/egtr/setup-logs/`. Afterwards, source the printed `runtime.sh` to preserve compiler/cache settings for inference and Slurm submission.


## Confirmed SOC PyTorch fix — 2026-09-21

Reuse `~/aigc-storage/fyp-envs/qwen-a100-cu121/bin/python` on an **x86-64** GPU node. The user confirmed **PyTorch 2.5.1+cu121**, **xgph1**, **NVIDIA A100 80GB PCIe**, and CUDA tensor output `[2.0, 4.0, 6.0]`. No Torch installation was required. The user explicitly requests **no Torch reinstallation**.

**xgpj0 is aarch64 (ARM)**. It cannot load the x86-64 Torch binaries: import reported missing `libtorch_global_deps.so` even though the file existed, `file` identified x86-64 ELF, and `ldd` said `not a dynamic executable`. Check `uname -m` before diagnosing package corruption. Requesting an A100 GPU alone does not guarantee an x86-64 CPU.

From the login node:

```bash
srun --partition=gpu --nodelist=xgph1 --gres=gpu:a100-80:1 \
  --cpus-per-task=4 --mem=24G --time=01:00:00 --pty bash -l
```

Availability can change; select another verified x86-64 GPU node if needed. Inside the allocation:

```bash
hostname
uname -m
export EGTR_PYTHON="$HOME/aigc-storage/fyp-envs/qwen-a100-cu121/bin/python"
"$EGTR_PYTHON" - <<'CHECK'
import torch
print(torch.__version__)
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
print((torch.tensor([1., 2., 3.], device='cuda') * 2).tolist())
CHECK
```

Do not rerun `bootstrap_native.sh` as the next step. Its earlier native-environment recommendation is superseded for this session. Do not assume environments created on ARM are usable on x86-64. Only PyTorch GPU execution is confirmed: EGTR dependency compatibility, checkpoint loading and graph inference remain unverified.

## User-selected manual 20-plan run (2026-09-21)

The user now requests evaluation/visual inspection on the same 20 previously
excluded plan identities. They remain excluded from training. Selection is an
explicit evaluation request, not permission to tune on an untouched test set.
If these plans inform subsequent model/threshold selection, report them as
exploratory evaluation rather than a pristine final holdout.

`prepare_manual.py` copies the original manually annotated raster bytes, checks
image hashes and reviewed status, and makes a portable manifest. It does not
substitute another image variant or change annotation labels/completeness.

```bash
python3 experiments/egtr_calibration/prepare_manual.py --output cubicasa_eval/manual20_v1
```

This needs the saved manual JSONs/images on the machine where it runs. The Mac
already has a prepared `cubicasa_eval/manual20_v1.tar.gz` bundle; transfer that
to SOC instead of reconstructing manual references from source SVGs.

After unpacking on SOC, initialize a fresh run:

```bash
export EGTR_RUN=$(python3 experiments/egtr_calibration/runs.py init \
  --data cubicasa_eval/manual20_v1 --name manual20_frozen_vg \
  --note "20 manual references; frozen model evaluation")
```

Before attempting inference, use the **existing** Qwen Python on x86-64 xgph1:

```bash
"$HOME/aigc-storage/fyp-envs/qwen-a100-cu121/bin/python" \
  experiments/egtr_calibration/check_existing_runtime.py
```

This installs nothing. It probes dependency versions and EGTR Python imports
without compiling CUDA or loading weights. A failure means dependency
compatibility still needs addressing; do not reinstall Torch. Full model
execution remains a separate check and has not yet succeeded in this session.

Generate a portable, self-contained visualization before or after any stage:

```bash
python3 experiments/egtr_calibration/report.py --root "$EGTR_RUN"
```

Open `report.html` in a browser. It embeds all rasters and records, works offline,
and provides a plan picker, original image, reference/raw/adapted overlays,
boxes/edges/labels toggles, errors, matching, scores, native triplets and
provenance. Missing predictions are explicitly identified. Regenerating this
derived report does not overwrite raw evidence or metrics. Transfer the HTML
back to the Mac after a SOC run to inspect results without a remote web server.
