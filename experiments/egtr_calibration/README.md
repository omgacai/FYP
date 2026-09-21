# Three-plan frozen EGTR calibration

Prepared 2026-09-21. This experiment uses **source CubiCasa model.svg room boxes plus CubiGraph rule-derived silver edges**, not manually reviewed graphs or CNN predictions. It does not change the 20 saved benchmark annotations.

Selected deterministically before model inference:

| Source plan | Rooms | Silver edges |
|---|---:|---:|
| high_quality/14343 | 12 | 12 |
| high_quality/1900 | 10 | 8 |
| high_quality/1097 | 8 | 7 |

All 20 supplied exclusion folders were excluded by category/plan identity, independent of local/cluster path prefixes. All image variants of excluded plans are therefore excluded. The runnable data snapshot lives in `cubicasa_eval/calibration/egtr_v1`; manifest and selection JSON preserve hashes and selection provenance. Reserve these three for calibration and exclude them from subsequent final test selection.

## Status

- Source-derived references and nine draft count questions generated locally.
- Adapter unit tests and graph evaluator checks available.
- Actual EGTR and Qwen inference **not run**: local CUDA unavailable; `soc-nus` SSH failed at the jump host with `Permission denied (publickey)`.
- Checkpoint, its matching config, ordered vocabularies, and upstream-compatible EGTR environment still required.
- Draft QA answers require raster review. Add direct-relation and compound questions after checking source-rule errors. Do not claim these nine count questions measure topology sensitivity.

The existing rule extractor does not detect `open_connected`; its adjacency rule is a buffered-polygon heuristic. Reference graphs deliberately retain `all_pairs_reviewed=false`. Scores measure agreement with this pipeline, not verified ground-truth topology. The QA condition is named `reference_graph`, not `manual_graph`.

## Run on SOC

Keep EGTR's legacy environment separate from the modern Qwen environment. Upstream setup: https://github.com/naver-ai/egtr . Use the **full trained EGTR** checkpoint link, not just its pretrained object detector. Its documented torch/transformers versions are in `third_party/egtr/requirements.txt`; install/build on an allocated compute node, not xlogin. No requirement that you use eight GPUs for this inference probe.

1. Copy this code directory, the existing `third_party/egtr`, the shared evaluator module, and the calibration snapshot into the corresponding FYP paths on SOC. The snapshot already contains the exact three rasters and source references, so there is no need to regenerate them remotely.
2. Obtain the official checkpoint artifact and retain `config.json`. Supply `labels.json` with `objects` and `relations` arrays in exact output-index order. For VG, objects follow COCO category ID minus one; relations follow `rel_categories[1:]`, excluding no_relation. Never guess label order. The runner checks dimensions, but semantic ordering must match the actual artifact.
3. From the FYP root set `EGTR_PYTHON`, `EGTR_ARTIFACT`, `EGTR_CHECKPOINT`, `EGTR_LABELS` to actual paths, then submit:

```bash
sbatch experiments/egtr_calibration/run_egtr.sbatch
```

Scheduler resource choices may need adapting to your allocation. The runner requires CUDA and loads the full checkpoint strictly. Saves full raw tensors plus object labels/scores, image/config/checkpoint hashes and timing. Existing raw records are protected against overwrite.

4. Inspect raw outputs and source vocabulary. Fill `ontology_mapping.json` only for defensible mappings:

```json
{"objects": {"ACTUAL_SOURCE_CLASS": {"target": "Kitchen", "justification": "Explain why this class denotes a kitchen room"}}, "relations": {"ACTUAL_SOURCE_RELATION": {"target": "connected_by_door", "justification": "Explain evidence of direct door access"}}}
```

The shipped mappings are intentionally empty. Mapping objects in a kitchen to a kitchen room, or `near` to door access, is not valid. If the ontology cannot represent the task, record that outcome and stop the frozen graph QA arm. Do not silently substitute an empty successful prediction.

5. Adapt and evaluate from the FYP root:

```bash
"$EGTR_PYTHON" experiments/egtr_calibration/adapt.py \
  --root cubicasa_eval/calibration/egtr_v1 \
  --mapping experiments/egtr_calibration/ontology_mapping.json
node experiments/egtr_calibration/evaluate.mjs cubicasa_eval/calibration/egtr_v1
```

Thresholds (object .3, combined relation .01, room-matching IoU .3) are provisional calibration settings, not validated defaults. Inspect class-blind spatial correspondences. Symmetric relation predictions reduce to the maximum-scoring supported type/direction per pair. Full tensors are preserved. Use a new snapshot version when changing thresholds; predictions are not overwritten.

## QA

Review each question against the raster, fix answer/evidence errors, and set `review_status` to `human_reviewed` in `questions.jsonl`. Accepted answers never enter model prompts. Add direct door/open-passage, negative, and compound questions only after checking the required evidence. Pin one Qwen3-VL checkpoint revision for every condition.

Run on a GPU allocation in the existing modern Qwen environment:

```bash
python experiments/egtr_calibration/qa.py \
  --root cubicasa_eval/calibration/egtr_v1 \
  --model YOUR_QWEN3_VL_MODEL --revision YOUR_PINNED_REVISION
```

This produces paired `image_only`, `reference_graph`, and `egtr_graph` records with raw answers, prompts, deterministic decoding, exact-answer scores, runtime and explicit failures/unavailable arms. Unsupported EGTR outputs remain unavailable rather than silently becoming image-only QA. Reference and predicted graphs use the same stripped serialization. No human-reference claim is made.

Use matched question IDs for QA comparisons, report arm coverage/failures, and calculate `reference_graph accuracy - egtr_graph accuracy` and `egtr_graph accuracy - image_only accuracy` in percentage points. Three plans are a feasibility check, not a reliable estimate of generalization or a graph-error tolerance threshold.

## Local verification

```bash
.venv/bin/python -m unittest experiments.egtr_calibration.test_pipeline
node --test review_react/src/graphEvaluation.test.js
node experiments/egtr_calibration/evaluate.mjs
```

Synthetic adapter tests are software checks only; they are not EGTR results.

## Cluster without Apptainer

Apptainer is not required. Use a **separate native Python/Conda environment** for EGTR; do not install its legacy dependencies into the existing Qwen environment. Before selecting install commands, inspect available Python versions, CUDA toolkit (`nvcc`), compiler and cluster modules. The custom deformable-attention extension requires a compatible CUDA build environment. GPU drivers alone do not supply `nvcc`. The original dependency set targets PyTorch 1.12.1/CUDA 11.3 and Transformers 4.18.0; compatibility on this cluster has not yet been validated.

After pulling this commit, fetch the pinned upstream repositories:

```bash
bash experiments/egtr_calibration/fetch_sources.sh
```

This does not install packages or overwrite existing checkouts. EGTR inference uses upstream source directly; the local Visual Genome training-loader modification is not needed for this experiment. Upstream repositories and datasets are not included in this commit.

The portable `exclusions.json` stores the 20 excluded category/plan identities. To regenerate the same three calibration inputs on the cluster after source-pipeline dependencies are installed, set `CUBICASA_DATASET` to the directory containing `high_quality/` and `colorful/`, then run from the FYP root:

```bash
python experiments/egtr_calibration/prepare.py \
  --dataset "$CUBICASA_DATASET" \
  --exclusions experiments/egtr_calibration/exclusions.json \
  --output cubicasa_eval/calibration/egtr_v1
```

Selection is deterministic for the same candidate inventory. Compare `selection.json` against the three identities listed above; a different dataset inventory may select different plans. This command requires Pillow, numpy, beautifulsoup4, lxml and shapely. It refuses to overwrite an existing output directory. Data remain gitignored; `git pull` alone does not transfer the locally generated references or image snapshot.
