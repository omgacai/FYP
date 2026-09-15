# CubiCasa5K graph validation and EGTR adaptation specification (v1)

## Purpose and boundary

Create a reproducible, provenance-preserving room scene-graph dataset from CubiCasa5K for an adapted EGTR baseline.  Compare three *validation* strategies without confusing a model prediction with ground truth:

1. human review through a web interface;
2. remote GPT/Claude image-to-graph disagreement triage; and
3. local Qwen image-to-graph disagreement triage and prompt refinement.

All three strategies consume the same frozen source-derived/silver corpus and write additive correction records.  The original SVG, CubiGraph edge, and raw VLM output are never overwritten.

This is an extension experiment for graph extraction.  It must not replace the primary buyer-query retrieval benchmark or its oracle facts.

## Fixed v1 label contract

### Room instances

One object represents one SVG room component.

| Field | Requirement |
| --- | --- |
| `room_id` | stable SVG/CubiGraph identifier such as `Kitchen_1` |
| `category_id` | positive integer from the frozen category map |
| `category_name` | original CubiCasa class, e.g. `Kitchen`, `Bedroom`, `Bath` |
| `bbox_xyxy` | `[x0, y0, x1, y1]` in pixels of `F1_scaled.png` |
| `source_polygon` | original SVG polygon/path reference or stored geometry |

Do not merge or drop rare classes before recording their count.  Decide any class merging only from training-split frequencies, document the mapping, and use the same map in every split.

The EGTR category map is not the same as the current small Qwen prompt schema.  If the prompt schema lacks a source class (for example `Garage` or `Outdoor`), record an explicit VLM-only mapping such as `Garage -> other`; do not change the canonical or EGTR category ID because of that limitation.

### Edges

There are exactly three mutually exclusive v1 predicates for each unordered room pair:

| Predicate ID | Name | Meaning |
| --- | --- | --- |
| 1 | `adjacent_to` | rooms share a meaningful boundary but no shared detected door |
| 2 | `connected_by_door` | an annotated/detected door joins the room pair |
| 3 | `open_connected` | distinct semantic room zones have a direct, unobstructed opening with no door |

The canonical corpus stores at most one undirected edge per lexicographically sorted room-ID pair. A reviewer must assign exactly one of the three predicates or leave the pair unlabelled (no relation). Review in this order: (1) `connected_by_door`, (2) `open_connected`, (3) `adjacent_to`. `adjacent_to` therefore explicitly means a shared boundary with neither form of direct access. Setting a new predicate for a pair replaces the prior one; duplicates with different predicate IDs are invalid.

The EGTR adapter emits two directed triplets for each edge, `A -> B` and `B -> A`, using the same predicate ID. Evaluation converts predictions back to undirected pairs before comparing with the canonical labels.

`near`, overlapping boxes, or merely touching a buffer are not evidence of an edge. Do not call an open-plan region `connected_by_door`; it is `open_connected` only when the reviewer can identify two distinct room semantics and a direct unblocked passage.

### Volunteer edge-review sequence

Do not ask novice reviewers to reconstruct the full graph from a dense overlay. First present every CubiGraph silver edge individually, with only its two rooms and one labelled link highlighted. For each suggestion, the reviewer selects exactly one outcome: accept the suggested predicate, change it to one of the other two predicates, or remove the relation. Store that decision as additive review evidence against the source edge.

Only after every silver suggestion has an outcome should the interface unlock **Add missing links**. In that mode, the reviewer selects two labelled room nodes and chooses one predicate. The interface replaces any prior relation on that unordered pair rather than creating a duplicate. A collapsed accepted-edge list and optional full-graph display are context aids, not the default review surface.

## Canonical corpus: source of truth for this project

The canonical record is JSONL, one plan per line.  It remains independent of a particular EGTR repository format.

```json
{
  "plan_id": "cubicasa-high_quality-3503",
  "split": "train",
  "image_path": "high_quality/3503/F1_scaled.png",
  "svg_path": "high_quality/3503/model.svg",
  "image_size": [1200, 800],
  "rooms": [
    {"room_id": "Kitchen_1", "category_id": 1, "category_name": "Kitchen", "bbox_xyxy": [34, 56, 290, 260]}
  ],
  "silver_edges": [
    {"room_a": "Kitchen_1", "room_b": "Other_1", "predicate_id": 2, "predicate": "connected_by_door"}
  ],
  "review": {"status": "pending", "corrections": []},
  "provenance": {
    "source": "CubiCasa5K model.svg",
    "graph_extractor": "cubigraph-door-first-v1",
    "svg_sha256": "<sha256>"
  }
}
```

Required validation before an example is included:

- source image and SVG exist and image dimensions match the stored dimensions;
- every box is finite, has positive area, and lies within the image;
- every edge endpoint is a declared room; no self loops or duplicate canonical edges;
- every edge has one allowed predicate;
- every plan belongs to exactly one deterministic, plan-ID-level split.

## CNN segmentor experiment: prediction versus target

For CubiCasa5K, run the pretrained CNN/FloorTrans segmentor on the raw `F1_scaled.png` only when evaluating the image-to-structure pipeline.  Its segmentation, reconstructed polygons/boxes, detected doors, and derived graph are **predictions**.  They must be compared with the source `model.svg`-derived rooms/boxes and CubiGraph silver edges; they must not replace those labels automatically.

```text
raw image -> CNN segmentation -> predicted rooms/boxes/doors -> predicted graph
model.svg -> source rooms/boxes/doors -> CubiGraph silver graph
                                  compare: room/box/edge metrics
```

Measure semantic segmentation IoU, room-instance class/box matching (IoU threshold fixed before evaluation), and edge precision/recall/F1 by predicate.  For the first EGTR training experiment, use the source-derived objects and silver/reviewed relations as targets.  A later, harder end-to-end experiment evaluates EGTR's image-only object and relation predictions against those same targets.

## EGTR adapter format

The public EGTR implementation uses a Visual Genome-like layout, rather than accepting generic graph JSON directly.  This repository pins it at `third_party/egtr` (commit `7f87450`). Its Visual Genome loader is patched only to derive the predicate dimension from `rel_categories` instead of assuming Visual Genome's 50 predicates. Name the exported dataset directory with `visual_genome` in its path (for example `cubicasa_visual_genome_v1`) so the upstream training entry point selects this compatible loader.

```text
cubicasa_egtr/
  images/                       # symlink to/read-only source images on SOC
  train.json                    # COCO object annotations
  val.json
  test.json
  rel.json                      # relations indexed by image ID
  label_map.json                # frozen room and predicate mappings
  manifest.jsonl                # canonical records; do not train directly from it
```

Each `train.json`, `val.json`, and `test.json` is COCO-style:

```json
{
  "images": [{"id": 0, "file_name": "high_quality/3503/F1_scaled.png", "width": 1200, "height": 800}],
  "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [34, 56, 256, 204], "area": 52224, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "Kitchen"}]
}
```

COCO uses `bbox = [x, y, width, height]`; the canonical corpus uses `bbox_xyxy`.  `annotation_id`/ordering must be deterministic.  Relation endpoints below are **zero-based positions in that image's ordered COCO annotation list**, not room IDs and not global annotation IDs.

```json
{
  "rel_categories": ["no_relation", "adjacent_to", "connected_by_door", "open_connected"],
  "train": {"0": [[0, 1, 2], [1, 0, 2]]},
  "val": {"100": []},
  "test": {"200": []}
}
```

Each triple is `[subject_object_index, object_object_index, predicate_id]`.  The official Visual Genome loader removes the `no_relation` slot internally, so predicate IDs in `rel.json` begin at 1.  It creates a relation tensor indexed by object-query positions; for this project, replace the repository's hard-coded 50 predicate dimension with `len(rel_categories) - 1` and assert that every plan has no more objects than `num_object_queries`.

### Mandatory EGTR smoke test

Before a full run, use 10 plans and verify all of the following:

1. the patched loader opens every image;
2. COCO categories become EGTR class labels without an off-by-one shift;
3. each relation triple indexes the intended two room annotations;
4. the relation tensor has exactly three predicate channels;
5. one forward/backward pass completes; and
6. a rendered overlay agrees with the exported boxes and edge endpoints.

Do not use the pretrained Visual Genome detector checkpoint as though it knows floor-plan room classes.  First adapt/pretrain the object detector on the CubiCasa room boxes, then train the relation component.

## Experimental split and audit plan

Freeze a deterministic 70/15/15 train/validation/test split by plan identity before prompts, reviews, or model selection.  The test split remains unseen during prompt refinement, validation-strategy selection, and checkpoint selection.

Create a separate 120-plan audit set sampled from the training/validation pool before running any VLM:

- 40 low-density/simple layouts;
- 40 medium-density layouts;
- 40 dense, geometrically irregular, invalid-geometry, or multi-storey cases.

If volunteers are used, two independent reviewers inspect every audit plan.  A third adjudication decision resolves disagreement.  Report edge precision, recall, and F1 by predicate and reviewer agreement on the candidate edge union; do not report accuracy over all possible room pairs because non-edges dominate.

## Validation arms

All arms receive the image and the same silver graph; they output a proposed graph plus a structured change set.  All prompts, models, temperatures, image resolution, and decoding limits are versioned.

### A — Volunteer web review

Review task: accept/reject/change-type each proposed edge; add missing edges; mark `uncertain`; provide a short reason.  Do not ask volunteers to redraw SVG rooms in v1.

Primary outcome: agreement and corrected-edge quality on the double-reviewed audit set.

### Volunteer queue, saving, and hand-off contract

The reviewer UI is a client of a central review service; browser downloads and local
uploads are only a prototype mode. Source images, SVG-derived rooms, and CubiGraph
silver edges remain read-only on SOC storage. The service gives the UI a review
packet and accepts additive review events/snapshots; it never mutates the canonical
record in place.

Use these states for each `(plan_id, review_round)` task:

`queued -> leased -> in_progress -> submitted -> adjudicated`.

- A reviewer enters a pseudonymous `reviewer_id`; the service assigns a small batch
  (for example 5–10 plans) and leases each task for 60 minutes.
- The UI autosaves a draft snapshot after edits (debounced) and sends a heartbeat.
  A browser close or expired lease leaves the task `in_progress`/`partial`, rather
  than losing work.
- The same reviewer may resume their partial task. A different reviewer may take a
  hand-off only after expiry; record `continued_from_session_id` and both reviewer
  IDs. Do not silently merge their edits.
- For the audit set, create two independent tasks (`review_round=1` and `2`) from
  the same frozen baseline. Reviewers should not see each other's corrections. An
  adjudicator produces the only `gold_reviewed` patch after disagreement resolution.
- For ordinary scaling, one submitted review may be used as `human_reviewed` silver
  supervision, but never as gold unless the predeclared acceptance rule is met.

Minimum persistent entities are: `plans` (immutable source/silver pointer and
version), `review_tasks` (queue/lease/status), `review_sessions` (reviewer and
timestamps), and append-only `review_snapshots`/`review_events` (rooms, edges,
notes, base revision). Use optimistic revision checks so two writers cannot overwrite
the same task. Store the final export as a review patch linked to the exact baseline
hash, not as a replacement SVG or graph.

### B — GPT/Claude disagreement triage

Run only after the audit protocol exists.  The model must return the strict room/edge schema, raw response, and confidence.  It may rank plans for review but cannot automatically change silver labels.

Primary outcome: on the frozen audit set, compare its edge precision/recall/F1 and the precision of its disagreement flags against adjudicated corrections.  Submit asynchronous batches or bounded parallel requests; never send one interactive request at a time.

### C — Local Qwen prompt refinement

Use the existing SOC Slurm Qwen runner.  Hold a fixed calibration subset separate from the audit and test sets.  Compare at least:

- baseline prompt;
- schema-constrained prompt with explicit door/adjacency definitions;
- at most one few-shot prompt, using only adjudicated calibration examples.

Primary outcome: valid-JSON rate, room count/type score, and edge precision/recall/F1 by predicate on the frozen audit set.  Select one prompt using calibration only, then run it once on the audit set.

### D — CNN segmentor graph extraction

Use the existing CubiCasa/FloorTrans parser to predict room regions and door/icon evidence from the raw image, then derive its graph with the same documented relation policy.  This is not a label-generation arm; it is an image-only baseline against the source/SVG-derived labels.

Primary outcome: segmentation IoU, matched-room box/class score, and edge precision/recall/F1 by predicate on the frozen audit set.

## Decision rules

| Question | Evidence required | Decision |
| --- | --- | --- |
| Is CubiGraph usable as silver training supervision? | audit edge F1 by predicate | use it only with its measured limitation stated |
| Does a VLM help? | disagreement-flag precision exceeds random/heuristic sampling | use it to rank review, never as automatic gold |
| Is volunteer review worth scaling? | acceptable agreement and correction yield | deploy review only to uncertainty-ranked plans |
| Which prompt is used? | best calibration score; frozen prompt/config | one Qwen configuration advances to the full corpus |
| Does EGTR improve extraction? | held-out full image-only relation metrics versus source-box relation control | report separately; do not conflate controls |

## Execution order on SOC

1. Count/validate images and SVGs on SOC storage; do not copy the corpus to a laptop.
2. Generate canonical records and CubiGraph silver labels in a Slurm job.
3. Render ten examples and inspect them; repair extraction only if the source evidence is wrong.
4. Freeze splits and the audit/calibration sets.
5. Run all three arms on the same audit examples.
6. Choose the validation mechanism using the decision rules, then generate review packets/queues for only selected plans.
7. Export EGTR files, pass the smoke test, pretrain room detection, then train relations.

## Required deliverables

- canonical manifest, split seed, category map, predicate map, extractor commit/config, and validation report;
- raw VLM outputs and reviewer records separate from corrected labels;
- 10-plan rendered smoke-test bundle;
- EGTR loader patch plus a loader test that asserts relation endpoint alignment;
- final test metrics with source-box and full image-only results clearly separated.
