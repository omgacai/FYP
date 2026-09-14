# CubiCasa5K → EGTR graph-supervision plan

## Goal

Fine-tune EGTR to predict a room graph from a floor-plan raster image.  A training example needs image-level room instances (class plus raster bounding box) and typed relations between room instances.

Do not label the entire corpus manually or let a VLM silently become ground truth.  CubiCasa's source SVG is the strongest annotation source available for room geometry and doors; CubiGraph5K is a reproducible, rule-derived graph from those SVGs.  VLM output is an independent prediction that can flag disagreement for review.

## Label provenance

| Field | Source | Label status |
| --- | --- | --- |
| image | `F1_scaled.png` | source |
| room polygon, room class, box | `model.svg`, transformed into the image coordinate system | source-derived |
| `adjacent_to`, `connected_by_door` | CubiGraph extractor with a pinned policy/version | silver |
| VLM graph and confidence | immutable model output | prediction |
| correction and rationale | human reviewer, with reviewer/date | gold-reviewed |

Keep all five layers.  A correction must be an additive record; never mutate or discard the original SVG/CubiGraph/VLM evidence.

## Relation contract

Use exactly two undirected predicates for version 1:

- `adjacent_to`: rooms share a meaningful boundary but there is no detected intervening door.
- `connected_by_door`: a door connects the two rooms.

Do not infer `connected_by_door` merely from touching polygons or a close bounding box.  EGTR normally represents relations as directed subject-predicate-object triplets, so export each undirected edge twice with the same predicate (`A → B` and `B → A`), and deduplicate at evaluation.

Initial room classes should retain the CubiCasa semantic types, including `Outdoor`, `Garage`, and `Other`; only merge classes after checking their frequency and visual consistency in the frozen audit set.  Do not invent `dining_room` unless the SVG annotation exposes it.

## Correct order of work

1. Download/extract the official CubiCasa5K data so every sample has `F1_scaled.png` and `model.svg`.  The present workspace has the code checkout but not the dataset files.
2. Create an immutable manifest with a stable source-path key, image/SVG checksums, image dimensions, and a deterministic train/validation/test split by plan ID.  Recommended split: 70/15/15, seeded and written once.
3. Derive rooms, boxes, and the door-first CubiGraph edges from the SVG.  Stamp extractor version, parser backend, and relation policy on every record.  Exclude or explicitly flag CubiGraph's published invalid-geometry and multi-storey lists rather than hiding failures.
4. Freeze a stratified 100-plan audit subset before any model training: 50 clean/simple plans, 30 dense plans, 20 difficult/ambiguous or multi-storey cases.  Sample from each visual subset and relation-density band.
5. Review that subset independently.  The reviewer sees the image, SVG/room overlay, silver graph, and VLM prediction; record only corrections with a reason.  Double-review at least 30 plans and resolve disagreements into a written relation policy.
6. Quantify silver-label quality on the audited sample: room-class agreement, box IoU, and edge precision/recall/F1 separately for `adjacent_to` and `connected_by_door`.  Only then run VLM disagreement triage over the rest.
7. Run the VLM only to rank uncertain examples: invalid output, room-count mismatch, an edge absent from silver, a missing silver edge, or low confidence.  Review a fixed high-disagreement sample; do not auto-apply changes.
8. Export the exact annotation format required by the pinned EGTR commit.  First train/evaluate using SVG-derived boxes to isolate relation learning; later train/evaluate full image-only scene-graph detection with predicted boxes.

## Minimum canonical record

```json
{
  "plan_id": "cubicasa-high_quality-3503",
  "split": "train",
  "image_path": "high_quality/3503/F1_scaled.png",
  "svg_path": "high_quality/3503/model.svg",
  "image_size": [1200, 800],
  "rooms": [{"id": "Kitchen_1", "class": "Kitchen", "bbox_xyxy": [34, 56, 290, 260]}],
  "silver_edges": [{"source": "Kitchen_1", "target": "Other_1", "predicate": "connected_by_door"}],
  "gold_review": {"status": "pending", "corrections": []},
  "provenance": {"svg_sha256": "…", "extractor": "cubigraph-door-first-v1"}
}
```

The EGTR adapter can derive directed triplets from `silver_edges`/reviewed edges while leaving this canonical record unchanged.

## Definition of done for the labeling milestone

- All included records have valid image, SVG, classes, boxes, edge endpoints, and split.
- Every edge has one of the two documented predicates and connects declared rooms.
- The audit set is frozen, double-review agreement is reported, and corrected labels have a rationale.
- The exported EGTR dataset can be loaded by a smoke-test dataloader without changing counts or edge endpoints.
- Results state whether they evaluate source-box relations or full image-only graph prediction.

## Immediate commands once data is present

```bash
python retrieval_app/scripts/build_cubicasa_manifest.py \
  --cubicasa-root /absolute/path/to/cubicasa5k/data \
  --output retrieval_app/data/cubicasa_manifest.jsonl

python retrieval_app/scripts/index_cubicasa_graphs.py \
  --manifest retrieval_app/data/cubicasa_manifest.jsonl \
  --corpus-root /absolute/path/to/cubicasa5k/data \
  --cubigraph-repo third_party/CubiGraph5K \
  --graph-dir retrieval_app/data/cubigraph_silver \
  --output retrieval_app/data/cubicasa_graph_manifest.jsonl \
  --limit 10
```

Run the second command with `--limit 10` first.  Inspect those ten records, then remove the limit only after the room IDs and door semantics look right.
