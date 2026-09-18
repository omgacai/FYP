# Raster floor plan → graph → question answering

Updated 2026-09-18. This is the active pilot direction, superseding the earlier 100-plan supervised EGTR audit as the immediate next step. Earlier validation documents remain historical context, not prerequisites for this pilot.

## Research question

How much graph accuracy is actually needed for reliable floor-plan QA? Compare extraction quality, downstream answer quality, cost and latency. A graph with fewer errors is not automatically the graph that helps the VLM most.

## 1. Annotate the benchmark first

- [x] Add a raster-only manual annotation mode to `review_react`.
- [x] Support node placement/movement, semantic labels, typed edges, undo and local JSON export/import.
- [x] Add concave room polygons, editable corners, automatically derived axis-aligned boxes, and version-1 JSON migration.
- [x] Include plan identity, image SHA-256, annotator, review status and ambiguity notes.
- [ ] Try the UI on 2–3 calibration images outside the final 25. Resolve room and relation conventions before annotating the benchmark.
- [ ] Select 25 CubiCasa plans with varied room counts, visual quality, open layouts and access complexity; record source split and source path. Keep multiple versions of the same plan together.
- [ ] Manually label all 25 without viewing method predictions. Recheck ambiguous cases in a second pass; ideally obtain an independent check of a subset.
- [ ] Freeze a versioned manifest and all reviewed annotations before comparing methods.

Use the downloaded dataset locally for now. The supplied [Drive folder](https://drive.google.com/drive/folders/1jYt_W33Pw8_5OxP-HOARkzY5fvyC8KSl) is a source reference; its contents have not been verified or imported. No Drive integration is required for the UI.

Local storage (created and ignored by Git):

```
cubicasa_eval/
  manifest.json                # plan_id, source path/split, image hash
  images/<unique-plan-id>.png
  annotations/<unique-plan-id>.graph.json
  questions.jsonl
  predictions/<method>/<unique-plan-id>.json
  results/
```

Browser downloads go to your configured download directory. Save or move JSON files into `cubicasa_eval/annotations/` and original images into `cubicasa_eval/images/`. This directory is ignored by Git; back it up separately. Browser autosave is only a single-current-plan recovery aid; downloaded JSON is the durable copy. Uploading a folder to Drive can be a later backup step.

### Annotation contract, version 2

One node represents one room or clearly distinct functional zone. Put its point inside that zone, preferably near its centre. Use stable IDs and a semantic type; a display label such as Bedroom 2 is for readability. Corridor is supported explicitly. For open kitchen/living layouts, annotate distinct zones only when the raster provides enough evidence; explain uncertain boundaries in notes. Do not invent a room behind an unclear mark.

Coordinates are normalized to [0,1], origin at the raster top-left. Keep image dimensions and hash. Node position refers to the actual room location, not an arbitrary graph layout. Trace each room’s visible interior boundary as a polygon, including concave corners. The UI derives its axis-aligned enclosing box as [xmin, ymin, xmax, ymax]. Keep the detailed polygon and derived box together: the rectangle may enclose non-room space for irregular rooms. Never infer adjacency or access from box overlap. Nodes remain room anchors; changing a polygon preserves node IDs and edges. All rooms need outlines before marking a new record reviewed. Point-only version-1 drafts can be imported and completed. Box metrics require verified room matching/class mappings and a compatible evaluator; annotation alone does not complete EGTR integration.

One symmetric relation per unordered pair, using this precedence:

1. `connected_by_door`: visible, traversable door between the rooms.
2. `open_connected`: visible direct opening between distinct zones, without a door.
3. `adjacent_to`: shared boundary, with neither of the above direct access relations.
4. `uncertain`: direct relation cannot be determined confidently; exclude from strict relation scoring and report coverage.

Missing edges mean no direct relation only after the annotator explicitly confirms checking all pairs. Drafts are incomplete, not negative labels. Never infer access from proximity. Marking a record `manually_reviewed` records the annotator's completion; it does not imply independent adjudication. Retain provenance and notes.

## 2. Make the evaluation reproducible

- [ ] Write a versioned canonical adapter and validator for all four methods. Preserve raw outputs, model IDs/checkpoint hashes, prompts, image preprocessing, runtime and failures.
- [ ] Define node matching before measuring edge errors. IDs differ across methods and repeated bedrooms must not be matched by label alone.
- [ ] For this small pilot, inspect and freeze one-to-one spatial node correspondences blind to predicted edges and QA outcomes. Save unmatched and ambiguous nodes explicitly. Later automate using normalized spatial distance and Hungarian matching, tuning distance thresholds on separate calibration plans. Avoid matching by class alone, which can hide type errors.
- [ ] Measure node precision/recall, missing/extra rooms, matched-node type accuracy, and typed-edge precision/recall/F1 per relation and per plan. Include unmatched-node edges as errors; deduplicate symmetric pairs.
- [ ] Report uncertain/excluded pairs and evaluable coverage. Define empty-set cases explicitly, and report micro totals plus plan-level averages.
- [ ] Implement a reproducible edit-cost measure on the fixed correspondence. Count missing nodes, extra nodes, wrong node types, missing edges, extra edges and relation substitutions separately. Use unit costs initially and count one wrong edge type as one substitution. Exclude incident edges already charged through node deletion/insertion to avoid double charging; record this convention.
- [ ] Call this **aligned graph edit cost**, not exact minimum graph edit distance. Exact GED optimizes correspondence as well and may be expensive. Report raw cost and cost divided by max(1, gold nodes + evaluable gold edges); this ratio can exceed 1.

Use the same frozen 25 for paired comparisons across methods. Tune prompts, thresholds and mappings on separate calibration images, not repeatedly on the 25. If only 25 images are available in total, preregister 5 calibration + 20 held-out instead and label the results accordingly. Twenty-five plans are a pilot, not a definitive large-scale benchmark.

## 3. Compare four extraction methods

| Method | Input and fixed configuration | Main check |
|---|---|---|
| Segmentation → CubiGraph rules | Raster → selected segmentation model → canonical geometry/SVG → deterministic relation extraction | Name the segmentation checkpoint; keep CubiGraph rule version fixed. Source `model.svg` must not replace predicted geometry in the scored raster-only method. |
| Frozen EGTR | Raster → released checkpoint → documented ontology adapter | First check whether output labels can express rooms and the required relations. Treat unsupported classes explicitly. |
| Codex graph labelling | Raster + frozen annotation prompt/schema | Record actual model/version, date, prompt, tool context and raw answer. No human correction in scored predictions. |
| Qwen current prompt | Raster + existing prompt in `retrieval_app/vlm_graph/prompts.py` | Snapshot the current prompt, checkpoint and inference settings; retain parse errors and retries. |

The [official EGTR repository](https://github.com/naver-ai/egtr) releases weights for Visual Genome and Open Images V6. Those are general scene-graph domains. Frozen weights are a valid transfer probe, but a room/door ontology is not guaranteed. Do not relabel unrelated predicates as door connectivity. If no defensible mapping exists, report unsupported coverage/failure and keep any floor-plan fine-tuned experiment separate from the frozen arm.

- [ ] Run each method on calibration plans; establish feasible adapters.
- [ ] Freeze all four configurations and run on identical raster files.
- [ ] Save failures as outcomes, not silently dropped samples.
- [ ] Evaluate extraction accuracy, latency, cost and unsupported output coverage.

## 4. Measure what matters: QA delta

- [ ] Manually write 5–8 questions per benchmark plan, before seeing predictions: counts/types, direct relations, reachability, and compound constraints. Only ask geometry questions supported by the annotation. Include negative questions and explicitly ambiguous/unanswerable examples.
- [ ] Save question ID, plan ID, category, accepted answer, supporting node/edge IDs and answerability. Verify answers from the raster and reviewed annotation.
- [ ] Keep the answering VLM, prompt, decoding, graph serialization and token budget fixed across arms.
- [ ] Run image-only, image + manual graph, image + each predicted graph; optionally add graph-only to isolate dependence on structure.
- [ ] Score normalized exact answers for counts/yes-no, documented rubrics for free text, and appropriate abstention on ambiguous questions. Inspect contradictions rather than relying only on a model judge.
- [ ] Report `QA gap(method) = accuracy(image + manual graph) − accuracy(image + predicted graph)` and `QA gain(method) = accuracy(image + predicted graph) − accuracy(image only)`, in percentage points.
- [ ] Plot per-plan/per-category graph error against QA gap. Bootstrap by plan, since questions from one plan are correlated.
- [ ] Perturb the manual graph at controlled levels: delete/add edges, swap relation types, relabel/drop nodes. Use multiple fixed seeds; keep the same questions and image. Separate errors touching question evidence from irrelevant errors.
- [ ] Predeclare a practical tolerance (for example, a 5-percentage-point QA gap as a pilot target), then examine the error types and levels within that tolerance. Do not conclude equivalence merely from a small nonsignificant difference.

If the manual graph offers no benefit over image-only QA, investigate that before investing heavily in extraction quality. Controlled corruptions test sensitivity; real model errors test actual usefulness. Both are needed to substantiate the idea that an imperfect graph is sufficient.

## Current next action

Start the React app, annotate the separate calibration images, save JSON locally, and settle the annotation conventions. Then select and label the frozen 25. Evaluation and model runners are subsequent work; this change implements the annotation starting point, not the complete experiment.

## Changelog

- 2026-09-18: Shifted immediate scope from supervised graph training/retrieval to a 25-plan raster graph/QA pilot; introduced normalized point-node manual annotation and explicit uncertain relations; retained door/open/adjacency separation.

- 2026-09-18: Added normalized room polygons and derived `bbox_xyxy` in schema `floorplan-manual-graph/2` at the user’s request. Geometry is preserved separately from graph relations. Full EGTR dataset conversion remains future work.
