# CubiCasa → EGTR data-preparation log

This log records the issues encountered while deriving the CubiCasa EGTR
fine-tuning corpus. The original CubiCasa5K files were never edited or
deleted. Every exclusion is represented in a versioned derived artifact.

## Final training input

- Canonical source: `~/vlm/data/derived/egtr_finetune_v1/canonical_v2.jsonl`
- EGTR adapter: `~/vlm/data/derived/egtr_finetune_v1/cubicasa_visual_genome_v2`
- Rejection record: `~/vlm/data/derived/egtr_finetune_v1/canonical_v2_rejections.jsonl`
- Training labels: source `model.svg` room boxes/classes plus CubiGraph
  silver edges.
- Object vocabulary: `LivingRoom`, `Bedroom`, `Kitchen`, `Dining`, `Bath`,
  `Storage`, `Entry`, `Garage`, `Other`, `Outdoor`.
- Relation vocabulary: `adjacent_to`, `direct_access`. `direct_access`
  currently represents source-CubiGraph door connectivity; unsupported open
  passages are not independently supervised.

## Data quality and cleaning outcomes

| Issue | What happened | Resolution | Audit trail |
| --- | --- | --- | --- |
| Manual-gold leakage risk | Twenty manually annotated plans are reserved for the external gold evaluation. | Their CubiCasa folder identities are filtered before the deterministic train/validation split. | `experiments/egtr_calibration/exclusions.json`; corpus metadata records exclusions. |
| Invalid source geometry | Twelve plans raised source parsing errors, primarily degenerate room boxes or invalid polygon topology. | They were excluded from canonical v1 rather than repaired silently. | `~/vlm/data/derived/egtr_finetune_v1/errors_v1.jsonl` |
| Invalid SVG polygon topology | Some source SVG room polygons triggered Shapely `TopologyException: side location conflict` during graph extraction. | The graph extractor repairs invalid geometries with `buffer(0)` before intersection; failures remain logged rather than producing a guessed graph. | CubiGraph output under `~/vlm/data/derived/egtr_finetune_v1/cubigraph_v1/` and canonical/error records. |
| Empty detector target | `high_quality/8426/F1_scaled.png` produced zero `model.svg` room instances for the CubiGraph parser. EGTR then had no matched boxes and its relation loss became `NaN` at training step 70. | A canonical cleaner removes records with no rooms, invalid/out-of-canvas boxes, duplicate room IDs, or edges without valid endpoints. The corpus builder now rejects empty-room records in future builds. | `canonical_v2_rejections.jsonl`; `retrieval_app/scripts/clean_egtr_canonical.py` |
| Very large raster plans | PIL emitted `DecompressionBombWarning` for a small number of unusually large floor-plan images. | This is a size warning, not a detected corrupt file. Those plans were retained when valid room annotations and boxes were present. | Corpus-build log; no automatic exclusion. |
| Source SVG parser availability | The cluster environment lacked `lxml`, initially preventing BeautifulSoup from parsing SVGs. | The data builders fall back to Python's `html.parser`; the parser used is recorded per canonical record. | `provenance.svg_parser` in canonical JSONL. |

## Validation gates

The final adapter is checked before GPU training with:

```bash
"$EGTR_PYTHON" experiments/egtr_calibration/audit_egtr_adapter.py \
  --data "$DERIVED/cubicasa_visual_genome_v2" \
  --canonical "$DERIVED/canonical_v2.jsonl" \
  --exclusions experiments/egtr_calibration/exclusions.json
```

The audit requires all of the following:

1. No manual-gold identity occurs in the canonical train/validation corpus.
2. Every image exists through the read-only CubiCasa image link.
3. Every room box is finite, positive-area, and inside its image canvas.
4. Every plan has at least one room annotation.
5. Every relation endpoint refers to an existing room, and all relation IDs
   are in the two-label training vocabulary.

The 100-train-batch / 25-validation-batch stability run completed with this
cleaned v2 adapter. It is a training sanity check, not an evaluation result.

## Interpretation limits

- The room boxes and classes are source-derived CubiCasa supervision. The
  graph edges are silver labels created by CubiGraph rules, so they are not a
  replacement for the manual external gold benchmark.
- The current source graph does not provide a separate `open_connected`
  target. Manual gold may still distinguish that relation during final
  evaluation, but this first EGTR training stage cannot learn it separately.
