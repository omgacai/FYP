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
