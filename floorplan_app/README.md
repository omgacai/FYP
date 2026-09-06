# Floorplan Pipeline Inspector

Run from the FYP directory:

```bash
cd ~/Downloads/FYP
python3 -m pip install -r requirements_macos.txt
python3 -m streamlit run floorplan_app/app.py
```

The default image is `input/floor.jpg`; change `DEFAULT_IMAGE` in `floorplan_app/app.py` if needed.

The app also supports **Upload SVG** mode. It bypasses image parsing and sends an uploaded CubiGraph-compatible SVG directly through the deterministic CubiGraph relation extraction step. Use it to inspect SVGs created by a VLM or another vectoriser.

## Architecture

`parsers/` contains model-specific adapters. Each must implement `FloorplanParser.parse()` and return `ParserResult`; importing the adapter registers it in `core/registry.py`. `pipeline/svg_extractor.py` and `pipeline/graph_extractor.py` consume that common result and therefore remain unchanged when you add a new parser (e.g. DeepLabV3).

The app distinguishes three levels of evidence:

1. Raw pixel confidence from the parser.
2. Post-processed polygons and SVG conversion.
3. Deterministic CubiGraph relations.

Do not treat a graph created from a predicted pseudo-SVG as ground truth. It is a diagnostic/model output.
