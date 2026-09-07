"""Build a local HTML bundle for manual Qwen-versus-CubiGraph review.

The page is read-only on purpose: the human reviewer records gold decisions in
the emitted JSONL template rather than silently overwriting silver labels.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any


TYPE_COLOURS = {
    "bedroom": "#00a8f3", "bathroom": "#44e0bb", "kitchen": "#ffc400",
    "living_room": "#ed174c", "dining_room": "#ff7b67", "corridor": "#666666",
    "storage": "#668b1b", "balcony": "#55cc00", "entrance": "#ff5ca8", "other": "#c95b99",
}
EDGE_COLOURS = {"adjacent_to": "#2563eb", "connected_by_door": "#c026d3"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(value: str, root: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def copy_asset(source: Path | None, assets: Path, filename: str) -> str | None:
    if source is None or not source.exists():
        return None
    destination = assets / filename
    shutil.copy2(source, destination)
    return f"assets/{destination.name}"


def panel(title: str, body: str) -> str:
    return f'<section class="panel"><h2>{html.escape(title)}</h2>{body}</section>'


def spatial_overlay(graph: dict[str, Any], source_image_name: str) -> str:
    """Render Qwen's normalised boxes/centroids directly over its input image."""
    rooms = graph.get("rooms", [])
    if not rooms or not all("bbox" in room and "centroid" in room for room in rooms):
        return ""
    positions = {room["id"]: room["centroid"] for room in rooms}
    parts = [
        "<svg viewBox='0 0 1000 1000' xmlns='http://www.w3.org/2000/svg' role='img' aria-label='Qwen spatial graph overlay'>",
        f"<image href='{html.escape(source_image_name)}' x='0' y='0' width='1000' height='1000' preserveAspectRatio='none'/>",
    ]
    for edge in graph.get("edges", []):
        source, target = positions.get(edge.get("source")), positions.get(edge.get("target"))
        if source and target:
            colour = EDGE_COLOURS.get(edge.get("type"), "#555")
            dash = "stroke-dasharray='9 7'" if edge.get("type") == "adjacent_to" else ""
            parts.append(f"<line x1='{source[0]}' y1='{source[1]}' x2='{target[0]}' y2='{target[1]}' stroke='{colour}' stroke-width='5' stroke-opacity='.72' {dash}/>")
    for room in rooms:
        x0, y0, x1, y1 = room["bbox"]
        cx, cy = room["centroid"]
        colour = TYPE_COLOURS.get(room["type"], "#666")
        label = html.escape(room["id"])
        parts.extend([
            f"<rect x='{x0}' y='{y0}' width='{x1-x0}' height='{y1-y0}' fill='none' stroke='{colour}' stroke-width='5'/>",
            f"<circle cx='{cx}' cy='{cy}' r='13' fill='{colour}' stroke='white' stroke-width='3'/>",
            f"<text x='{cx}' y='{cy-20}' text-anchor='middle' font-size='19' font-family='sans-serif' font-weight='700' fill='{colour}' stroke='white' stroke-width='4' paint-order='stroke'>{label}</text>",
        ])
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a side-by-side manual review bundle for VLM graph predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5, help="Review a small gold subset first.")
    parser.add_argument("--plan-id", action="append", default=[], help="Optional repeatable plan ID filter.")
    args = parser.parse_args()

    root = args.corpus_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    manifest = {row["plan_id"]: row for row in read_jsonl(args.manifest.expanduser().resolve())}
    predictions = read_jsonl(args.predictions.expanduser().resolve())
    wanted = set(args.plan_id)
    if wanted:
        predictions = [row for row in predictions if row["plan_id"] in wanted]
    if args.limit:
        predictions = predictions[: args.limit]

    notes: list[dict[str, Any]] = []
    cards: list[str] = []
    for position, prediction in enumerate(predictions, start=1):
        plan_id = str(prediction["plan_id"])
        source = manifest.get(plan_id)
        if source is None:
            raise ValueError(f"Prediction {plan_id} is not present in the manifest.")
        image = copy_asset(resolve(str(source["image_path"]), root), assets, f"{position:02d}_{plan_id}_image.png")
        graph_path = resolve(str(source["graph_path"]), root) if source.get("graph_path") else None
        relation_svg = graph_path.with_name(f"{plan_id}_relations.svg") if graph_path else None
        relation = copy_asset(relation_svg, assets, f"{position:02d}_{plan_id}_cubigraph_relations.svg")
        reference = graph_path.read_text(encoding="utf-8") if graph_path and graph_path.exists() else "No CubiGraph JSON found."
        graph = prediction.get("graph")
        qwen_graph = json.dumps(graph, indent=2) if graph else "INVALID OUTPUT"
        raw = str(prediction.get("raw_output", ""))
        image_html = f'<img src="{image}" alt="Original floorplan" />' if image else "<p>Original image missing.</p>"
        relation_html = f'<img src="{relation}" alt="CubiGraph relation SVG" />' if relation else "<p>Relation SVG missing.</p>"
        overlay_html = ""
        if graph and image:
            overlay = spatial_overlay(graph, Path(image).name)
            if overlay:
                overlay_name = f"{position:02d}_{plan_id}_qwen_spatial_overlay.svg"
                (assets / overlay_name).write_text(overlay, encoding="utf-8")
                overlay_html = panel("Qwen spatial graph overlay — predicted boxes", f'<img src="assets/{overlay_name}" alt="Qwen box and graph overlay" /><p>Boxes and edges are Qwen predictions, not ground truth.</p>')
        checklist = """
        <ul>
          <li>Are room counts/types sensible?</li>
          <li>Which Qwen edges are unsupported by the drawing?</li>
          <li>Which true adjacency/door edges did Qwen miss?</li>
          <li>Is the CubiGraph edge itself plausible, or a buffering/rule error?</li>
        </ul>"""
        cards.append(f"""
        <article>
          <h1>{position}. {html.escape(plan_id)}</h1>
          <p><strong>Qwen JSON valid:</strong> {html.escape(str(prediction.get('valid')))} &nbsp;
          <strong>Reference provenance:</strong> silver (CubiGraph from source SVG)</p>
          <div class="grid">
            {panel("Original floorplan image", image_html)}
            {panel("CubiGraph relation SVG — silver reference", relation_html)}
            {panel("Qwen direct image-to-graph prediction", f"<pre>{html.escape(qwen_graph)}</pre>")}
            {overlay_html}
          </div>
          <details><summary>CubiGraph adjacency JSON — silver reference</summary><pre>{html.escape(reference)}</pre></details>
          <details><summary>Raw Qwen answer</summary><pre>{html.escape(raw)}</pre></details>
          {panel("Manual review checklist", checklist)}
        </article>""")
        notes.append({
            "plan_id": plan_id,
            "qwen_prediction_path": str(args.predictions.expanduser().resolve()),
            "cubigraph_provenance": "silver",
            "review_status": "pending",
            "reviewer_notes": "",
            "gold_graph": None,
        })

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Qwen vs CubiGraph manual review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f7f7fb; color: #171721; }}
article {{ background: white; margin: 0 0 2rem; padding: 1.25rem; border-radius: 12px; box-shadow: 0 1px 5px #0002; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }}
.panel {{ min-width: 0; border: 1px solid #ddd; border-radius: 8px; padding: .75rem; }}
h1 {{ margin-top: 0; }} h2 {{ font-size: 1rem; margin-top: 0; }}
img {{ display: block; max-width: 100%; max-height: 520px; margin: auto; object-fit: contain; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; max-height: 520px; overflow: auto; font-size: .78rem; }}
details {{ margin-top: 1rem; }}
@media (max-width: 1000px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>Qwen direct graph vs CubiGraph manual review</h1>
<p><strong>Important:</strong> CubiGraph is a rule-derived <em>silver</em> reference. Record your own gold judgement in <code>review_notes.jsonl</code>; do not assume either output is automatically correct.</p>
{''.join(cards)}
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "review_notes.jsonl").write_text("".join(json.dumps(note) + "\n" for note in notes), encoding="utf-8")
    print(f"Wrote {len(notes)} review records to {output}")
    print(f"Open: {output / 'index.html'}")
    print(f"Record gold decisions in: {output / 'review_notes.jsonl'}")


if __name__ == "__main__":
    main()
