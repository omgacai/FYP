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
