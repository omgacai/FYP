"""Create one portable HTML report for the Qwen image-only manual-20 run.

The report deliberately includes invalid Qwen responses.  JSON validity is an
experimental result, so omitting those plans would make the graph scores look
better than the actual end-to-end method.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Any


TYPE_COLOURS = {
    "Bedroom": "#0284c7", "Bath": "#0f766e", "Kitchen": "#ca8a04",
    "LivingRoom": "#dc2626", "Dining": "#f97316", "Corridor": "#64748b",
    "Storage": "#65a30d", "Entry": "#db2777", "Garage": "#7c3aed",
    "Outdoor": "#16a34a", "Other": "#a855f7",
}
QWEN_TYPE_MAP = {
    "bedroom": "Bedroom", "bathroom": "Bath", "kitchen": "Kitchen",
    "living_room": "LivingRoom", "dining_room": "Dining", "corridor": "Corridor",
    "storage": "Storage", "entrance": "Entry", "garage": "Garage",
    "balcony": "Outdoor", "outdoor": "Outdoor", "other": "Other",
}
EDGE_COLOURS = {
    "adjacent_to": "#2563eb", "connected_by_door": "#9333ea",
    "open_connected": "#ea580c", "direct_access": "#9333ea",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_data(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def scalar(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def metric_table(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "<p class='muted'>Not evaluated because Qwen's response was invalid.</p>"
    node, edge = metrics.get("node_metrics", {}), metrics.get("edge_metrics", {})
    per_relation = metrics.get("per_relation", {})
    rows = [
        ("Nodes", node), ("Edges", edge),
        *[(f"Relation: {name}", value) for name, value in per_relation.items()],
    ]
    return "<table><tr><th>Measure</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr>" + "".join(
        f"<tr><td>{html.escape(name)}</td><td>{scalar(value.get('tp'))}</td><td>{scalar(value.get('fp'))}</td>"
        f"<td>{scalar(value.get('fn'))}</td><td>{scalar(value.get('precision'))}</td>"
        f"<td>{scalar(value.get('recall'))}</td><td>{scalar(value.get('f1'))}</td></tr>"
        for name, value in rows
    ) + "</table>" + (
        f"<p class='muted'>Matched type accuracy: {scalar(metrics.get('type_accuracy'))}; "
        f"mean matched IoU: {scalar(metrics.get('mean_matched_iou'))}; "
        f"aligned graph edit cost: {scalar(metrics.get('aligned_edit_cost'))}; "
        f"normalized edit cost: {scalar(metrics.get('normalized_edit_cost'))}.</p>"
    )


def gold_graph(annotation: dict[str, Any]) -> dict[str, Any]:
    return {"nodes": annotation.get("nodes", []), "edges": annotation.get("edges", [])}


def qwen_graph(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record or not record.get("valid") or not isinstance(record.get("graph"), dict):
        return None
    graph = record["graph"]
    nodes = []
    for room in graph.get("rooms", []):
        bbox = room.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        nodes.append({"id": str(room.get("id")), "type": QWEN_TYPE_MAP.get(str(room.get("type", "other")).lower(), "Other"),
                      "bbox_xyxy": [float(value) / 1000 for value in bbox]})
    return {"nodes": nodes, "edges": [{"a": edge.get("source"), "b": edge.get("target"), "relation": edge.get("type")}
                                            for edge in graph.get("edges", [])]}


def overlay(graph: dict[str, Any], image: str, title: str) -> str:
    nodes = graph.get("nodes", [])
    points: dict[str, tuple[float, float]] = {}
    parts = [f"<svg viewBox='0 0 1000 1000' role='img' aria-label='{html.escape(title)}'>",
             f"<image href='{image}' x='0' y='0' width='1000' height='1000' preserveAspectRatio='none'/>"]
    for node in nodes:
        box = node.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (float(value) * 1000 for value in box)
        points[str(node.get("id"))] = ((x0 + x1) / 2, (y0 + y1) / 2)
    for edge in graph.get("edges", []):
        a, b = points.get(str(edge.get("a"))), points.get(str(edge.get("b")))
        if not a or not b:
            continue
        relation = str(edge.get("relation"))
        dash = " stroke-dasharray='11 8'" if relation == "adjacent_to" else ""
        parts.append(f"<line x1='{a[0]:.1f}' y1='{a[1]:.1f}' x2='{b[0]:.1f}' y2='{b[1]:.1f}' "
                     f"stroke='{EDGE_COLOURS.get(relation, '#334155')}' stroke-width='5' stroke-opacity='.80'{dash}/>")
    for node in nodes:
        box = node.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (float(value) * 1000 for value in box)
        colour = TYPE_COLOURS.get(str(node.get("type")), "#334155")
        label = html.escape(str(node.get("type", "Other")))
        parts.extend([f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{x1-x0:.1f}' height='{y1-y0:.1f}' fill='none' stroke='{colour}' stroke-width='5'/>",
                      f"<text x='{(x0+x1)/2:.1f}' y='{max(22, y0-7):.1f}' text-anchor='middle' font-size='20' font-family='system-ui' font-weight='700' fill='{colour}' stroke='white' stroke-width='4' paint-order='stroke'>{label}</text>"])
    return "".join(parts) + "</svg>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-root", type=Path, required=True, help="Packaged manual20 folder with manifest.json, images/, annotations/.")
    parser.add_argument("--predictions", type=Path, required=True, help="Merged raw Qwen JSONL.")
    parser.add_argument("--graph-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .html file; all image data is embedded.")
    args = parser.parse_args()
    root = args.manual_root.expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    records = {str(row["plan_id"]): row for row in read_jsonl(args.predictions.expanduser().resolve())}
    metrics = read_json(args.graph_metrics.expanduser().resolve())
    plan_metrics = {str(row["plan_id"]): row for row in metrics.get("per_plan", [])}
    expected = [str(row["plan_id"]) for row in manifest]
    missing = set(expected) - set(records)
    if missing:
        raise ValueError(f"Predictions do not cover every manual plan: {sorted(missing)}")

    cards: list[str] = []
    for position, row in enumerate(manifest, start=1):
        plan_id = str(row["plan_id"])
        image_path = root / str(row["image_path"])
        annotation = read_json(root / "annotations" / f"{plan_id}.graph.json")
        record = records[plan_id]
        image = image_data(image_path)
        qgraph = qwen_graph(record)
        state = "valid" if qgraph else "invalid"
        qpanel = overlay(qgraph, image, f"Qwen prediction for {plan_id}") if qgraph else (
            "<div class='invalid'><strong>Invalid Qwen response — excluded from graph scoring.</strong>"
            f"<p>{html.escape(str(record.get('error', 'No validated graph')))}</p></div>"
        )
        repaired = record.get("json_repair_outputs", [])
        cards.append(f"""<article id='{html.escape(plan_id)}'>
<header><h2>{position}. {html.escape(plan_id)}</h2><span class='badge {state}'>{state.upper()}</span></header>
<p class='muted'>Gold: manually reviewed raster graph. Qwen: image-only model prediction. Boxes are Qwen's rectangular approximations.</p>
<div class='grid'><section><h3>Manual gold graph</h3>{overlay(gold_graph(annotation), image, f'Manual gold for {plan_id}')}</section>
<section><h3>Qwen graph</h3>{qpanel}</section></div>
<h3>Per-plan evaluation</h3>{metric_table(plan_metrics.get(plan_id))}
<details><summary>Manual gold JSON</summary><pre>{html.escape(json.dumps(annotation, indent=2))}</pre></details>
<details><summary>Validated Qwen graph / validation error</summary><pre>{html.escape(json.dumps(record.get('graph') if qgraph else {'error': record.get('error')}, indent=2))}</pre></details>
<details><summary>Raw Qwen response</summary><pre>{html.escape(str(record.get('raw_output', '')))}</pre></details>
<details><summary>JSON repair outputs ({len(repaired)})</summary><pre>{html.escape(json.dumps(repaired, indent=2))}</pre></details>
</article>""")
    global_metrics = json.dumps({key: value for key, value in metrics.items() if key != "per_plan"}, indent=2)
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Qwen manual-20 image-to-graph report</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#f1f5f9;color:#0f172a}} main{{max-width:1600px;margin:auto;padding:28px}} article, .summary{{background:#fff;border-radius:14px;padding:20px;margin:20px 0;box-shadow:0 1px 4px #0f172a1f}} header{{display:flex;align-items:center;gap:12px}} h1,h2,h3{{margin:0 0 10px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} section{{min-width:0}} svg{{display:block;width:100%;background:#e2e8f0;border-radius:8px}} .muted{{color:#475569}} .badge{{font-weight:800;border-radius:999px;padding:4px 9px;font-size:12px}} .valid{{background:#dcfce7;color:#166534}} .invalid{{background:#fee2e2;color:#991b1b}} .invalid{{padding:20px;border-radius:8px;min-height:150px}} details{{margin-top:12px}} summary{{cursor:pointer;font-weight:650}} pre{{max-height:430px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px;border-radius:8px}} table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:6px 8px;text-align:right;border-bottom:1px solid #e2e8f0}} th:first-child,td:first-child{{text-align:left}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}main{{padding:12px}}}}
</style></head><body><main><h1>Qwen3-VL image-only graph evaluation: manual-20</h1>
<p>This is a self-contained report: all 20 raster images are embedded, so download this single HTML file and open it locally. Invalid Qwen outputs remain visible and count against end-to-end coverage.</p>
<div class='summary'><h2>Aggregate metrics</h2><pre>{html.escape(global_metrics)}</pre></div>{''.join(cards)}</main></body></html>"""
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Report: {output}")
    print(f"Plans: {len(manifest)}; valid Qwen JSON: {sum(bool(records[plan_id].get('valid')) for plan_id in expected)}/{len(expected)}")


if __name__ == "__main__":
    main()
