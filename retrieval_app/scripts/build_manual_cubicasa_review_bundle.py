"""Create an auditable proposed-manual-review bundle from a Qwen review HTML bundle.

The source CubiGraph SVG/JSON is never changed.  This script materialises a small,
explicit set of visual-review proposals into a new HTML+JSONL bundle so that a human
can approve or reject them before they become gold labels.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from retrieval_app.scripts.build_vlm_review import panel, spatial_overlay
from retrieval_app.scripts.render_qwen_review_graphs import svg_graph
from retrieval_app.vlm_graph.schema import cubigraph_adjacency, validate_graph


CODE_TO_RELATION = {1: "adjacent_to", 2: "connected_by_door", 3: "open_connected"}

# These are deliberately small, image-reviewed proposals for the supplied
# three-plan smoke bundle.  They are not an automatic transformation for 5k.
TYPE_UPDATES: dict[str, dict[str, str]] = {
    "cubicasa-ee3807b12f2f": {
        "Outdoor_1": "balcony", "Other_1": "living_room", "Other_2": "bedroom",
        "Other_3": "bathroom", "Other_4": "storage",
    },
    "cubicasa-ced7e73e1203": {
        "Other_1": "living_room", "Other_2": "bedroom", "Other_3": "corridor",
        "Other_4": "bathroom", "Other_5": "storage", "Other_6": "entrance",
    },
    "cubicasa-5e5d8f423156": {
        "Outdoor_1": "balcony", "Outdoor_2": "outdoor", "Outdoor_3": "balcony",
        "Dining_1": "dining_room", "Other_1": "bedroom", "Kitchen_1": "kitchen",
        "Entry_1": "entrance", "Other_2": "bedroom", "Other_3": "bedroom",
        "Other_4": "bedroom", "Bath_1": "bathroom", "Bath_2": "bathroom",
        "Other_5": "other", "Storage_1": "other", "Storage_2": "storage",
        "Entry_2": "corridor", "Other_6": "living_room", "Outdoor_4": "balcony",
        "Other_7": "bedroom", "Other_8": "bedroom", "Other_9": "bedroom",
        "Bath_3": "bathroom",
    },
}

# The first plan visibly has bathroom door access which Qwen downgraded.  The
# third has an open kitchen/dining zone rather than a mere shared wall.
EDGE_UPDATES: dict[str, dict[tuple[str, str], str]] = {
    "cubicasa-ee3807b12f2f": {
        ("Other_1", "Other_3"): "connected_by_door",
        ("Other_3", "Other_4"): "connected_by_door",
    },
    "cubicasa-5e5d8f423156": {
        ("Dining_1", "Kitchen_1"): "open_connected",
    },
}

NOTES = {
    "cubicasa-ee3807b12f2f": "Restore the two direct-access edges that the earlier Qwen prompt downgraded to adjacency.",
    "cubicasa-ced7e73e1203": "Keep CubiGraph topology; semantic labels remain proposed visual classifications.",
    "cubicasa-5e5d8f423156": "Apply visible room semantics; Kitchen_1 and Dining_1 form an open zone. Sauna/technical areas remain Other because the current 12-class schema has no dedicated classes.",
}


def pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def extract_bundle(source: Path) -> list[dict[str, Any]]:
    soup = BeautifulSoup((source / "index.html").read_text(encoding="utf-8"), "html.parser")
    records: list[dict[str, Any]] = []
    for article in soup.select("article"):
        heading = article.find("h1")
        if heading is None:
            continue
        plan_id = heading.get_text(" ", strip=True).split(". ", 1)[-1]
        panels = {panel.find("h2").get_text(" ", strip=True): panel for panel in article.select("section.panel") if panel.find("h2")}
        graph_panel = panels.get("Qwen fixed-node correction result")
        if graph_panel is None or graph_panel.find("pre") is None:
            raise ValueError(f"No fixed-node graph found for {plan_id}")
        qwen_graph = json.loads(graph_panel.find("pre").get_text())
        rooms = qwen_graph["rooms"]
        adjacency: dict[str, dict[str, int]] | None = None
        for details in article.select("details"):
            summary = details.find("summary")
            pre = details.find("pre")
            if summary and pre and summary.get_text(" ", strip=True) == "CubiGraph adjacency JSON — silver reference":
                adjacency = json.loads(pre.get_text())
                break
        if adjacency is None:
            raise ValueError(f"No CubiGraph adjacency found for {plan_id}")
        records.append({"plan_id": plan_id, "rooms": rooms, "qwen_graph": qwen_graph, "adjacency": adjacency})
    return records


def graph_from_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_id = record["plan_id"]
    rooms = [dict(room) for room in record["rooms"]]
    updates = TYPE_UPDATES.get(plan_id, {})
    changed_nodes: list[dict[str, Any]] = []
    for room in rooms:
        prior = room["type"]
        room["type"] = updates.get(room["id"], prior)
        if room["type"] != prior:
            changed_nodes.append({"room_id": room["id"], "from": prior, "to": room["type"]})
    edge_map: dict[tuple[str, str], str] = {}
    for source, neighbours in record["adjacency"].items():
        for target, code in neighbours.items():
            if code in CODE_TO_RELATION:
                edge_map[pair(source, target)] = CODE_TO_RELATION[code]
    qwen_edges = {
        pair(edge["source"], edge["target"]): edge["type"]
        for edge in record["qwen_graph"].get("edges", [])
    }
    changes: list[dict[str, Any]] = []
    for edge_pair, relation in EDGE_UPDATES.get(plan_id, {}).items():
        prior = qwen_edges.get(edge_pair)
        edge_map[edge_pair] = relation
        if prior != relation:
            changes.append({"room_a": edge_pair[0], "room_b": edge_pair[1], "from": prior, "to": relation})
    edges = [{"source": source, "target": target, "type": relation, "confidence": 1.0} for (source, target), relation in sorted(edge_map.items())]
    return validate_graph({"canvas": {"width": 1000, "height": 1000}, "rooms": rooms, "edges": edges}, require_spatial=True), changed_nodes + changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a proposed-manual CubiCasa graph review bundle.")
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_bundle.expanduser().resolve(), args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    records, cards, exports = extract_bundle(source), [], []
    for index, record in enumerate(records, start=1):
        plan_id = record["plan_id"]
        graph, changes = graph_from_record(record)
        prefix = f"{index:02d}_{plan_id}"
        image_source = source / "assets" / f"{prefix}_image.png"
        relation_source = source / "assets" / f"{prefix}_cubigraph_relations.svg"
        image_name, relation_name = image_source.name, relation_source.name
        shutil.copy2(image_source, assets / image_name)
        shutil.copy2(relation_source, assets / relation_name)
        overlay_name, graph_name = f"{prefix}_manual_overlay.svg", f"{prefix}_manual_graph.svg"
        (assets / overlay_name).write_text(spatial_overlay(graph, image_name), encoding="utf-8")
        (assets / graph_name).write_text(svg_graph(graph), encoding="utf-8")
        graph_json = json.dumps(graph, indent=2)
        changes_json = json.dumps(changes, indent=2)
        cards.append(f"""
<article><h1>{index}. {html.escape(plan_id)}</h1>
<p class=note><strong>Proposed manual review only.</strong> Source CubiGraph SVG/JSON is unchanged. {html.escape(NOTES.get(plan_id, ''))}</p>
<div class=grid>
{panel('Original floorplan', f'<img src="assets/{image_name}" alt="Original floorplan">')}
{panel('CubiGraph source relation SVG', f'<img src="assets/{relation_name}" alt="CubiGraph source relation SVG">')}
{panel('Proposed corrected graph overlay', f'<img src="assets/{overlay_name}" alt="Proposed corrected overlay">')}
{panel('Proposed node-edge graph', f'<img src="assets/{graph_name}" alt="Proposed node-edge graph">')}
{panel('Proposed node and edge changes', f'<pre>{html.escape(changes_json)}</pre>')}
{panel('Proposed canonical graph JSON', f'<pre>{html.escape(graph_json)}</pre>')}
</div><details><summary>Compact CubiGraph-compatible adjacency JSON</summary><pre>{html.escape(json.dumps(cubigraph_adjacency(graph), indent=2))}</pre></details></article>""")
        exports.append({"plan_id": plan_id, "source_provenance": "cubigraph_silver", "review_status": "proposed_manual_review", "manual_review_note": NOTES.get(plan_id, ""), "graph": graph, "cubigraph_adjacency": cubigraph_adjacency(graph), "changes": changes})
    document = f"""<!doctype html><html><head><meta charset=utf-8><title>Proposed CubiCasa graph review</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;background:#f7f7fb;color:#172033}}article{{background:white;margin:0 0 2rem;padding:1.25rem;border-radius:12px;box-shadow:0 1px 5px #0002}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}}.panel{{min-width:0;border:1px solid #ddd;border-radius:8px;padding:.75rem}}h2{{font-size:1rem;margin-top:0}}img{{display:block;max-width:100%;max-height:540px;margin:auto;object-fit:contain}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:540px;overflow:auto;font-size:.78rem}}.note{{padding:.7rem;border-left:4px solid #f59e0b;background:#fffbeb}}details{{margin-top:1rem}}@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Proposed CubiCasa node and edge corrections</h1><p>These are visually inspected proposals for human approval. They must remain separate from source SVG and CubiGraph silver labels until reviewed.</p>{''.join(cards)}</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "manual_review_proposals.jsonl").write_text("".join(json.dumps(row) + "\n" for row in exports), encoding="utf-8")
    print(f"Wrote {len(exports)} proposed manual-review records to {output}")


if __name__ == "__main__":
    main()
