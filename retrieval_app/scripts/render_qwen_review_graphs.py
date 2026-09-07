"""Add rendered Qwen node-edge diagrams to an existing manual review bundle."""
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any


TYPE_COLOURS = {
    "bedroom": "#b7d7f0", "bathroom": "#b9ead8", "kitchen": "#f7d9a8",
    "living_room": "#e6c5f2", "dining_room": "#f5d0ca", "corridor": "#d9dde3",
    "storage": "#e6e7a7", "balcony": "#bbdfaa", "entrance": "#f4b8c5", "other": "#dddddd",
}
EDGE_COLOURS = {"adjacent_to": "#2563eb", "connected_by_door": "#c026d3"}


def svg_graph(graph: dict[str, Any]) -> str:
    rooms = graph.get("rooms", [])
    edges = graph.get("edges", [])
    if not rooms:
        return "<svg viewBox='0 0 800 600' xmlns='http://www.w3.org/2000/svg'><text x='400' y='300' text-anchor='middle'>Qwen predicted no rooms.</text></svg>"
    width, height = 800, 600
    centre_x, centre_y = width / 2, height / 2
    radius = min(220, max(125, 28 * len(rooms)))
    positions: dict[str, tuple[float, float]] = {}
    for index, room in enumerate(rooms):
        angle = -math.pi / 2 + 2 * math.pi * index / len(rooms)
        positions[room["id"]] = (centre_x + radius * math.cos(angle), centre_y + radius * math.sin(angle))

    edge_parts: list[str] = []
    for edge in edges:
        source, target = positions.get(edge.get("source")), positions.get(edge.get("target"))
        if not source or not target:
            continue
        colour = EDGE_COLOURS.get(edge.get("type"), "#6b7280")
        edge_parts.append(
            f"<line x1='{source[0]:.1f}' y1='{source[1]:.1f}' x2='{target[0]:.1f}' y2='{target[1]:.1f}' "
            f"stroke='{colour}' stroke-width='3' stroke-opacity='.78'/>"
        )

    node_parts: list[str] = []
    for room in rooms:
        x, y = positions[room["id"]]
        colour = TYPE_COLOURS.get(room["type"], TYPE_COLOURS["other"])
        label = html.escape(str(room["id"]))
        type_label = html.escape(str(room["type"]))
        node_parts.append(
            f"<g><circle cx='{x:.1f}' cy='{y:.1f}' r='43' fill='{colour}' stroke='#273043' stroke-width='2'/>"
            f"<text x='{x:.1f}' y='{y - 4:.1f}' text-anchor='middle' font-size='11' font-weight='600'>{label}</text>"
            f"<text x='{x:.1f}' y='{y + 14:.1f}' text-anchor='middle' font-size='10'>({type_label})</text></g>"
        )
    legend = "".join(
        f"<line x1='{30 + i * 230}' y1='570' x2='{58 + i * 230}' y2='570' stroke='{colour}' stroke-width='4'/>"
        f"<text x='{66 + i * 230}' y='574' font-size='13'>{kind}</text>"
        for i, (kind, colour) in enumerate(EDGE_COLOURS.items())
    )
    return f"""<svg viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' role='img' aria-label='Qwen predicted room graph'>
<rect width='{width}' height='{height}' fill='white'/>{''.join(edge_parts)}{''.join(node_parts)}{legend}</svg>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render node-edge SVGs from Qwen JSON embedded in review index.html.")
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    review_dir = args.review_dir.expanduser().resolve()
    index = review_dir / "index.html"
    assets = review_dir / "assets"
    source = index.read_text(encoding="utf-8")
    article_pattern = re.compile(r"<article>(.*?)</article>", re.DOTALL)
    graph_panel = re.compile(
        r'(<section class="panel"><h2>Qwen direct image-to-graph prediction</h2><pre>)(.*?)(</pre></section>)',
        re.DOTALL,
    )
    changed = 0

    def transform(article_match: re.Match[str]) -> str:
        nonlocal changed
        article = article_match.group(1)
        plan_match = re.search(r"<h1>\d+\. ([^<]+)</h1>", article)
        prediction_match = graph_panel.search(article)
        if not plan_match or not prediction_match:
            return article_match.group(0)
        prediction_text = html.unescape(prediction_match.group(2)).strip()
        try:
            graph = json.loads(prediction_text)
        except json.JSONDecodeError:
            graph_html = "<p>No graph diagram: Qwen output was invalid JSON.</p>"
        else:
            plan_id = plan_match.group(1)
            filename = f"{plan_id}_qwen_graph.svg"
            (assets / filename).write_text(svg_graph(graph), encoding="utf-8")
            graph_html = f'<img src="assets/{html.escape(filename)}" alt="Qwen node-edge graph" /><p class="legend"><span class="adjacent">Blue</span>: adjacent_to &nbsp; <span class="door">Magenta</span>: connected_by_door</p>'
        new_panel = prediction_match.group(0) + f'<section class="panel"><h2>Rendered Qwen node-edge graph</h2>{graph_html}</section>'
        changed += 1
        return "<article>" + article[:prediction_match.start()] + new_panel + article[prediction_match.end():] + "</article>"

    source = article_pattern.sub(transform, source)
    source = source.replace(
        "pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 520px; overflow: auto; font-size: .78rem; }",
        "pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 520px; overflow: auto; font-size: .78rem; } .legend { font-size: .8rem; } .adjacent { color: #2563eb; font-weight: 700; } .door { color: #c026d3; font-weight: 700; }",
    )
    index.write_text(source, encoding="utf-8")
    print(f"Added rendered Qwen graph panels for {changed} review records: {index}")


if __name__ == "__main__":
    main()
