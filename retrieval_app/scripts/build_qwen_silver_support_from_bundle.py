"""Make a *prototype-only* few-shot support manifest from a Qwen review bundle.

This consumes the original plan images, CubiGraph relation SVGs, and adjacency
JSON embedded in the generated HTML.  Its output is explicitly silver support:
use it to test prompt mechanics, never to claim human-verified prompt quality.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image

from retrieval_app.scripts.build_qwen_spatial_support import qwen_type
from retrieval_app.vlm_graph.schema import validate_graph


RELATIONS = {1: "adjacent_to", 2: "connected_by_door", 3: "open_connected"}


def points_bbox(points: str) -> list[float]:
    values = [float(value) for value in re.findall(r"-?(?:\d+\.?\d*|\.\d+)", points)]
    if len(values) < 6 or len(values) % 2:
        raise ValueError(f"Invalid polygon points: {points!r}")
    xs, ys = values[::2], values[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalise(box: list[float], width: float, height: float) -> list[float]:
    return [round(box[0] * 1000 / width, 2), round(box[1] * 1000 / height, 2), round(box[2] * 1000 / width, 2), round(box[3] * 1000 / height, 2)]


def bundle_graph(html_text: str, plan_id: str) -> dict[str, dict[str, int]]:
    article = re.search(rf"<article>.*?<h1>[^<]*{re.escape(plan_id)}</h1>(.*?)</article>", html_text, flags=re.DOTALL)
    if article is None:
        raise ValueError(f"Could not find plan {plan_id!r} in index.html")
    block = re.search(r"CubiGraph adjacency JSON.*?<pre>(.*?)</pre>", article.group(1), flags=re.DOTALL)
    if block is None:
        raise ValueError(f"Could not find CubiGraph adjacency for {plan_id!r}")
    graph = json.loads(html.unescape(block.group(1)).strip())
    if not isinstance(graph, dict):
        raise ValueError(f"CubiGraph adjacency for {plan_id!r} is not an object")
    return graph


def rooms_from_relation_svg(path: Path) -> tuple[float, float, list[tuple[str, list[float]]]]:
    root = ET.parse(path).getroot()
    width, height = float(root.attrib["width"]), float(root.attrib["height"])
    ns = {"svg": "http://www.w3.org/2000/svg"}
    polygons = [points_bbox(element.attrib["points"]) for element in root.findall("svg:polygon", ns) if element.attrib.get("stroke", "").lower() not in {"black", "grey", "gray"}]
    labels = ["".join(element.itertext()).strip() for element in root.findall("svg:text", ns) if element.attrib.get("fill", "").lower() not in {"black", "grey", "gray"}]
    if not polygons or len(polygons) != len(labels):
        raise ValueError(f"Expected equal coloured room polygons and labels in {path}; found {len(polygons)} and {len(labels)}")
    return width, height, list(zip(labels, polygons, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build silver prototype Qwen supports from qwen_review_* HTML bundle assets.")
    parser.add_argument("--bundle", type=Path, required=True, help="Directory containing index.html and assets/.")
    parser.add_argument("--plan-id", action="append", required=True, help="Repeat once or twice for the support plans.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= len(args.plan_id) <= 2:
        parser.error("Use one or two --plan-id values to keep few-shot controlled.")

    bundle = args.bundle.expanduser().resolve()
    index = (bundle / "index.html").read_text(encoding="utf-8")
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for plan_id in args.plan_id:
        assets = bundle / "assets"
        images = list(assets.glob(f"*_{plan_id}_image.png"))
        svgs = list(assets.glob(f"*_{plan_id}_cubigraph_relations.svg"))
        if len(images) != 1 or len(svgs) != 1:
            raise ValueError(f"Expected one image and one relation SVG for {plan_id}; found {len(images)} and {len(svgs)}")
        image = images[0].resolve()
        image_width, image_height = Image.open(image).size
        svg_width, svg_height, source_rooms = rooms_from_relation_svg(svgs[0])
        rooms = []
        for room_id, box in source_rooms:
            scaled = normalise(box, svg_width, svg_height)
            semantic_hint = re.sub(r"[_-]\d+$", "", room_id)
            rooms.append({"id": room_id, "type": qwen_type(semantic_hint), "bbox": scaled, "centroid": [round((scaled[0] + scaled[2]) / 2, 2), round((scaled[1] + scaled[3]) / 2, 2)]})
        adjacency = bundle_graph(index, plan_id)
        edges, seen = [], set()
        for source, neighbours in adjacency.items():
            for target, code in neighbours.items():
                key = tuple(sorted((source, target)))
                if key in seen:
                    continue
                seen.add(key)
                if code in RELATIONS:
                    edges.append({"source": source, "target": target, "type": RELATIONS[code], "confidence": 1.0})
        graph = validate_graph({"canvas": {"width": 1000, "height": 1000}, "rooms": rooms, "edges": edges}, require_spatial=True)
        records.append({"plan_id": plan_id, "image_path": str(image), "graph": graph, "provenance": "cubigraph_silver_prototype", "source_bundle": str(bundle), "source_image_size": [image_width, image_height]})
    output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(json.dumps({"output": str(output), "plans": [record["plan_id"] for record in records], "provenance": "cubigraph_silver_prototype"}, indent=2))


if __name__ == "__main__":
    main()
