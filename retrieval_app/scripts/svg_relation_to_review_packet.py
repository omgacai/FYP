"""Convert a CubiGraph relation-diagram SVG into a reviewer JSONL record."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image


def tag(element): return element.tag.rsplit("}", 1)[-1]

def convert(svg_path: Path, image_path: Path, plan_id: str) -> dict:
    """Return one provisional reviewer record from a CubiGraph diagram SVG."""
    root = ET.parse(svg_path).getroot()
    rooms = [e for e in root if tag(e) == "polygon" and e.attrib.get("stroke") != "black"]
    circles = [e for e in root if tag(e) == "circle"]
    labels = ["".join(e.itertext()).strip() for e in root if tag(e) == "text"]
    if len(rooms) != len(circles) or len(circles) != len(labels):
        raise ValueError(f"Expected equal room polygons/circles/labels, got {len(rooms)}/{len(circles)}/{len(labels)}")
    image_size = list(Image.open(image_path).size)
    parsed_rooms, centres = [], []
    for polygon, circle, room_id in zip(rooms, circles, labels):
        points = [tuple(map(float, point.split(","))) for point in polygon.attrib["points"].split()]
        xs, ys = zip(*points)
        room_type = room_id.rsplit("_", 1)[0]
        parsed_rooms.append({"room_id": room_id, "category_id": 0, "category_name": room_type, "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)]})
        centres.append((float(circle.attrib["cx"]), float(circle.attrib["cy"]), room_id))
    def nearest(x, y): return min(centres, key=lambda item: (item[0]-x)**2 + (item[1]-y)**2)[2]
    edges = []
    for line in root:
        if tag(line) == "line":
            first, second = nearest(float(line.attrib["x1"]), float(line.attrib["y1"])), nearest(float(line.attrib["x2"]), float(line.attrib["y2"]))
            edges.append({"room_a": first, "room_b": second, "predicate": "connected_by_door" if "stroke-dasharray" in line.attrib else "adjacent_to"})
    return {"plan_id": plan_id, "image_size": image_size, "rooms": parsed_rooms, "silver_edges": edges, "provenance": {"graph_provenance": "silver", "source_relation_svg": str(svg_path)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--svg", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--plan-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    record = convert(a.svg, a.image, a.plan_id)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(f"Wrote {a.output}: {len(record['rooms'])} rooms, {len(record['silver_edges'])} edges")
if __name__ == "__main__": main()
