"""Convert an adjudicated reviewer export into a verified Qwen few-shot example.

The output is deliberately separate from the review export. It is a small,
versioned prompt asset, not a replacement for the canonical CubiCasa labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from retrieval_app.vlm_graph.schema import validate_graph


TYPE_MAP = {
    "bedroom": "bedroom", "bath": "bathroom", "bathroom": "bathroom",
    "kitchen": "kitchen", "livingroom": "living_room", "living_room": "living_room",
    "dining": "dining_room", "diningroom": "dining_room", "dining_room": "dining_room",
    "corridor": "corridor", "hall": "corridor", "storage": "storage",
    "balcony": "balcony", "entry": "entrance", "entrance": "entrance",
    "garage": "garage", "outdoor": "outdoor", "other": "other",
}


def qwen_type(value: str) -> str:
    key = value.replace(" ", "").replace("-", "").replace("_", "").lower()
    return TYPE_MAP.get(key, "other")


def normalise_box(box: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = (float(value) for value in box)
    return [round(x0 * 1000 / width, 2), round(y0 * 1000 / height, 2), round(x1 * 1000 / width, 2), round(y1 * 1000 / height, 2)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one verified Qwen spatial few-shot support record from a reviewer export.")
    parser.add_argument("--review", type=Path, required=True, help="Adjudicated human-review JSON export.")
    parser.add_argument("--image", type=Path, required=True, help="Matching floorplan raster available on the SOC cluster.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL support manifest; append one record per reviewed example.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    review = json.loads(args.review.expanduser().read_text(encoding="utf-8"))
    if not review.get("room_types_confirmed") or not review.get("graph_edges_confirmed"):
        raise ValueError("Only use an export after room types and graph edges are confirmed/adjudicated.")
    image = args.image.expanduser().resolve()
    if not image.exists():
        raise FileNotFoundError(image)
    width, height = Image.open(image).size
    rooms: list[dict[str, Any]] = []
    for room in review.get("rooms", []):
        box = normalise_box(room["bbox_xyxy"], width, height)
        rooms.append({
            "id": room["room_id"], "type": qwen_type(str(room.get("category_name", "other"))),
            "bbox": box, "centroid": [round((box[0] + box[2]) / 2, 2), round((box[1] + box[3]) / 2, 2)],
        })
    graph = validate_graph({
        "canvas": {"width": 1000, "height": 1000}, "rooms": rooms,
        "edges": [{"source": edge["room_a"], "target": edge["room_b"], "type": edge["predicate"], "confidence": 1.0} for edge in review.get("edges", [])],
    }, require_spatial=True)
    record = {"plan_id": review.get("plan_id", image.stem), "image_path": str(image), "graph": graph, "provenance": "human_reviewed", "reviewer": review.get("reviewer", "")}
    output = args.output.expanduser()
    if output.exists() and not args.overwrite:
        existing = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("plan_id") == record["plan_id"] for row in existing):
            raise ValueError(f"Support manifest already contains {record['plan_id']}; use --overwrite only for a new manifest.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps({"plan_id": record["plan_id"], "image_path": record["image_path"], "rooms": len(rooms), "edges": len(graph["edges"])}, indent=2))


if __name__ == "__main__":
    main()
