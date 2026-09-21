"""Export source-SVG room instances and silver edges into canonical JSONL.

This consumes CubiCasa's annotated ``model.svg`` files, not predictions from the
FloorTrans parser.  It is therefore the target builder for the EGTR experiment;
the CNN/VLM pipelines are evaluated against its output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image

from retrieval_app.data.manifest import load_adjacency


CATEGORIES = [
    "LivingRoom", "Bedroom", "Kitchen", "Dining", "Bath", "Storage",
    "Entry", "Garage", "Other", "Outdoor",
]
RELATIONS = {1: "adjacent_to", 2: "connected_by_door", 3: "open_connected"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def parse_length(value: object) -> float:
    text = str(value).strip()
    number = "".join(char for char in text if char.isdigit() or char in ".-+eE")
    if not number:
        raise ValueError(f"Cannot parse SVG dimension {value!r}.")
    return float(number)


def canvas_transform(svg, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return SVG origin and size, preferring the viewBox when present."""
    view_box = svg.attrs.get("viewBox") or svg.attrs.get("viewbox")
    if view_box:
        values = [float(value) for value in str(view_box).replace(",", " ").split()]
        if len(values) == 4 and values[2] > 0 and values[3] > 0:
            return values[0], values[1], values[2], values[3]
    width, height = parse_length(svg.attrs["width"]), parse_length(svg.attrs["height"])
    if width <= 0 or height <= 0:
        raise ValueError("SVG dimensions must be positive.")
    return 0.0, 0.0, width, height


def deterministic_split(plan_id: str, seed: str) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{plan_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def source_rooms(svg_path: Path, image_size: tuple[int, int], cubigraph_repo: Path, *, svg_coordinate_size: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    src = cubigraph_repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from plan import Plan

    # Match CubiGraph's parser: its class selectors expect HTML-style class lists.
    soup = BeautifulSoup(svg_path.read_text(encoding="utf-8"), "lxml")
    svg = soup.find("svg")
    if svg is None:
        raise ValueError(f"No SVG root in {svg_path}")
    # Official CubiCasa loading rasterizes SVG coordinates directly on F1_scaled.
    # SVG viewport dimensions may differ and must not define coordinate scaling.
    if svg_coordinate_size is None:
        with Image.open(svg_path.with_name("F1_scaled.png")) as scaled_image:
            svg_coordinate_size = scaled_image.size
    origin_x, origin_y = 0.0, 0.0
    svg_width, svg_height = svg_coordinate_size
    if svg_width <= 0 or svg_height <= 0:
        raise ValueError("SVG coordinate canvas must be positive")
    scale_x, scale_y = image_size[0] / svg_width, image_size[1] / svg_height
    plan = Plan(svg)
    rooms: list[dict[str, Any]] = []
    for room in plan.rooms:
        if room.type not in CATEGORIES:
            raise ValueError(f"Unsupported CubiGraph room type {room.type!r} in {svg_path}")
        xs = [(point[0] - origin_x) * scale_x for point in room.points]
        ys = [(point[1] - origin_y) * scale_y for point in room.points]
        x0, x1 = max(0.0, min(xs)), min(float(image_size[0]), max(xs))
        y0, y1 = max(0.0, min(ys)), min(float(image_size[1]), max(ys))
        if not x0 < x1 or not y0 < y1:
            raise ValueError(f"Degenerate room box for {room.name} in {svg_path}")
        rooms.append({
            "room_id": room.name,
            "category_id": CATEGORIES.index(room.type) + 1,
            "category_name": room.type,
            "bbox_xyxy": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
        })
    return rooms


def canonical_edges(adjacency: dict[str, dict[str, int]], room_ids: set[str]) -> list[dict[str, Any]]:
    edges: dict[tuple[str, str], int] = {}
    for source, neighbours in adjacency.items():
        for target, code in neighbours.items():
            if source not in room_ids or target not in room_ids or source == target or code not in RELATIONS:
                continue
            key = tuple(sorted((source, target)))
            prior = edges.get(key)
            if prior is not None and prior != code:
                raise ValueError(f"Conflicting relation codes for {key}: {prior} and {code}")
            edges[key] = code
    return [
        {"room_a": source, "room_b": target, "predicate_id": code, "predicate": RELATIONS[code]}
        for (source, target), code in sorted(edges.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical CubiCasa room/edge records for EGTR.")
    parser.add_argument("--manifest", type=Path, required=True, help="Enriched manifest containing graph_path.")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--cubigraph-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path, help="Optional JSONL error report.")
    parser.add_argument("--seed", default="cubicasa-egtr-v1")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = args.corpus_root.expanduser().resolve()
    rows = read_jsonl(args.manifest.expanduser().resolve())
    if args.limit:
        rows = rows[: args.limit]
    output_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        plan_id = str(row["plan_id"])
        try:
            if not row.get("graph_path"):
                raise ValueError("Manifest record has no graph_path; run index_cubicasa_graphs first.")
            image_path = resolve(str(row["image_path"]), root)
            svg_path = resolve(str(row["svg_path"]), root)
            graph_path = resolve(str(row["graph_path"]), root)
            if not image_path.exists() or not svg_path.exists() or not graph_path.exists():
                raise FileNotFoundError("image_path, svg_path, or graph_path is missing")
            with Image.open(image_path) as image:
                image_size = image.size
            rooms = source_rooms(svg_path, image_size, args.cubigraph_repo.expanduser().resolve())
            room_ids = {room["room_id"] for room in rooms}
            edges = canonical_edges(load_adjacency(graph_path), room_ids)
            output_rows.append({
                "plan_id": plan_id,
                "split": deterministic_split(plan_id, args.seed),
                "image_path": str(row["image_path"]),
                "svg_path": str(row["svg_path"]),
                "image_size": list(image_size),
                "rooms": rooms,
                "silver_edges": edges,
                "review": {"status": "pending", "corrections": []},
                "provenance": {
                    "source": "CubiCasa5K model.svg",
                    "svg_sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(),
                    "graph_path": str(row["graph_path"]),
                    "graph_provenance": str(row.get("graph_provenance", "silver")),
                    "split_seed": args.seed,
                },
            })
            print(f"[{index}/{len(rows)}] {plan_id}: {len(rooms)} rooms, {len(edges)} edges")
        except Exception as error:  # Keep an auditable account of exclusions.
            errors.append({"plan_id": plan_id, "error": str(error)})
            print(f"[{index}/{len(rows)}] {plan_id}: ERROR {error}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")
    label_map = {
        "object_categories": [{"id": index + 1, "name": name} for index, name in enumerate(CATEGORIES)],
        "predicate_categories": [{"id": 1, "name": "adjacent_to"}, {"id": 2, "name": "connected_by_door"}, {"id": 3, "name": "open_connected"}],
    }
    args.output.with_name("label_map.json").write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")
    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        args.errors.write_text("".join(json.dumps(row) + "\n" for row in errors), encoding="utf-8")
    print(json.dumps({"written": len(output_rows), "errors": len(errors), "room_counts": Counter(room["category_name"] for row in output_rows for room in row["rooms"])}, indent=2))


if __name__ == "__main__":
    main()
