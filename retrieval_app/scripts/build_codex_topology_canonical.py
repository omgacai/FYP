#!/usr/bin/env python3
"""Convert the frozen Codex 100-plan annotation bundle to QA canonical JSONL.

The source bundle retains raw images and graph proposals.  This script makes a
separate canonical training view, preserves the original plan-level split, and
keeps the ten internal-holdout plans out of SFT train/validation files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TYPE_MAP = {
    "bedroom": "Bedroom", "bathroom": "Bath", "kitchen": "Kitchen",
    "living_room": "LivingRoom", "dining_room": "Dining", "corridor": "Corridor",
    "storage": "Storage", "entrance": "Entry", "garage": "Garage",
    "outdoor": "Outdoor", "other": "Other", "balcony": "Balcony",
}
EDGE_TYPES = {"adjacent_to", "connected_by_door", "open_connected"}
SPLIT_MAP = {"train": "train", "validation": "val"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_record(source: dict[str, Any], graph: dict[str, Any], *, image_dir_name: str) -> dict[str, Any]:
    plan_id = source["plan_id"]
    canvas = graph.get("canvas", {})
    width, height = canvas.get("width"), canvas.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"{plan_id}: graph canvas must have positive integer width/height")
    rooms = []
    room_ids = set()
    for room in graph.get("rooms", []):
        room_id, room_type, bbox = room.get("id"), room.get("type"), room.get("bbox")
        category = TYPE_MAP.get(room_type)
        if not isinstance(room_id, str) or not room_id or room_id in room_ids:
            raise ValueError(f"{plan_id}: rooms need unique IDs")
        if category is None:
            raise ValueError(f"{plan_id}: unsupported Codex room type {room_type!r}")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, (int, float)) for v in bbox):
            raise ValueError(f"{plan_id}: invalid room bbox for {room_id}")
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"{plan_id}: out-of-canvas room bbox for {room_id}")
        room_ids.add(room_id)
        rooms.append({"room_id": room_id, "category_name": category, "bbox_xyxy": bbox})
    if not rooms:
        raise ValueError(f"{plan_id}: graph has no rooms")
    edges = []
    seen_pairs = set()
    for edge in graph.get("edges", []):
        a, b, predicate = edge.get("source"), edge.get("target"), edge.get("type")
        if a not in room_ids or b not in room_ids or a == b or predicate not in EDGE_TYPES:
            raise ValueError(f"{plan_id}: invalid graph edge {edge!r}")
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            raise ValueError(f"{plan_id}: multiple relations for room pair {pair!r}")
        seen_pairs.add(pair)
        edges.append({
            "room_a": a, "room_b": b, "predicate": predicate,
            "confidence": edge.get("confidence"), "provenance": "codex", "review_status": "unreviewed",
        })
    if not edges:
        raise ValueError(f"{plan_id}: graph has no edges")
    return {
        "schema_version": "codex-topology-canonical/1",
        "plan_id": plan_id,
        "split": SPLIT_MAP[source["assigned_split"]],
        "source_identity": source["source_dir"],
        "image_path": f"codex_topology_100/{image_dir_name}/{plan_id}.png",
        "image_size": [width, height],
        "rooms": rooms,
        "reviewed_edges": edges,
        "topology_coverage": "partial",
        "label_tier": "codex_silver",
        "review_status": "unreviewed",
        "provenance": {
            "annotation_source": "codex",
            "selection_manifest_plan_id": plan_id,
            "source_partition": source["source_partition"],
            "source_dir": source["source_dir"],
            "graph_file": f"graphs/{plan_id}.graph.json",
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-dir-name", default="images", help="Remote directory name used under codex_topology_100/")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("selection manifest must contain exactly 100 records")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    canonical, holdout, plan_ids = [], [], set()
    for source in sorted(records, key=lambda item: item["plan_id"]):
        plan_id, assigned = source.get("plan_id"), source.get("assigned_split")
        if not isinstance(plan_id, str) or plan_id in plan_ids:
            raise ValueError(f"invalid/duplicate plan id {plan_id!r}")
        plan_ids.add(plan_id)
        image = args.images_dir / f"{plan_id}.png"
        graph_path = args.graphs_dir / f"{plan_id}.graph.json"
        if not image.is_file() or not graph_path.is_file():
            raise FileNotFoundError(f"{plan_id}: missing {'image' if not image.is_file() else 'graph'}")
        if assigned == "internal_holdout":
            holdout.append({**source, "graph_file": str(graph_path), "image_file": str(image)})
        elif assigned in SPLIT_MAP:
            canonical.append(canonical_record(source, read_json(graph_path), image_dir_name=args.image_dir_name))
        else:
            raise ValueError(f"{plan_id}: unexpected assigned split {assigned!r}")
    write_jsonl(output / "canonical.jsonl", canonical)
    write_jsonl(output / "internal_holdout.jsonl", holdout)
    metadata = {
        "schema_version": "codex-topology-canonical/1",
        "source_manifest": str(args.manifest), "source_manifest_sha256": sha256(args.manifest),
        "records": len(canonical), "splits": {"train": sum(r["split"] == "train" for r in canonical), "val": sum(r["split"] == "val" for r in canonical)},
        "internal_holdout_plans": len(holdout), "label_tier": "codex_silver",
        "topology_coverage": "partial",
        "policy": "Explicit door/open edges are positive evidence only; missing edges remain unknown.",
    }
    (output / "canonical_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
