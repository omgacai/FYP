#!/usr/bin/env python3
"""Build deterministic Qwen-VL SFT data from SVG-derived CubiCasa silver graphs.

The input is the canonical JSONL exported by ``build_cubicasa_egtr_corpus``.
Images are model inputs; SVG and rule-derived data are labels only.  Manual20
or any other explicitly excluded source identity is never emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-floorplan-graph-sft/1"
LABEL_TIER = "silver_svg_rooms_cubigraph_edges"
TYPE_MAP = {
    "Bedroom": "bedroom", "Bath": "bathroom", "Kitchen": "kitchen",
    "LivingRoom": "living_room", "Dining": "dining_room", "Corridor": "corridor",
    "Storage": "storage", "Entry": "entrance", "Garage": "garage",
    "Outdoor": "outdoor", "Other": "other",
}
EDGE_TYPES = {"adjacent_to", "connected_by_door", "open_connected"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def load_exclusions(path: Path) -> set[str]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("--exclude-identities must be a JSON array of source-plan identity strings")
    return {item.strip("/") for item in value}


def normalise_bbox(box: Any, image_size: Any) -> list[float]:
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) for value in box):
        raise ValueError("room bbox_xyxy must contain four numbers")
    if not isinstance(image_size, list) or len(image_size) != 2 or not all(isinstance(value, int) and value > 0 for value in image_size):
        raise ValueError("image_size must be [positive width, positive height]")
    width, height = image_size
    x0, y0, x1, y1 = (float(value) for value in box)
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError(f"bbox_xyxy must lie inside image_size; got {box!r} for {image_size!r}")
    return [round(x0 * 1000 / width, 2), round(y0 * 1000 / height, 2),
            round(x1 * 1000 / width, 2), round(y1 * 1000 / height, 2)]


def canonical_graph(record: dict[str, Any]) -> dict[str, Any]:
    image_size = record.get("image_size")
    prepared = []
    for room in record.get("rooms", []):
        if not isinstance(room, dict):
            raise ValueError("every room must be an object")
        source_id = room.get("room_id")
        room_type = TYPE_MAP.get(room.get("category_name"))
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("every room needs room_id")
        if room_type is None:
            raise ValueError(f"unsupported room category {room.get('category_name')!r}")
        bbox = normalise_bbox(room.get("bbox_xyxy"), image_size)
        prepared.append((bbox[1], bbox[0], bbox[3], bbox[2], room_type, source_id, bbox))
    if not prepared:
        raise ValueError("record has no rooms")
    prepared.sort()
    source_to_target: dict[str, str] = {}
    rooms = []
    for number, (_, _, _, _, room_type, source_id, bbox) in enumerate(prepared, start=1):
        if source_id in source_to_target:
            raise ValueError(f"duplicate source room ID {source_id!r}")
        target_id = f"R{number}"
        source_to_target[source_id] = target_id
        rooms.append({
            "id": target_id,
            "type": room_type,
            "bbox": bbox,
            "centroid": [round((bbox[0] + bbox[2]) / 2, 2), round((bbox[1] + bbox[3]) / 2, 2)],
        })
    edges: dict[tuple[str, str], str] = {}
    for edge in record.get("silver_edges", []):
        if not isinstance(edge, dict):
            raise ValueError("every silver edge must be an object")
        source, target, relation = edge.get("room_a"), edge.get("room_b"), edge.get("predicate")
        if source not in source_to_target or target not in source_to_target or source == target:
            raise ValueError(f"edge has invalid source-room endpoints: {edge!r}")
        if relation not in EDGE_TYPES:
            raise ValueError(f"unsupported silver predicate {relation!r}")
        pair = tuple(sorted((source_to_target[source], source_to_target[target]), key=lambda item: int(item[1:])))
        prior = edges.get(pair)
        if prior is not None and prior != relation:
            raise ValueError(f"conflicting relations for target pair {pair!r}")
        edges[pair] = relation
    return {
        "canvas": {"width": 1000, "height": 1000},
        "rooms": rooms,
        "edges": [{"source": source, "target": target, "type": relation, "confidence": 1.0}
                  for (source, target), relation in sorted(edges.items(), key=lambda item: (int(item[0][0][1:]), int(item[0][1][1:]), item[1]))],
    }


def system_prompt() -> str:
    return (
        "You are an architectural floorplan parser. Return exactly one valid JSON object and nothing else. "
        "From the supplied floorplan image, identify enclosed rooms and direct room relations. "
        "Use room types bedroom, bathroom, kitchen, living_room, dining_room, corridor, storage, balcony, "
        "entrance, garage, outdoor, or other. Use a normalised 1000 by 1000 canvas; x increases left-to-right "
        "and y increases top-to-bottom. Every room needs an approximate bbox [x0,y0,x1,y1] and centroid [x,y]. "
        "Use adjacent_to for a shared boundary, connected_by_door for a visible door, and open_connected for a "
        "direct unobstructed opening. Use at most one relation per unordered pair."
    )


def make_record(record: dict[str, Any]) -> dict[str, Any]:
    plan_id, split, image_path, identity = (record.get("plan_id"), record.get("split"), record.get("image_path"), record.get("source_identity"))
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("missing plan_id")
    if split not in {"train", "val"}:
        raise ValueError(f"{plan_id}: split must be train or val")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"{plan_id}: missing image_path")
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"{plan_id}: missing source_identity")
    target = canonical_graph(record)
    target_text = json.dumps(target, separators=(",", ":"), ensure_ascii=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": plan_id,
        "plan_id": plan_id,
        "split": split,
        "image_path": image_path,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Parse this floorplan image into the required spatial room graph JSON."},
            ]},
            {"role": "assistant", "content": target_text},
        ],
        "target_graph": target,
        "provenance": {
            "label_tier": LABEL_TIER,
            "source_identity": identity,
            "source_svg": record.get("svg_path"),
            "source_graph": record.get("provenance", {}).get("graph_path"),
            "source_record_provenance": record.get("provenance", {}),
            "known_limitations": "CubiGraph source-SVG rule labels; open-connected positives and absent-edge negatives may be incomplete.",
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-corpus", type=Path, required=True, help="JSONL from build_cubicasa_egtr_corpus.py")
    parser.add_argument("--exclude-identities", type=Path, required=True, help="Manual/calibration source identities excluded from all SFT splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Deterministic smoke-test record limit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.canonical_corpus.expanduser().resolve()
    exclusions = load_exclusions(args.exclude_identities)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {destination}; use --overwrite to replace it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    source_rows = sorted(read_jsonl(source), key=lambda row: str(row.get("plan_id", "")))
    if args.limit:
        source_rows = source_rows[:args.limit]
    records, excluded, errors = [], [], []
    for row in source_rows:
        identity = row.get("source_identity")
        if isinstance(identity, str) and identity.strip("/") in exclusions:
            excluded.append({"plan_id": row.get("plan_id"), "source_identity": identity, "reason": "explicit_training_exclusion"})
            continue
        try:
            records.append(make_record(row))
        except (TypeError, ValueError) as error:
            errors.append({"plan_id": row.get("plan_id"), "error": str(error)})
    plan_ids = [record["plan_id"] for record in records]
    if len(set(plan_ids)) != len(plan_ids):
        raise ValueError("duplicate plan_id after conversion")
    by_split = {split: [record for record in records if record["split"] == split] for split in ("train", "val")}
    if not by_split["train"] or not by_split["val"]:
        raise ValueError("both train and val need at least one valid, non-excluded record")
    write_jsonl(destination / "train.jsonl", by_split["train"])
    write_jsonl(destination / "val.jsonl", by_split["val"])
    write_jsonl(destination / "excluded.jsonl", excluded)
    write_jsonl(destination / "errors.jsonl", errors)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "label_tier": LABEL_TIER,
        "source_canonical_corpus": str(source),
        "source_canonical_corpus_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "excluded_identities_sha256": hashlib.sha256(args.exclude_identities.expanduser().read_bytes()).hexdigest(),
        "records": {"train": len(by_split["train"]), "val": len(by_split["val"]), "excluded": len(excluded), "errors": len(errors)},
        "room_types": dict(sorted(Counter(room["type"] for record in records for room in record["target_graph"]["rooms"]).items())),
        "edge_types": dict(sorted(Counter(edge["type"] for record in records for edge in record["target_graph"]["edges"]).items())),
        "manual20_policy": "All identities in --exclude-identities are excluded from training and validation.",
    }
    (destination / "dataset_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
