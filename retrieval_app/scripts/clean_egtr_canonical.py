"""Validate a canonical EGTR corpus and write a cleaned, auditable copy.

The original CubiCasa corpus and input canonical file are never modified.  A
record is rejected when it cannot form a valid detector target: no rooms,
invalid/out-of-canvas boxes, duplicate room identifiers, or edges whose
endpoints are absent after validation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def rows(path: Path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            yield line_number, json.loads(line)


def error_for(record: dict) -> str | None:
    plan_id = record.get("plan_id", "<unknown>")
    rooms = record.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        return "no_source_rooms"
    size = record.get("image_size")
    if not isinstance(size, list) or len(size) != 2 or not all(isinstance(value, (int, float)) and value > 0 for value in size):
        return "invalid_image_size"
    width, height = size
    identifiers = set()
    for index, room in enumerate(rooms):
        room_id = room.get("room_id")
        if not isinstance(room_id, str) or not room_id or room_id in identifiers:
            return f"invalid_or_duplicate_room_id:{index}"
        identifiers.add(room_id)
        box = room.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
            return f"invalid_room_box:{room_id}"
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            return f"out_of_canvas_or_degenerate_box:{room_id}"
    for edge in record.get("silver_edges", []):
        if edge.get("room_a") not in identifiers or edge.get("room_b") not in identifiers:
            return f"edge_endpoint_missing:{edge}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    args = parser.parse_args()
    valid, rejected = [], []
    for line_number, record in rows(args.input.expanduser().resolve()):
        issue = error_for(record)
        if issue:
            rejected.append({"line": line_number, "plan_id": record.get("plan_id"),
                             "source_identity": record.get("source_identity"), "image_path": record.get("image_path"),
                             "reason": issue})
        else:
            valid.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in valid), encoding="utf-8")
    args.rejections.write_text("".join(json.dumps(record) + "\n" for record in rejected), encoding="utf-8")
    print(json.dumps({"input_records": len(valid) + len(rejected), "written": len(valid),
                      "rejected": len(rejected), "rejection_reasons": Counter(row["reason"] for row in rejected),
                      "output": str(args.output), "rejections": str(args.rejections)}, indent=2))


if __name__ == "__main__":
    main()
