"""Convert Qwen spatial graphs to the manual-20 normalized graph contract.

The output is a model prediction, never a replacement for the manual reference.
Each predicted box becomes a rectangular polygon because Qwen is asked for boxes,
not room boundaries.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TYPE_MAP = {
    "bedroom": "Bedroom", "bathroom": "Bath", "kitchen": "Kitchen",
    "living_room": "LivingRoom", "dining_room": "Dining", "corridor": "Corridor",
    "storage": "Storage", "balcony": "Outdoor", "entrance": "Entry",
    "garage": "Garage", "outdoor": "Outdoor", "other": "Other",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manual-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.manual_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {str(row["plan_id"]) for row in manifest}
    records = {str(row.get("plan_id")): row for row in read_jsonl(args.predictions.expanduser().resolve())}
    if set(records) != expected:
        raise ValueError(f"Prediction coverage mismatch: missing={sorted(expected - set(records))}, unexpected={sorted(set(records) - expected)}")
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    for plan_id in sorted(expected):
        record = records[plan_id]
        destination = output / f"{plan_id}.json"
        if not record.get("valid") or not isinstance(record.get("graph"), dict):
            destination.write_text(json.dumps({"plan_id": plan_id, "valid": False, "error": record.get("error", "Invalid Qwen JSON"),
                                               "raw_output": record.get("raw_output"), "prompt_system": record.get("prompt_system")}, indent=2) + "\n")
            counts["invalid"] += 1
            continue
        graph = record["graph"]
        nodes, identifiers, type_counts = [], set(), Counter()
        for room in graph["rooms"]:
            room_id = str(room["id"])
            if room_id in identifiers:
                raise ValueError(f"{plan_id}: duplicate Qwen room ID {room_id}")
            identifiers.add(room_id)
            x0, y0, x1, y1 = (float(value) / 1000 for value in room["bbox"])
            kind = TYPE_MAP.get(str(room["type"]), str(room["type"]))
            type_counts[kind] += 1
            nodes.append({"id": room_id, "type": kind, "label": f"{kind} {type_counts[kind]}",
                          "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                          "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "bbox_xyxy": [x0, y0, x1, y1]})
        edges = [{"a": str(edge["source"]), "b": str(edge["target"]), "relation": str(edge["type"]),
                  "confidence": float(edge.get("confidence", 1.0))} for edge in graph["edges"]]
        converted = {"schema_version": "floorplan-manual-graph/2", "plan_id": plan_id, "valid": True,
                     "status": "model_predicted", "provenance": "qwen3vl_image_only_spatial", "nodes": nodes,
                     "edges": edges, "all_pairs_reviewed": False, "geometry_complete": False,
                     "geometry_note": "Qwen-predicted axis-aligned boxes converted to rectangular polygons; not manual geometry.",
                     "raw_output": record.get("raw_output"), "prompt_system": record.get("prompt_system"),
                     "prompt_messages": record.get("prompt_messages"), "model": record.get("model"),
                     "decoded_at": record.get("decoded_at")}
        destination.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
        counts["valid"] += 1
    print(json.dumps({"plans": len(expected), **counts, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
