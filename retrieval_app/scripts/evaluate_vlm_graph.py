"""Evaluate VLM graph outputs against silver graph labels.

Semantic-only graphs can be scored only by room-type pair. Spatial graphs may
also be matched to canonical source-SVG boxes for genuine instance-edge scores.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from retrieval_app.data.manifest import load_adjacency
from retrieval_app.representations.normalise import room_type

RELATIONS = {1: "adjacent_to", 2: "connected_by_door", 3: "open_connected"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def type_reference(graph_path: Path) -> tuple[dict[str, int], set[tuple[str, str, str]]]:
    adjacency = load_adjacency(graph_path)
    counts = Counter(room_type(name) for name in adjacency)
    edges = {
        (*sorted((room_type(source), room_type(target))), RELATIONS[code])
        for source, neighbours in adjacency.items() for target, code in neighbours.items()
        if code in RELATIONS
    }
    return dict(counts), edges


def type_predictions(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
    types = {room["id"]: room_type(room["type"]) for room in graph["rooms"]}
    return {(*sorted((types[edge["source"]], types[edge["target"]])), edge["type"]) for edge in graph["edges"]}


def box_iou(left: list[float], right: list[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def spatial_instance_predictions(graph: dict[str, Any], canonical: dict[str, Any], min_iou: float) -> tuple[set[tuple[str, str, str]], int]:
    """Greedily create a one-to-one, same-class source-room assignment."""
    width, height = canonical["image_size"]
    candidates: list[tuple[float, str, str]] = []
    for predicted in graph["rooms"]:
        if "bbox" not in predicted:
            raise ValueError("Spatial instance metrics require bbox on every predicted room.")
        x0, y0, x1, y1 = predicted["bbox"]
        prediction_box = [x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000]
        for reference in canonical["rooms"]:
            if room_type(predicted["type"]) == room_type(reference["category_name"]):
                candidates.append((box_iou(prediction_box, reference["bbox_xyxy"]), predicted["id"], reference["room_id"]))
    matches: dict[str, str] = {}
    used_predictions: set[str] = set()
    used_references: set[str] = set()
    for overlap, predicted_id, reference_id in sorted(candidates, reverse=True):
        if overlap < min_iou or predicted_id in used_predictions or reference_id in used_references:
            continue
        matches[predicted_id] = reference_id
        used_predictions.add(predicted_id)
        used_references.add(reference_id)
    edges = set()
    for edge in graph["edges"]:
        source = matches.get(edge["source"], f"<unmatched:{edge['source']}>")
        target = matches.get(edge["target"], f"<unmatched:{edge['target']}>")
        edges.add((*sorted((source, target)), edge["type"]))
    return edges, len(matches)


def score(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score VLM graph JSON against an enriched CubiCasa manifest.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, help="Optional canonical JSONL for spatial instance metrics.")
    parser.add_argument("--min-iou", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 < args.min_iou <= 1:
        parser.error("--min-iou must be in (0, 1].")

    root = args.corpus_root.expanduser().resolve()
    references = {row["plan_id"]: row for row in read_jsonl(args.manifest.expanduser().resolve())}
    canonicals = {row["plan_id"]: row for row in read_jsonl(args.canonical.expanduser().resolve())} if args.canonical else {}
    predictions = read_jsonl(args.predictions.expanduser().resolve())
    valid = [row for row in predictions if row.get("valid") and row.get("graph")]
    count_error = type_tp = type_fp = type_fn = instance_tp = instance_fp = instance_fn = matched_rooms = instance_plans = 0
    per_plan: list[dict[str, Any]] = []
    for prediction in valid:
        reference = references.get(prediction["plan_id"])
        if not reference or not reference.get("graph_path"):
            continue
        graph_path = Path(reference["graph_path"])
        graph_path = graph_path if graph_path.is_absolute() else root / graph_path
        expected_counts, expected_edges = type_reference(graph_path)
        actual_counts = Counter(room_type(room["type"]) for room in prediction["graph"]["rooms"])
        error = sum(abs(actual_counts.get(name, 0) - expected_counts.get(name, 0)) for name in set(actual_counts) | set(expected_counts))
        count_error += error
        actual_edges = type_predictions(prediction["graph"])
        item: dict[str, Any] = {"plan_id": prediction["plan_id"], "room_count_l1_error": error, "coarse_edge_tp": len(actual_edges & expected_edges), "coarse_edge_fp": len(actual_edges - expected_edges), "coarse_edge_fn": len(expected_edges - actual_edges)}
        type_tp += item["coarse_edge_tp"]; type_fp += item["coarse_edge_fp"]; type_fn += item["coarse_edge_fn"]
        canonical = canonicals.get(prediction["plan_id"])
        if canonical and all("bbox" in room for room in prediction["graph"]["rooms"]):
            actual_instance_edges, matched = spatial_instance_predictions(prediction["graph"], canonical, args.min_iou)
            expected_instance_edges = {(*sorted((edge["room_a"], edge["room_b"])), edge["predicate"]) for edge in canonical["silver_edges"]}
            item.update({"instance_edge_tp": len(actual_instance_edges & expected_instance_edges), "instance_edge_fp": len(actual_instance_edges - expected_instance_edges), "instance_edge_fn": len(expected_instance_edges - actual_instance_edges), "matched_room_instances": matched})
            instance_tp += item["instance_edge_tp"]; instance_fp += item["instance_edge_fp"]; instance_fn += item["instance_edge_fn"]
            matched_rooms += matched; instance_plans += 1
        per_plan.append(item)
    summary: dict[str, Any] = {
        "label_provenance": "silver" if any(row.get("graph_provenance") == "silver" for row in references.values()) else "unknown",
        "prediction_records": len(predictions), "valid_json_rate": len(valid) / len(predictions) if predictions else 0.0,
        "evaluated_records": len(per_plan), "mean_room_count_l1_error": count_error / len(per_plan) if per_plan else None,
        "coarse_type_pair_edge_metrics": score(type_tp, type_fp, type_fn),
        "coarse_metric_warning": "Semantic-only graphs are compared by room-type pairs; this is not room-instance edge accuracy.",
        "per_plan": per_plan,
    }
    if args.canonical:
        summary["spatial_instance_edge_metrics"] = {**score(instance_tp, instance_fp, instance_fn), "plans": instance_plans, "matched_room_instances": matched_rooms, "min_iou": args.min_iou}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_plan"}, indent=2))


if __name__ == "__main__":
    main()
