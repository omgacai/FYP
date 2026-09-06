"""Evaluate direct VLM graph outputs against a manifest's typed graph labels.

For CubiGraph references, metrics are explicitly labelled silver-label metrics.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from retrieval_app.data.manifest import load_adjacency
from retrieval_app.representations.normalise import room_type


RELATIONS = {1: "adjacent_to", 2: "connected_by_door"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reference_graph(graph_path: Path) -> tuple[dict[str, int], set[tuple[str, str, str]]]:
    adjacency = load_adjacency(graph_path)
    counts = Counter(room_type(name) for name in adjacency)
    edges: set[tuple[str, str, str]] = set()
    for source, neighbours in adjacency.items():
        for target, code in neighbours.items():
            relation = RELATIONS.get(code)
            if relation:
                source_type, target_type = sorted((room_type(source), room_type(target)))
                edges.add((source_type, target_type, relation))
    return dict(counts), edges


def predicted_edges(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
    types = {room["id"]: room_type(room["type"]) for room in graph["rooms"]}
    return {(*sorted((types[edge["source"]], types[edge["target"]])), edge["type"]) for edge in graph["edges"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score VLM graph JSON against graph paths in an enriched manifest.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.corpus_root.expanduser().resolve()
    refs = {row["plan_id"]: row for row in read_jsonl(args.manifest.expanduser().resolve())}
    predictions = read_jsonl(args.predictions.expanduser().resolve())
    valid = [row for row in predictions if row.get("valid") and row.get("graph")]
    total_abs_error = 0
    true_positive = false_positive = false_negative = 0
    per_plan = []
    for prediction in valid:
        record = refs.get(prediction["plan_id"])
        if not record or not record.get("graph_path"):
            continue
        graph_path = Path(record["graph_path"])
        graph_path = graph_path if graph_path.is_absolute() else root / graph_path
        expected_counts, expected_edges = reference_graph(graph_path)
        actual_counts = dict(Counter(room_type(room["type"]) for room in prediction["graph"]["rooms"]))
        total_abs_error += sum(abs(actual_counts.get(kind, 0) - expected_counts.get(kind, 0)) for kind in set(actual_counts) | set(expected_counts))
        actual_edges = predicted_edges(prediction["graph"])
        true_positive += len(actual_edges & expected_edges)
        false_positive += len(actual_edges - expected_edges)
        false_negative += len(expected_edges - actual_edges)
        per_plan.append({"plan_id": prediction["plan_id"], "room_count_l1_error": sum(abs(actual_counts.get(kind, 0) - expected_counts.get(kind, 0)) for kind in set(actual_counts) | set(expected_counts)), "edge_tp": len(actual_edges & expected_edges), "edge_fp": len(actual_edges - expected_edges), "edge_fn": len(expected_edges - actual_edges)})

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    summary = {
        "label_provenance": "silver" if any(row.get("graph_provenance") == "silver" for row in refs.values()) else "unknown",
        "prediction_records": len(predictions), "valid_json_rate": len(valid) / len(predictions) if predictions else 0.0,
        "evaluated_records": len(per_plan),
        "mean_room_count_l1_error": total_abs_error / len(per_plan) if per_plan else None,
        "typed_edge_precision": precision, "typed_edge_recall": recall,
        "typed_edge_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "per_plan": per_plan,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_plan"}, indent=2))


if __name__ == "__main__":
    main()
