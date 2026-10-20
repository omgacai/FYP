#!/usr/bin/env python3
"""Build image-only Qwen QA SFT data from CubiGraph silver plan records.

This is deliberately a separate task from raster-to-graph extraction.  It
uses the CubiCasa raster as the only model input and derives deterministic QA
answers from source-SVG rooms and CubiGraph edges.  Manual20 identities are
excluded defensively and are never emitted into train or validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, deque
from pathlib import Path
from typing import Any

from retrieval_app.scripts.build_qwen_graph_sft_dataset import load_exclusions, normalise_identity
from retrieval_app.scripts.generate_manual_graph_qa import (
    COUNTABLE_TYPES,
    RELATIVE_POSITION_MIN_DELTA,
    TYPE_PLURALS,
)
from retrieval_app.scripts.run_qwen_qa import make_messages


SCHEMA_VERSION = "qwen-floorplan-qa-sft/1"
LABEL_TIER = "silver_svg_rooms_cubigraph_edges"
ACCESS_RELATIONS = {"connected_by_door"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def room_label(category: str, index: int) -> str:
    return f"{category} {index}"


def prepare_plan(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]], str | None]:
    """Normalise one canonical record and return a QA-ready room/edge view."""
    source_rooms = record.get("rooms")
    image_size = record.get("image_size")
    if not isinstance(source_rooms, list) or not source_rooms:
        return [], [], "no_nodes"
    if not (isinstance(image_size, list) and len(image_size) == 2 and all(isinstance(v, int) and v > 0 for v in image_size)):
        return [], [], "invalid_image_size"
    width, height = image_size
    counts: Counter[str] = Counter()
    rooms: list[dict[str, Any]] = []
    for room in source_rooms:
        if not isinstance(room, dict):
            return [], [], "invalid_room"
        room_id, category, bbox = room.get("room_id"), room.get("category_name"), room.get("bbox_xyxy")
        if not isinstance(room_id, str) or not isinstance(category, str) or not isinstance(bbox, list) or len(bbox) != 4:
            return [], [], "invalid_room"
        if not all(isinstance(value, (int, float)) for value in bbox):
            return [], [], "invalid_room"
        x0, y0, x1, y1 = (float(value) for value in bbox)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            return [], [], "invalid_room_bbox"
        counts[category] += 1
        rooms.append({"id": room_id, "type": category, "label": room_label(category, counts[category]),
                      "x": (x0 + x1) / (2 * width), "y": (y0 + y1) / (2 * height)})
    weak_fraction = sum(room["type"] in {"Other", "Outdoor"} for room in rooms) / len(rooms)
    if weak_fraction > 0.5:
        return [], [], "other_outdoor_fraction_gt_0_5"
    room_ids = {room["id"] for room in rooms}
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in record.get("silver_edges", []):
        if not isinstance(edge, dict):
            continue
        a, b, relation = edge.get("room_a"), edge.get("room_b"), edge.get("predicate")
        if a not in room_ids or b not in room_ids or a == b or relation not in {"adjacent_to", "connected_by_door"}:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        edges.append({"a": a, "b": b, "relation": relation})
    if not edges:
        return [], [], "no_cubigraph_edges"
    return rooms, edges, None


def relative_position(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    dx, dy = float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])
    if max(abs(dx), abs(dy)) < RELATIVE_POSITION_MIN_DELTA:
        return None
    horizontal = "right" if dx > 0 else "left"
    vertical = "bottom" if dy > 0 else "top"
    if abs(dx) >= RELATIVE_POSITION_MIN_DELTA and abs(dy) >= RELATIVE_POSITION_MIN_DELTA:
        return f"{vertical}_{horizontal}"
    return horizontal if abs(dx) >= abs(dy) else vertical


def answer_json(answer: str, evidence: str, reasoning: str) -> str:
    return json.dumps({"answer": answer, "evidence": [evidence], "reasoning": reasoning}, separators=(",", ":"))


def candidates(plan_id: str, rooms: list[dict[str, Any]], edges: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Generate answerable image-only QA candidates from positive silver facts."""
    by_id = {room["id"]: room for room in rooms}
    result = {name: [] for name in ("count", "direct_access", "relative_position", "compound_access", "shortest_access_path")}
    type_counts = Counter(room["type"] for room in rooms)
    for room_type in sorted(COUNTABLE_TYPES & set(type_counts)):
        count = type_counts[room_type]
        plural = TYPE_PLURALS[room_type]
        result["count"].append({"question": f"How many {plural} are there?", "gold_answer": str(count),
                                "answer_format": "integer", "category": "count", "difficulty": "D0",
                                "target": answer_json(str(count), f"The floor plan contains {count} {plural}.",
                                                      f"Counting the labelled {plural} gives {count}.")})
    access_edges = [edge for edge in edges if edge["relation"] in ACCESS_RELATIONS]
    for edge in access_edges:
        a, b = by_id[edge["a"]], by_id[edge["b"]]
        result["direct_access"].append({"question": f"Is {a['label']} directly accessible from {b['label']}?", "gold_answer": "yes",
                                         "answer_format": "yes_no_unknown", "category": "direct_access", "difficulty": "D1",
                                         "target": answer_json("yes", f"A door connects {a['label']} and {b['label']}.",
                                                               f"{a['label']} and {b['label']} have a direct door connection.")})
    for a in rooms:
        for b in rooms:
            if a["id"] == b["id"]:
                continue
            relation = relative_position(a, b)
            if relation:
                result["relative_position"].append({"question": f"Where is {a['label']} relative to {b['label']}?", "gold_answer": relation,
                                                    "answer_format": "direction", "category": "relative_position", "difficulty": "D2",
                                                    "target": answer_json(relation, f"{a['label']} is {relation.replace('_', ' ')} of {b['label']}.",
                                                                          f"The layout places {a['label']} {relation.replace('_', ' ')} of {b['label']}.")})
    neighbours: dict[str, set[str]] = {room["id"]: set() for room in rooms}
    for edge in access_edges:
        neighbours[edge["a"]].add(edge["b"]); neighbours[edge["b"]].add(edge["a"])
    for anchor in rooms:
        for other_id in sorted(neighbours[anchor["id"]]):
            other = by_id[other_id]
            article = "an" if other["type"] in {"Entry", "Outdoor"} else "a"
            result["compound_access"].append({"question": f"Is there {article} {other['type'].lower()} directly accessible from {anchor['label']}?", "gold_answer": "yes",
                                                "answer_format": "yes_no_unknown", "category": "compound_access", "difficulty": "D3",
                                                "target": answer_json("yes", f"A door connects {anchor['label']} to {other['label']}.",
                                                                      f"{other['label']} is a directly accessible {other['type'].lower()} from {anchor['label']}.")})
    for start in rooms:
        queue: deque[tuple[str, list[str]]] = deque([(start["id"], [start["id"]])])
        visited = {start["id"]}
        while queue:
            current, path = queue.popleft()
            if current != start["id"] and len(path) > 1:
                end = by_id[current]
                transitions = len(path) - 1
                result["shortest_access_path"].append({"question": f"What is the minimum number of direct room-to-room transitions needed to get from {start['label']} to {end['label']}?", "gold_answer": str(transitions),
                                                       "answer_format": "integer", "category": "shortest_access_path", "difficulty": "D4",
                                                       "target": answer_json(str(transitions), f"The direct door route from {start['label']} to {end['label']} has {transitions} transition(s).",
                                                                             f"The shortest direct-door route contains {transitions} transition(s).")})
            for next_id in sorted(neighbours[current]):
                if next_id not in visited:
                    visited.add(next_id); queue.append((next_id, [*path, next_id]))
    return result


def select(candidates_by_category: dict[str, list[dict[str, str]]], plan_id: str, seed: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for category in ("count", "direct_access", "relative_position", "compound_access", "shortest_access_path"):
        choices = candidates_by_category[category]
        if category == "shortest_access_path" and choices:
            longest = max(int(row["gold_answer"]) for row in choices)
            choices = [row for row in choices if int(row["gold_answer"]) == longest]
        if choices:
            selected.append(random.Random(f"{seed}:{plan_id}:{category}").choice(choices))
    return selected


def build_record(plan: dict[str, Any], question: dict[str, str], number: int) -> dict[str, Any]:
    row = {"question": question["question"], "answer_format": question["answer_format"], "category": question["category"]}
    messages = make_messages(row, "image_only", Path(str(plan["image_path"])), None)
    return {"schema_version": SCHEMA_VERSION, "sample_id": f"qa_silver_v1_{number:07d}_{plan['plan_id']}", "plan_id": plan["plan_id"],
            "split": plan["split"], "image_path": plan["image_path"], "question": question["question"],
            "gold_answer": question["gold_answer"], "answer_format": question["answer_format"], "category": question["category"],
            "messages": [*messages, {"role": "assistant", "content": question["target"]}],
            "provenance": {"label_tier": LABEL_TIER, "source_identity": plan["source_identity"], "source_svg": plan.get("svg_path"),
                           "question_generation": "deterministic_cubigraph_v1", "known_limitations": "Only CubiGraph threshold doors are direct-access positives; open passages are not supervised."}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-corpus", type=Path, required=True)
    parser.add_argument("--exclude-identities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20261020)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; choose a new output directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    exclusions = load_exclusions(args.exclude_identities)
    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    excluded: list[dict[str, str]] = []
    plan_question_counts: Counter[str] = Counter()
    for plan in sorted(read_jsonl(args.canonical_corpus.expanduser().resolve()), key=lambda item: str(item.get("plan_id", ""))):
        plan_id, split, identity = plan.get("plan_id"), plan.get("split"), plan.get("source_identity")
        if not isinstance(plan_id, str) or split not in output_rows or not isinstance(identity, str):
            excluded.append({"plan_id": str(plan_id), "reason": "invalid_plan_metadata"}); continue
        if normalise_identity(identity) in exclusions:
            excluded.append({"plan_id": plan_id, "reason": "explicit_manual20_exclusion"}); continue
        rooms, edges, reason = prepare_plan(plan)
        if reason:
            excluded.append({"plan_id": plan_id, "reason": reason}); continue
        selected = select(candidates(plan_id, rooms, edges), plan_id, args.selection_seed)
        if not selected:
            excluded.append({"plan_id": plan_id, "reason": "no_answerable_qa_pairs"}); continue
        plan_question_counts[len(selected)] += 1
        for question in selected:
            output_rows[split].append(build_record(plan, question, len(output_rows[split]) + 1))
    for split, rows in output_rows.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "excluded_plans.jsonl").write_text("".join(json.dumps(row) + "\n" for row in excluded), encoding="utf-8")
    metadata = {"schema_version": SCHEMA_VERSION, "label_tier": LABEL_TIER, "source_canonical_corpus": str(args.canonical_corpus.expanduser()),
                "source_canonical_corpus_sha256": hashlib.sha256(args.canonical_corpus.expanduser().read_bytes()).hexdigest(),
                "excluded_identities_sha256": hashlib.sha256(args.exclude_identities.expanduser().read_bytes()).hexdigest(),
                "selection_seed": args.selection_seed, "records": {split: len(rows) for split, rows in output_rows.items()},
                "plans_by_questions_emitted": dict(sorted(plan_question_counts.items())), "excluded_plans": len(excluded),
                "manual20_policy": "All listed Manual20 identities are excluded before QA generation.",
                "cleaning_policy": "Drop no nodes, no CubiGraph edges, or Other+Outdoor fraction > 0.5.",
                "relation_policy": "CubiGraph v1: only connected_by_door is direct access; open_connected is not supervised."}
    (output / "dataset_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
