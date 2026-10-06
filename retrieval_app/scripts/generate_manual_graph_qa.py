#!/usr/bin/env python3
"""Create deterministic QA candidates from manual floor-plan graph annotations.

The generator deliberately treats the manual graph as the sole answer source.
It never infers a negative direct-access/adjacency relation from a missing edge
unless ``all_pairs_reviewed`` is true.  Every CSV row retains the graph evidence
needed for a human reviewer to confirm or reject the candidate before it is
used in a benchmark.

Example:
    python3 -m retrieval_app.scripts.generate_manual_graph_qa \
      --input-dir cubicasa_benchmark/annotations \
      --output cubicasa_benchmark/questions/manual20_qa_v1.csv \
      --per-plan 5
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GENERATOR_VERSION = "manual-graph-qa/1"
ACCESS_RELATIONS = {"connected_by_door", "open_connected"}
COUNTABLE_TYPES = {"Bedroom", "Bath", "Kitchen", "LivingRoom", "Dining", "Corridor", "Entry"}
TYPE_PLURALS = {
    "Bedroom": "bedrooms",
    "Bath": "bathrooms",
    "Kitchen": "kitchens",
    "LivingRoom": "living rooms",
    "Dining": "dining rooms",
    "Corridor": "corridors",
    "Entry": "entries",
}


@dataclass(frozen=True)
class Room:
    id: str
    room_type: str
    label: str
    x: float
    y: float
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Relation:
    a: str
    b: str
    relation: str

    @property
    def evidence_id(self) -> str:
        return "--".join(sorted((self.a, self.b))) + f":{self.relation}"


@dataclass
class Plan:
    plan_id: str
    annotation_file: Path
    all_pairs_reviewed: bool
    connectivity_passed: bool
    rooms: list[Room]
    relations: list[Relation]

    def room_by_id(self) -> dict[str, Room]:
        return {room.id: room for room in self.rooms}


CSV_FIELDS = [
    "question_id",
    "plan_id",
    "category",
    "difficulty",
    "question",
    "gold_answer",
    "answer_format",
    "answerability",
    "supporting_node_ids",
    "supporting_edge_ids",
    "supporting_path_ids",
    "source_rule",
    "annotation_file",
    "generator_version",
    "needs_human_review",
]


def json_cell(value: Iterable[str]) -> str:
    return json.dumps(list(value), separators=(",", ":"))


def polygon_area(points: tuple[tuple[float, float], ...]) -> float | None:
    """Return normalised polygon area; invalid/missing polygons have no area."""
    if len(points) < 3:
        return None
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0])):
        total += x1 * y2 - x2 * y1
    area = abs(total) / 2.0
    return area if area > 0 else None


def load_plan(path: Path) -> Plan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "floorplan-manual-graph/2":
        raise ValueError(f"{path}: expected floorplan-manual-graph/2")
    if raw.get("status") != "manually_reviewed":
        raise ValueError(f"{path}: annotation is not manually reviewed")
    plan_id = raw.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError(f"{path}: missing plan_id")

    rooms: list[Room] = []
    for index, node in enumerate(raw.get("nodes", []), start=1):
        room_id = node.get("id")
        room_type = node.get("type")
        if not isinstance(room_id, str) or not isinstance(room_type, str):
            raise ValueError(f"{path}: node {index} needs id and type")
        label = node.get("label") if isinstance(node.get("label"), str) else f"{room_type} {index}"
        polygon = tuple(
            (float(point[0]), float(point[1]))
            for point in node.get("polygon", [])
            if isinstance(point, list) and len(point) == 2
        )
        rooms.append(Room(room_id, room_type, label, float(node.get("x", 0)), float(node.get("y", 0)), polygon))

    known_ids = {room.id for room in rooms}
    relations: list[Relation] = []
    for edge in raw.get("edges", []):
        a, b, relation = edge.get("a"), edge.get("b"), edge.get("relation")
        if a not in known_ids or b not in known_ids or a == b or not isinstance(relation, str):
            raise ValueError(f"{path}: invalid edge {edge!r}")
        relations.append(Relation(a, b, relation))

    connectivity = raw.get("connectivity_check") or {}
    return Plan(
        plan_id=plan_id,
        annotation_file=path,
        all_pairs_reviewed=raw.get("all_pairs_reviewed") is True,
        connectivity_passed=connectivity.get("status") == "pass",
        rooms=rooms,
        relations=relations,
    )


def candidate(plan: Plan, *, category: str, difficulty: str, question: str, answer: str,
              answer_format: str, node_ids: Iterable[str], edge_ids: Iterable[str] = (),
              path_ids: Iterable[str] = (), source_rule: str, answerability: str = "answerable") -> dict[str, str]:
    return {
        "question_id": "",  # assigned only after deterministic selection
        "plan_id": plan.plan_id,
        "category": category,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": answer,
        "answer_format": answer_format,
        "answerability": answerability,
        "supporting_node_ids": json_cell(node_ids),
        "supporting_edge_ids": json_cell(edge_ids),
        "supporting_path_ids": json_cell(path_ids),
        "source_rule": source_rule,
        "annotation_file": str(plan.annotation_file),
        "generator_version": GENERATOR_VERSION,
        "needs_human_review": "true",
    }


def access_adjacency(plan: Plan) -> dict[str, list[tuple[str, Relation]]]:
    result = {room.id: [] for room in plan.rooms}
    for edge in plan.relations:
        if edge.relation in ACCESS_RELATIONS:
            result[edge.a].append((edge.b, edge))
            result[edge.b].append((edge.a, edge))
    return result


def shortest_path(neighbors: dict[str, list[tuple[str, Relation]]], start: str, end: str) -> tuple[list[str], list[Relation]] | None:
    queue: deque[str] = deque([start])
    previous: dict[str, tuple[str | None, Relation | None]] = {start: (None, None)}
    while queue:
        current = queue.popleft()
        if current == end:
            break
        for next_id, edge in neighbors[current]:
            if next_id not in previous:
                previous[next_id] = (current, edge)
                queue.append(next_id)
    if end not in previous:
        return None
    nodes: list[str] = []
    edges: list[Relation] = []
    current: str | None = end
    while current is not None:
        nodes.append(current)
        prior, edge = previous[current]
        if edge is not None:
            edges.append(edge)
        current = prior
    return list(reversed(nodes)), list(reversed(edges))


def generate_candidates(plan: Plan, include_unknown: bool) -> dict[str, list[dict[str, str]]]:
    """Create candidates in fixed buckets; selection later enforces diversity."""
    rooms = plan.room_by_id()
    buckets: dict[str, list[dict[str, str]]] = {"D0": [], "D1": [], "D2": [], "D3": [], "D4": []}
    counts = Counter(room.room_type for room in plan.rooms)

    # D0 — facts that do not require topology.
    for room_type in sorted(COUNTABLE_TYPES & set(counts)):
        plural = TYPE_PLURALS[room_type]
        buckets["D0"].append(candidate(
            plan, category="count", difficulty="D0",
            question=f"How many {plural} are there?", answer=str(counts[room_type]),
            answer_format="integer", node_ids=[room.id for room in plan.rooms if room.room_type == room_type],
            source_rule=f"count_rooms(type={room_type})",
        ))

    # D1 — positive single-edge questions.  Negative claims are intentionally
    # not inferred from missing edges in current annotations.
    for edge in sorted(plan.relations, key=lambda item: (item.relation, item.evidence_id)):
        if edge.relation not in ACCESS_RELATIONS | {"adjacent_to"}:
            continue
        a, b = rooms[edge.a], rooms[edge.b]
        if edge.relation in ACCESS_RELATIONS:
            text = f"Is {a.label} directly accessible from {b.label}?"
            category, rule = "direct_access", "has_direct_access"
        else:
            text = f"Does {a.label} share a boundary with {b.label}?"
            category, rule = "adjacency", "has_adjacency"
        buckets["D1"].append(candidate(
            plan, category=category, difficulty="D1", question=text, answer="yes", answer_format="yes_no_unknown",
            node_ids=[a.id, b.id], edge_ids=[edge.evidence_id], source_rule=f"{rule}({a.id},{b.id})",
        ))

    # Optional unknown questions are useful only as an explicit coverage test:
    # they do not silently turn an unreviewed absent edge into a false claim.
    if include_unknown and not plan.all_pairs_reviewed:
        related_pairs = {tuple(sorted((edge.a, edge.b))) for edge in plan.relations}
        for index, a in enumerate(plan.rooms):
            for b in plan.rooms[index + 1:]:
                if tuple(sorted((a.id, b.id))) not in related_pairs:
                    buckets["D1"].append(candidate(
                        plan, category="direct_access", difficulty="D1",
                        question=f"Is {a.label} directly accessible from {b.label}?", answer="unknown",
                        answer_format="yes_no_unknown", node_ids=[a.id, b.id],
                        source_rule=f"missing_relation_with_incomplete_pair_review({a.id},{b.id})",
                        answerability="unknown",
                    ))

    # D2 — relative direction, only when the room centres are clearly separated.
    for index, a in enumerate(plan.rooms):
        for b in plan.rooms[index + 1:]:
            dx, dy = a.x - b.x, a.y - b.y
            if abs(dx) >= 0.18 and abs(dx) >= abs(dy):
                answer = "left" if dx < 0 else "right"
            elif abs(dy) >= 0.18:
                answer = "above" if dy < 0 else "below"
            else:
                continue
            buckets["D2"].append(candidate(
                plan, category="relative_position", difficulty="D2",
                question=f"Where is {a.label} relative to {b.label}?", answer=answer,
                answer_format="direction", node_ids=[a.id, b.id],
                source_rule=f"centroid_direction({a.id},{b.id},min_delta=0.18)",
            ))

    # D3 — existential, typed access constraint.  It uses a relation plus room type.
    neighbors = access_adjacency(plan)
    for anchor in plan.rooms:
        for target_type in sorted(COUNTABLE_TYPES):
            matches = [(target_id, edge) for target_id, edge in neighbors[anchor.id] if rooms[target_id].room_type == target_type]
            if not matches:
                continue
            match_ids = [target_id for target_id, _ in matches]
            edge_ids = [edge.evidence_id for _, edge in matches]
            article = "an" if target_type[0].lower() in "aeiou" else "a"
            buckets["D3"].append(candidate(
                plan, category="compound_access", difficulty="D3",
                question=f"Is there {article} {target_type.lower()} directly accessible from {anchor.label}?",
                answer="yes", answer_format="yes_no_unknown", node_ids=[anchor.id, *match_ids], edge_ids=edge_ids,
                source_rule=f"exists_direct_access(anchor={anchor.id},type={target_type})",
            ))

    # D4 — shortest access paths.  Only create them if the annotation's own
    # connectivity review passed, so a disconnected/incomplete topology does
    # not masquerade as an answerable navigation question.
    if plan.connectivity_passed:
        entries = [room for room in plan.rooms if room.room_type == "Entry"]
        targets = [room for room in plan.rooms if room.room_type in {"Bedroom", "Bath", "Kitchen"}]
        for entry in entries:
            for target in targets:
                result = shortest_path(neighbors, entry.id, target.id)
                if result is None:
                    continue
                path_nodes, path_edges = result
                if len(path_edges) < 2:  # D1 already covers direct access.
                    continue
                buckets["D4"].append(candidate(
                    plan, category="shortest_access_path", difficulty="D4",
                    question=f"What is the minimum number of direct room-to-room transitions needed to get from {entry.label} to {target.label}?",
                    answer=str(len(path_edges)), answer_format="integer", node_ids=path_nodes,
                    edge_ids=[edge.evidence_id for edge in path_edges], path_ids=path_nodes,
                    source_rule=f"shortest_access_path(start={entry.id},end={target.id})",
                ))
    return buckets


def pick_per_plan(buckets: dict[str, list[dict[str, str]]], per_plan: int) -> list[dict[str, str]]:
    """Select one item per difficulty first, then deterministic fallbacks."""
    selected: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for difficulty in ("D0", "D1", "D2", "D3", "D4"):
        # A direct-access relation has priority over shared-wall adjacency for
        # the single D1 slot because it is the more informative QA condition.
        rows = buckets[difficulty]
        if difficulty == "D1":
            rows = sorted(rows, key=lambda row: (0 if row["category"] == "direct_access" else 1, row["question"]))
        for row in rows:
            if row["question"] not in seen_questions:
                selected.append(row)
                seen_questions.add(row["question"])
                break
        if len(selected) == per_plan:
            return selected
    for difficulty in ("D1", "D2", "D3", "D4", "D0"):
        rows = buckets[difficulty]
        if difficulty == "D1":
            rows = sorted(rows, key=lambda row: (0 if row["category"] == "direct_access" else 1, row["question"]))
        for row in rows:
            if row["question"] in seen_questions:
                continue
            selected.append(row)
            seen_questions.add(row["question"])
            if len(selected) == per_plan:
                return selected
    return selected


def generate(input_dir: Path, output: Path, per_plan: int, include_unknown: bool) -> tuple[int, list[str]]:
    if per_plan < 1:
        raise ValueError("--per-plan must be at least 1")
    annotations = sorted(input_dir.glob("*.graph.json"))
    if not annotations:
        raise ValueError(f"No *.graph.json annotations found in {input_dir}")
    selected_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    for path in annotations:
        plan = load_plan(path)
        rows = pick_per_plan(generate_candidates(plan, include_unknown), per_plan)
        if len(rows) < per_plan:
            warnings.append(f"{plan.plan_id}: generated {len(rows)}/{per_plan} candidates")
        selected_rows.extend(rows)

    for number, row in enumerate(selected_rows, start=1):
        row["question_id"] = f"qa_v1_{number:04d}_{row['plan_id']}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(selected_rows)
    return len(selected_rows), warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing *.graph.json manual annotations.")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path.")
    parser.add_argument("--per-plan", type=int, default=5, help="Target number of diverse questions per plan (default: 5).")
    parser.add_argument("--include-unknown", action="store_true", help="Add explicit unknown-access candidates for plans without complete pair review.")
    args = parser.parse_args()
    count, warnings = generate(args.input_dir, args.output, args.per_plan, args.include_unknown)
    print(f"Wrote {count} QA candidates to {args.output}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
