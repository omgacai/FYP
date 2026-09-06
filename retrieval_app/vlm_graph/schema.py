from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


ROOM_TYPES = {
    "bedroom", "bathroom", "kitchen", "living_room", "dining_room",
    "corridor", "storage", "balcony", "entrance", "other",
}
EDGE_TYPES = {"adjacent_to", "connected_by_door"}


def graph_schema() -> str:
    """Compact schema included in every prompt; keep it small for reproducibility."""
    return json.dumps(
        {
            "rooms": [{"id": "bedroom_1", "type": "bedroom"}],
            "edges": [{
                "source": "bedroom_1", "target": "corridor_1",
                "type": "connected_by_door", "confidence": 0.86,
            }],
        },
        indent=2,
    )


def extract_json(text: str) -> dict[str, Any]:
    """Extract a single JSON object, tolerating accidental markdown fences."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object was found in model output.")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The model output must be a JSON object.")
    return value


def validate_graph(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise the deliberately small experiment schema."""
    rooms = value.get("rooms")
    edges = value.get("edges")
    if not isinstance(rooms, list) or not isinstance(edges, list):
        raise ValueError("Expected top-level 'rooms' and 'edges' arrays.")

    clean_rooms: list[dict[str, str]] = []
    ids: set[str] = set()
    for room in rooms:
        if not isinstance(room, dict):
            raise ValueError("Every room must be an object.")
        room_id, room_type = room.get("id"), room.get("type")
        if not isinstance(room_id, str) or not room_id.strip():
            raise ValueError("Every room needs a non-empty string id.")
        if room_id in ids:
            raise ValueError(f"Duplicate room id: {room_id}")
        if room_type not in ROOM_TYPES:
            raise ValueError(f"Unsupported room type {room_type!r} for {room_id}.")
        ids.add(room_id)
        clean_rooms.append({"id": room_id, "type": room_type})

    clean_edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Every edge must be an object.")
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("type")
        if source not in ids or target not in ids or source == target:
            raise ValueError(f"Edge must connect two different declared rooms: {edge!r}")
        if relation not in EDGE_TYPES:
            raise ValueError(f"Unsupported edge type: {relation!r}")
        key = (*sorted((source, target)), relation)
        if key in seen:
            continue
        seen.add(key)
        confidence = edge.get("confidence", 1.0)
        if not isinstance(confidence, (float, int)) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Edge confidence must be a number in [0, 1].")
        clean_edges.append({"source": source, "target": target, "type": relation, "confidence": float(confidence)})

    return {"rooms": clean_rooms, "edges": clean_edges}


def room_counts(graph: dict[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(room["type"] for room in graph["rooms"]).items()))
