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


def graph_schema(spatial: bool = False) -> str:
    """Compact schema included in every prompt; keep it small for reproducibility."""
    value: dict[str, Any] = {
        "rooms": [{"id": "bedroom_1", "type": "bedroom"}],
        "edges": [{
            "source": "bedroom_1", "target": "corridor_1",
            "type": "connected_by_door", "confidence": 0.86,
        }],
    }
    if spatial:
        value = {
            "canvas": {"width": 1000, "height": 1000},
            "rooms": [{
                "id": "bedroom_1", "type": "bedroom",
                "bbox": [80, 520, 410, 850], "centroid": [245, 685],
            }],
            "edges": value["edges"],
        }
    return json.dumps(value, indent=2)


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


def validate_graph(value: dict[str, Any], require_spatial: bool = False) -> dict[str, Any]:
    """Validate and canonicalise the deliberately small experiment schema."""
    rooms = value.get("rooms")
    edges = value.get("edges")
    if not isinstance(rooms, list) or not isinstance(edges, list):
        raise ValueError("Expected top-level 'rooms' and 'edges' arrays.")

    canvas: dict[str, int] | None = None
    if require_spatial:
        raw_canvas = value.get("canvas")
        if not isinstance(raw_canvas, dict) or raw_canvas.get("width") != 1000 or raw_canvas.get("height") != 1000:
            raise ValueError("Spatial graphs require canvas {width: 1000, height: 1000}.")
        canvas = {"width": 1000, "height": 1000}

    clean_rooms: list[dict[str, Any]] = []
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
        clean_room: dict[str, Any] = {"id": room_id, "type": room_type}
        if require_spatial:
            bbox, centroid = room.get("bbox"), room.get("centroid")
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(item, (int, float)) for item in bbox):
                raise ValueError(f"Spatial room {room_id} needs bbox [x0, y0, x1, y1].")
            x0, y0, x1, y1 = (float(item) for item in bbox)
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                raise ValueError(f"Spatial room {room_id} has an invalid bbox range.")
            if not isinstance(centroid, list) or len(centroid) != 2 or not all(isinstance(item, (int, float)) for item in centroid):
                raise ValueError(f"Spatial room {room_id} needs centroid [x, y].")
            cx, cy = (float(item) for item in centroid)
            if not (0 <= cx <= 1000 and 0 <= cy <= 1000):
                raise ValueError(f"Spatial room {room_id} has an invalid centroid range.")
            clean_room["bbox"] = [x0, y0, x1, y1]
            clean_room["centroid"] = [cx, cy]
        clean_rooms.append(clean_room)

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

    result: dict[str, Any] = {"rooms": clean_rooms, "edges": clean_edges}
    if canvas is not None:
        result["canvas"] = canvas
    return result


def room_counts(graph: dict[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(room["type"] for room in graph["rooms"]).items()))
