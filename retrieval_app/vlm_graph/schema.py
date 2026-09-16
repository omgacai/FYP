from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROOM_TYPES = {
    "bedroom", "bathroom", "kitchen", "living_room", "dining_room",
    "corridor", "storage", "balcony", "entrance", "garage", "outdoor", "other",
}
EDGE_TYPES = {"adjacent_to", "connected_by_door", "open_connected"}


class _StrictPayload(BaseModel):
    """Reject accidental prose keys and unsupported model fields early."""

    model_config = ConfigDict(extra="forbid")


class CanvasPayload(_StrictPayload):
    width: int
    height: int


class RoomPayload(_StrictPayload):
    id: str = Field(min_length=1)
    type: str
    bbox: list[float] | None = None
    centroid: list[float] | None = None


class EdgePayload(_StrictPayload):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphPayload(_StrictPayload):
    canvas: CanvasPayload | None = None
    rooms: list[RoomPayload]
    edges: list[EdgePayload]


class TypeUpdatePayload(_StrictPayload):
    room_id: str = Field(min_length=1)
    type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EdgeDecisionPayload(_StrictPayload):
    room_a: str = Field(min_length=1)
    room_b: str = Field(min_length=1)
    action: str
    predicate: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CorrectionPatchPayload(_StrictPayload):
    room_type_updates: list[TypeUpdatePayload] = Field(default_factory=list)
    edge_decisions: list[EdgeDecisionPayload] = Field(default_factory=list)


def correction_patch_schema() -> str:
    """Schema for correcting a graph whose room geometry/identity is fixed."""
    return json.dumps({
        "room_type_updates": [{"room_id": "Other_1", "type": "living_room", "confidence": 0.86}],
        "edge_decisions": [{
            "room_a": "Other_1", "room_b": "Other_3", "action": "set",
            "predicate": "connected_by_door", "confidence": 0.91,
        }],
    }, indent=2)


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
    # Pydantic gives a precise error for malformed field shapes, extra prose
    # fields, bad confidence values, and missing top-level arrays before the
    # graph-specific topology checks below.
    value = GraphPayload.model_validate(value).model_dump(exclude_none=False)
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
    # The dataset contract permits at most one topology label for an unordered
    # room pair.  This rejects the otherwise tempting but contradictory output
    # of marking the same pair both adjacent and directly connected.
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Every edge must be an object.")
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("type")
        if source not in ids or target not in ids or source == target:
            raise ValueError(f"Edge must connect two different declared rooms: {edge!r}")
        if relation not in EDGE_TYPES:
            raise ValueError(f"Unsupported edge type: {relation!r}")
        key = tuple(sorted((source, target)))
        if key in seen:
            raise ValueError(f"Duplicate or conflicting relation for room pair: {source!r}, {target!r}")
        seen.add(key)
        confidence = edge.get("confidence", 1.0)
        if not isinstance(confidence, (float, int)) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Edge confidence must be a number in [0, 1].")
        clean_edges.append({"source": source, "target": target, "type": relation, "confidence": float(confidence)})

    result: dict[str, Any] = {"rooms": clean_rooms, "edges": clean_edges}
    if canvas is not None:
        result["canvas"] = canvas
    return result


def validate_correction_patch(value: dict[str, Any], fixed_room_ids: set[str]) -> dict[str, Any]:
    """Validate an additive VLM proposal against immutable source room nodes."""
    value = CorrectionPatchPayload.model_validate(value).model_dump(exclude_none=False)
    updates = value.get("room_type_updates", [])
    decisions = value.get("edge_decisions", [])
    if not isinstance(updates, list) or not isinstance(decisions, list):
        raise ValueError("Correction patches need room_type_updates and edge_decisions arrays.")

    clean_updates: list[dict[str, Any]] = []
    seen_rooms: set[str] = set()
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError("Every room_type_update must be an object.")
        room_id, room_type = update.get("room_id"), update.get("type")
        if room_id not in fixed_room_ids:
            raise ValueError(f"Unknown fixed room id in type update: {room_id!r}")
        if room_id in seen_rooms:
            raise ValueError(f"Duplicate type update for room: {room_id!r}")
        if room_type not in ROOM_TYPES:
            raise ValueError(f"Unsupported room type in correction patch: {room_type!r}")
        confidence = update.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Patch confidence must be in [0, 1].")
        seen_rooms.add(room_id)
        clean_updates.append({"room_id": room_id, "type": room_type, "confidence": float(confidence)})

    clean_decisions: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Every edge_decision must be an object.")
        room_a, room_b, action = decision.get("room_a"), decision.get("room_b"), decision.get("action")
        if room_a not in fixed_room_ids or room_b not in fixed_room_ids or room_a == room_b:
            raise ValueError("Every patch edge must use two different fixed room ids.")
        key = tuple(sorted((room_a, room_b)))
        if key in seen_pairs:
            raise ValueError(f"Duplicate edge decision for room pair: {key!r}")
        if action not in {"set", "remove"}:
            raise ValueError("Patch edge action must be 'set' or 'remove'.")
        predicate = decision.get("predicate")
        if action == "set" and predicate not in EDGE_TYPES:
            raise ValueError("A set edge decision needs a supported predicate.")
        if action == "remove":
            predicate = None
        confidence = decision.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Patch confidence must be in [0, 1].")
        seen_pairs.add(key)
        clean = {"room_a": room_a, "room_b": room_b, "action": action, "confidence": float(confidence)}
        if predicate is not None:
            clean["predicate"] = predicate
        clean_decisions.append(clean)
    return {"room_type_updates": clean_updates, "edge_decisions": clean_decisions}


def graph_from_fixed_candidate(candidate: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a validated patch without ever changing source-derived geometry."""
    rooms = [dict(room) for room in candidate["rooms"]]
    room_by_id = {room["id"]: room for room in rooms}
    for update in patch["room_type_updates"]:
        room_by_id[update["room_id"]]["type"] = update["type"]

    edge_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in candidate["edges"]:
        edge_by_pair[tuple(sorted((edge["source"], edge["target"])))] = dict(edge)
    for decision in patch["edge_decisions"]:
        key = tuple(sorted((decision["room_a"], decision["room_b"])))
        if decision["action"] == "remove":
            edge_by_pair.pop(key, None)
        else:
            edge_by_pair[key] = {
                "source": key[0], "target": key[1], "type": decision["predicate"],
                "confidence": decision["confidence"],
            }
    return validate_graph({"canvas": candidate["canvas"], "rooms": rooms, "edges": list(edge_by_pair.values())}, require_spatial=True)


def cubigraph_adjacency(graph: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Export canonical undirected labels in the compact CubiGraph JSON form."""
    codes = {"adjacent_to": 1, "connected_by_door": 2, "open_connected": 3}
    adjacency = {room["id"]: {} for room in graph["rooms"]}
    for edge in graph["edges"]:
        source, target, code = edge["source"], edge["target"], codes[edge["type"]]
        adjacency[source][target] = code
        adjacency[target][source] = code
    return adjacency


def room_counts(graph: dict[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(room["type"] for room in graph["rooms"]).items()))
