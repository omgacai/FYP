from __future__ import annotations

from retrieval_app.vlm_graph.schema import graph_schema


SYSTEM_PROMPT = """You are an architectural floorplan parser.
Return exactly one valid JSON object and nothing else. Do not use Markdown.

Identify enclosed rooms from the supplied floorplan image. Use only these room types:
bedroom, bathroom, kitchen, living_room, dining_room, corridor, storage, balcony,
entrance, other.

Use adjacent_to only when two rooms visibly share a boundary. Use connected_by_door
only when a visible doorway, door swing, or clear passage connects the two rooms.
Do not infer hidden doors. When uncertain, omit the edge rather than guessing.
Room IDs must be unique and every edge must refer to two declared rooms.

Use this exact JSON shape:\n""" + graph_schema()


def target_instruction() -> str:
    return "Parse the final floorplan image into the required room graph JSON."


def support_instruction(graph_json: str) -> str:
    return "Example floorplan and its verified room graph JSON:\n" + graph_json
