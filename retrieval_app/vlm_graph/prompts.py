from __future__ import annotations

from retrieval_app.vlm_graph.schema import graph_schema


def system_prompt(spatial: bool = False, prompt_version: str = "baseline") -> str:
    prompt = """You are an architectural floorplan parser.
Return exactly one valid JSON object and nothing else. Do not use Markdown.

Identify enclosed rooms from the supplied floorplan image. Use only these room types:
bedroom, bathroom, kitchen, living_room, dining_room, corridor, storage, balcony,
entrance, garage, outdoor, other.

Use adjacent_to only when two rooms visibly share a boundary. Use connected_by_door
only when a visible doorway or door swing connects the two rooms. Do not infer hidden
doors. open_connected means two distinct room zones have a direct, unobstructed opening
without a door. Every unordered pair may have at most one edge. When uncertain, omit
the edge rather than guessing.
Room IDs must be unique and every edge must refer to two declared rooms.

"""
    if prompt_version == "cubicasa_fewshot_v1":
        prompt += """The examples are CubiCasa-style plans. Learn their annotation convention,
but inspect the target image yourself. Reason silently in this order: (1) locate room
zones and walls, (2) find visible door swings/open passages, (3) test only direct room
pairs, and (4) write the final JSON. A shared wall without access is adjacent_to; a
door always takes precedence over adjacency; an unblocked opening is open_connected.
Never connect rooms merely because they are nearby, overlap in their bounding boxes, or
can be reached through another room.

"""
    if spatial:
        prompt += """Use a normalised 1000 by 1000 canvas: x increases left-to-right and y increases top-to-bottom.
For every room, provide an approximate axis-aligned bbox [x0, y0, x1, y1] and centroid [x, y].
These coordinates must locate the visible room in the input image. Do not output SVG, polygons, or wall coordinates.

"""
    return prompt + "Use this exact JSON shape:\n" + graph_schema(spatial=spatial)


SYSTEM_PROMPT = system_prompt()


def target_instruction(spatial: bool = False, prompt_version: str = "baseline") -> str:
    if spatial:
        suffix = " First reason silently, then return JSON only." if prompt_version == "cubicasa_fewshot_v1" else ""
        return "Parse the final floorplan image into the required spatial room graph JSON, including normalised room boxes and centroids." + suffix
    return "Parse the final floorplan image into the required room graph JSON."


def support_instruction(graph_json: str) -> str:
    return "Verified CubiCasa-style example. Its image and graph are paired evidence; copy its annotation rules, not its room IDs or layout:\n" + graph_json


def silver_correction_instruction(graph_json: str) -> str:
    return """CubiGraph supplied this rule-derived SILVER candidate graph for the target plan. It may contain missing, false, or wrongly typed links. Inspect the target image yourself and return a complete corrected graph in the required schema. Do not repeat a candidate edge unless visible image evidence supports it. CubiGraph codes are 1=adjacent_to, 2=connected_by_door, 3=open_connected.\n""" + graph_json
