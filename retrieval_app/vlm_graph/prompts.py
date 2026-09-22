from __future__ import annotations

from retrieval_app.vlm_graph.schema import correction_patch_schema, graph_schema


def system_prompt(spatial: bool = False, prompt_version: str = "baseline") -> str:
    prompt = """You are an architectural floorplan parser.
Return exactly one valid JSON object and nothing else. Do not use Markdown.

Identify enclosed rooms from the supplied floorplan image. Use only these room types:
bedroom, bathroom, kitchen, living_room, dining_room, corridor, storage, balcony,
entrance, garage, outdoor, other.

"""
    if prompt_version == "edge_recall_json_v1":
        prompt += """Use adjacent_to when two rooms visibly share a boundary. Use connected_by_door
when a visible doorway or door swing connects the rooms. open_connected means two distinct
room zones have a direct, unobstructed opening without a door. Silently inspect every
plausible pair of declared rooms and include every visible relationship; do not leave out
a visible shared boundary, doorway, or open passage. Do not infer hidden doors, connect
rooms through an intervening room, or assign more than one relation to an unordered pair.
Room IDs must be unique and every edge must refer to two declared rooms.

Before answering, silently check that the result is one parseable JSON object: use double
quotes around every key and string, commas between array/object items, no trailing commas,
and no text before or after the closing brace.

"""
    else:
        prompt += """Use adjacent_to only when two rooms visibly share a boundary. Use connected_by_door
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
    if prompt_version == "edge_recall_v1":
        prompt += """Before writing JSON, silently inspect every plausible pair of declared rooms.
Add an edge for every visibly supported direct relationship: use connected_by_door for a
visible traversable door, open_connected for a direct unobstructed opening, and
adjacent_to for rooms that visibly share a boundary without direct access. Do not omit a
visible shared boundary or doorway merely because the room type is uncertain. Still do
not invent hidden connections, connect rooms through an intervening room, or assign more
than one relation to an unordered room pair.

"""
    if spatial:
        prompt += """Use a normalised 1000 by 1000 canvas: x increases left-to-right and y increases top-to-bottom.
For every room, provide an approximate axis-aligned bbox [x0, y0, x1, y1] and centroid [x, y].
These coordinates must locate the visible room in the input image. Do not output SVG, polygons, or wall coordinates.

"""
    if prompt_version not in {"baseline", "cubicasa_fewshot_v1", "edge_recall_v1", "edge_recall_json_v1"}:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")
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
    return """CubiGraph supplied this rule-derived SILVER candidate for the target plan. Treat its room IDs and source-SVG-derived boxes as the default structure. A room ID such as Other_1 is an opaque stable identifier, NOT its semantic type: never rename it. Existing candidate edges are retained by default. Only propose a removal when the image clearly contradicts the edge; lack of clear evidence is not evidence to remove it. If one supplied room visibly contains two or more distinct rooms, you may use room_splits to replace only that source room with 2–4 new boxed child rooms; otherwise do not create, delete, rename, or move rooms. Return only corrections, omitting all retained values. CubiGraph codes are 1=adjacent_to, 2=connected_by_door, 3=open_connected.\n""" + graph_json


def correction_system_prompt() -> str:
    return """You are reviewing a CubiCasa floorplan graph.
Return exactly one valid JSON object and nothing else. Do not use Markdown.

The target image is accompanied by a CubiGraph candidate. Its room IDs, boxes, and
centroids come from source model.svg and must be retained by default. Room IDs are
opaque identities: `Other_1` may be a kitchen, bedroom, or any other semantic type.
Never rename an ID; use room_type_updates to correct its type. Review every candidate
room's semantic type, and when a room currently typed `other` has clear visual evidence
for a supported type, emit a room_type_update. Propose only corrections with visible
image evidence. Classify primarily from geometry, walls,
doors, fixtures, and furniture; readable room text may support but must not override
contradictory visual evidence.
Use only these room types: bedroom, bathroom, kitchen, living_room, dining_room,
corridor, storage, balcony, entrance, garage, outdoor, other.

Use room_splits only when one candidate box visibly covers 2–4 distinct rooms. Replace
that source room with new, unique child IDs and normalised 1000 × 1000 boxes/centroids.
Do not split a room merely because it has furniture, a small alcove, or an uncertain
internal boundary. A split automatically removes the source room's old edges, so add
new edge decisions only where direct visual evidence supports them.

For an edge decision, use action=set with exactly one predicate, or action=remove.
Use connected_by_door only for a visible traversable door, open_connected only for a
direct unobstructed opening, and adjacent_to for a shared boundary with neither kind
of direct access. Each unordered pair has at most one relation. Existing candidate
edges are a conservative prior: omit them to retain them. Use action=remove only when
there is strong visible evidence that the two candidate rooms have no direct relation;
do not remove an edge merely because its door/opening is faint, occluded, or uncertain.
Do not guess hidden doors or add speculative edges.

Use this exact JSON shape:\n""" + correction_patch_schema()


def correction_target_instruction() -> str:
    return "Inspect the final floorplan and return a correction patch for the fixed CubiGraph candidate JSON only."
