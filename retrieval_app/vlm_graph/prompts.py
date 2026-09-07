from __future__ import annotations

from retrieval_app.vlm_graph.schema import graph_schema


def system_prompt(spatial: bool = False) -> str:
    prompt = """You are an architectural floorplan parser.
Return exactly one valid JSON object and nothing else. Do not use Markdown.

Identify enclosed rooms from the supplied floorplan image. Use only these room types:
bedroom, bathroom, kitchen, living_room, dining_room, corridor, storage, balcony,
entrance, other.

Use adjacent_to only when two rooms visibly share a boundary. Use connected_by_door
only when a visible doorway, door swing, or clear passage connects the two rooms.
Do not infer hidden doors. When uncertain, omit the edge rather than guessing.
Room IDs must be unique and every edge must refer to two declared rooms.

"""
    if spatial:
        prompt += """Use a normalised 1000 by 1000 canvas: x increases left-to-right and y increases top-to-bottom.
For every room, provide an approximate axis-aligned bbox [x0, y0, x1, y1] and centroid [x, y].
These coordinates must locate the visible room in the input image. Do not output SVG, polygons, or wall coordinates.

"""
    return prompt + "Use this exact JSON shape:\n" + graph_schema(spatial=spatial)


SYSTEM_PROMPT = system_prompt()


def target_instruction(spatial: bool = False) -> str:
    if spatial:
        return "Parse the final floorplan image into the required spatial room graph JSON, including normalised room boxes and centroids."
    return "Parse the final floorplan image into the required room graph JSON."


def support_instruction(graph_json: str) -> str:
    return "Example floorplan and its verified room graph JSON:\n" + graph_json
