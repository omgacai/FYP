# Exact Qwen prompt: image-only spatial graph baseline

This is the exact prompt content used by the `qwen3vl8b_manual20_direct_spatial`
run. The runner sends the system text below, then attaches the target floorplan
raster as a user image and sends the target instruction. There are no examples,
no source SVG, and no CubiGraph candidate in this condition.

## System message

```text
You are an architectural floorplan parser.
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

Use a normalised 1000 by 1000 canvas: x increases left-to-right and y increases top-to-bottom.
For every room, provide an approximate axis-aligned bbox [x0, y0, x1, y1] and centroid [x, y].
These coordinates must locate the visible room in the input image. Do not output SVG, polygons, or wall coordinates.

Use this exact JSON shape:
{
  "canvas": {
    "width": 1000,
    "height": 1000
  },
  "rooms": [
    {
      "id": "bedroom_1",
      "type": "bedroom",
      "bbox": [80, 520, 410, 850],
      "centroid": [245, 685]
    }
  ],
  "edges": [
    {
      "source": "bedroom_1",
      "target": "corridor_1",
      "type": "connected_by_door",
      "confidence": 0.86
    }
  ]
}
```

## User content

1. The final floorplan raster is attached as an image.
2. The following text is sent with it:

```text
Parse the final floorplan image into the required spatial room graph JSON, including normalised room boxes and centroids.
```

## One bounded repair attempt

If Pydantic or graph validation rejects the initial response, the image is not
sent again. Qwen receives this system text and the prior response/error as a
text-only user message:

```text
[the complete system message above]

Your previous answer was rejected. Return only a corrected object in this exact schema; preserve its intended factual claims but remove Markdown, prose, and unsupported fields.
```

The experiment records the first raw answer, the repair output, the validation
error, and whether the final graph was valid. Pydantic validates the response
after generation; it does not constrain Qwen's decoding while it generates.

## Source of truth

The runnable prompt functions are in
[`retrieval_app/vlm_graph/prompts.py`](../../retrieval_app/vlm_graph/prompts.py).
The strict Pydantic and graph checks are in
[`retrieval_app/vlm_graph/schema.py`](../../retrieval_app/vlm_graph/schema.py).

## Edge-recall follow-up (`edge_recall_v1`)

The original baseline remains unchanged. The follow-up adds this paragraph after
the room/edge rules, before the spatial instructions:

```text
Before writing JSON, silently inspect every plausible pair of declared rooms.
Add an edge for every visibly supported direct relationship: use connected_by_door for a
visible traversable door, open_connected for a direct unobstructed opening, and
adjacent_to for rooms that visibly share a boundary without direct access. Do not omit a
visible shared boundary or doorway merely because the room type is uncertain. Still do
not invent hidden connections, connect rooms through an intervening room, or assign more
than one relation to an unordered room pair.
```

Use it only in a new run named `qwen3vl8b_manual20_direct_spatial_edge_recall_v1`.
It may improve edge recall while reducing precision; compare both metrics and
JSON-validity coverage against the original baseline.
