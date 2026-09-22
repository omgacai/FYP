# Qwen3-VL image-only graph benchmark on manual-20

## Question

Can Qwen3-VL-8B infer room instances and access/adjacency relations directly
from the held-out floorplan raster, in the same graph contract used to score
EGTR against the manually reviewed 20-plan reference set?

## Condition

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Input: raster only; no `model.svg`, CubiGraph candidate, or few-shot support.
- Decoding: deterministic (`do_sample=False`), maximum 1,024 generated tokens,
  maximum 1,048,576 image pixels, at most one text-only JSON repair attempt.
- Output: a 1,000 × 1,000 normalized spatial graph. Qwen proposes room boxes,
  types, and edges. A converter writes rectangular box polygons in the
  `floorplan-manual-graph/2` prediction schema; these are model geometry, not
  manual annotations.
- Parallelism: four disjoint five-plan shards; at most two Slurm array tasks
  execute at once. Shards are merged only if they cover each plan exactly once.

## Exact system prompt

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
These coordinates must locate the visible room in the input image. Do not output SVG,
polygons, or wall coordinates.
```

The target instruction is: `Parse the final floorplan image into the required
spatial room graph JSON, including normalised room boxes and centroids.` Each
raw response, validated graph, and full prompt-message payload is saved in the
Qwen shard output.

## Metrics

All graph metrics use the existing manual evaluator: class-blind one-to-one
room matching at IoU >= 0.30, then room type and edge scoring. Door and open
relations collapse to `direct_access` for primary metrics, matching EGTR.

1. JSON validity rate and graph-evaluation coverage.
2. Micro and macro room detection precision, recall, and F1.
3. Matched-room type accuracy, matched-node count, and mean matched-box IoU.
4. Micro and macro edge precision, recall, and F1.
5. Per-relation precision, recall, and F1 for `direct_access` and
   `adjacent_to`.
6. Mean aligned graph edit cost and normalized edit cost per plan.

Compare the generated `qwen_manual20_metrics.json` with the corresponding
`results/graph_metrics.json` from frozen/fine-tuned EGTR runs. Report coverage
with every score; invalid Qwen JSON is not silently removed.
