"""Run the local CubiCasa CNN → SVG → CubiGraph pipeline on manual-20.

Creates a self-contained evaluation snapshot. Gold annotations are copied read-only
from the supplied manual root; generated CNN/CubiGraph predictions live under
``predictions/cubicasa_cnn_cubigraph`` and are never presented as gold.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floorplan_app.core.registry import registry
from floorplan_app.pipeline.graph_extractor import extract_graph
from floorplan_app.pipeline.svg_extractor import CUBIGRAPH_NAMES, extract_svg
import floorplan_app.parsers  # noqa: F401 -- register parser


TYPE_MAP = {"LivingRoom": "LivingRoom", "Bedroom": "Bedroom", "Kitchen": "Kitchen", "Dining": "Dining",
            "Bath": "Bath", "Storage": "Storage", "Entry": "Entry", "Garage": "Garage", "Outdoor": "Outdoor",
            "Undefined": "Other"}


def copy_snapshot(source: Path, output: Path, rows: list[dict]) -> None:
    for relative in ["manifest.json", *[row["image_path"] for row in rows],
                     *[f"annotations/{row['plan_id']}.graph.json" for row in rows]]:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


def polygon_points(geometry, width: int, height: int) -> list[list[float]]:
    return [[round(float(x) / width, 7), round(float(y) / height, 7)] for x, y in geometry.exterior.coords[:-1]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manual-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=.20)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    source, output = args.manual_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; use a new directory to preserve predictions.")
    rows = json.loads((source / "manifest.json").read_text())
    if args.limit:
        rows = rows[:args.limit]
    output.mkdir(parents=True)
    copy_snapshot(source, output, rows)
    for folder in ("svg", "graphs", "diagnostics", "predictions/cubicasa_cnn_cubigraph"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    parser = registry.create("CubiCasa5K (pretrained)")
    summary = []
    for position, row in enumerate(rows, start=1):
        plan_id = str(row["plan_id"])
        parsed = parser.parse(source / row["image_path"], max_side=args.max_side, postprocess_threshold=args.threshold)
        svg = extract_svg(parsed, output / "svg" / f"{plan_id}.svg")
        graph = extract_graph(svg, ROOT / "third_party/CubiGraph5K",
                              output / "graphs" / f"{plan_id}_relations.svg")
        width, height = parsed.processed_image.size
        counts: Counter[str] = Counter(); nodes = []
        for geometry, metadata in zip(parsed.room_polygons, parsed.room_metadata):
            source_type = parsed.room_class_names[metadata["class"]]
            kind = TYPE_MAP.get(CUBIGRAPH_NAMES.get(source_type, "Undefined"), "Other")
            counts[kind] += 1
            node_id = f"{kind}_{counts[kind]}"
            points = polygon_points(geometry, width, height)
            if len(points) < 3:
                continue
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            nodes.append({"id": node_id, "type": kind, "label": f"{kind} {counts[kind]}",
                          "x": sum(xs) / len(xs), "y": sum(ys) / len(ys), "polygon": points,
                          "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)]})
        node_ids = {node["id"] for node in nodes}; edges = []; seen = set()
        for a, neighbours in graph.adjacency.items():
            for b, code in neighbours.items():
                key = tuple(sorted((a, b)))
                if key in seen or a not in node_ids or b not in node_ids:
                    continue
                seen.add(key)
                edges.append({"a": a, "b": b, "relation": "connected_by_door" if code == 2 else "adjacent_to"})
        prediction = {"schema_version": "floorplan-manual-graph/2", "plan_id": plan_id, "valid": True,
                      "status": "model_predicted", "provenance": "cubicasa5k_cnn_then_cubigraph",
                      "nodes": nodes, "edges": edges, "all_pairs_reviewed": False, "geometry_complete": True,
                      "geometry_note": "CubiCasa CNN polygons, then deterministic CubiGraph relations; not gold."}
        (output / "predictions/cubicasa_cnn_cubigraph" / f"{plan_id}.json").write_text(json.dumps(prediction, indent=2) + "\n")
        diagnostic = {"plan_id": plan_id, "parser": parsed.diagnostics, "svg": svg.diagnostics, "graph": graph.diagnostics}
        (output / "diagnostics" / f"{plan_id}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        summary.append({"plan_id": plan_id, "nodes": len(nodes), "edges": len(edges), "parser": parsed.diagnostics})
        print(f"[{position}/{len(rows)}] {plan_id}: {len(nodes)} nodes, {len(edges)} edges", flush=True)
    (output / "run.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
        "method": "CubiCasa5K CNN → pseudo-SVG → CubiGraph", "gold_reference": str(source),
        "max_side": args.max_side, "threshold": args.threshold, "plans": summary}, indent=2) + "\n")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
