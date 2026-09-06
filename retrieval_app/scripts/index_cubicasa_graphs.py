from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from floorplan_app.core.models import SVGResult
from floorplan_app.pipeline.graph_extractor import extract_graph
from retrieval_app.representations.normalise import room_type


def _resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive versioned CubiGraph JSON from CubiCasa source SVGs and write a new manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--cubigraph-repo", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL manifest; source manifest is preserved.")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit.")
    args = parser.parse_args()

    corpus_root = args.corpus_root.expanduser().resolve()
    graph_dir = args.graph_dir.expanduser().resolve()
    graph_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.manifest.expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
    enriched: list[dict] = []

    for index, record in enumerate(records, start=1):
        if args.limit and index > args.limit:
            enriched.extend(records[index - 1:])
            break
        svg_path = _resolve(record["svg_path"], corpus_root)
        if not svg_path.exists():
            raise FileNotFoundError(f"Missing SVG for {record['plan_id']}: {svg_path}")
        relation_svg = graph_dir / f"{record['plan_id']}_relations.svg"
        graph_json = graph_dir / f"{record['plan_id']}.json"
        graph = extract_graph(
            SVGResult(svg_text=svg_path.read_text(encoding="utf-8"), path=svg_path, diagnostics={"source": "CubiCasa ground-truth SVG"}),
            args.cubigraph_repo.expanduser().resolve(),
            relation_svg,
        )
        graph_json.write_text(json.dumps(graph.adjacency, indent=2), encoding="utf-8")
        counts = Counter(room_type(node) for node in graph.adjacency)
        record = dict(record)
        record["graph_path"] = _relative_or_absolute(graph_json, corpus_root)
        record["room_counts"] = dict(sorted(counts.items()))
        record["graph_provenance"] = "silver"
        record.setdefault("metadata", {})["graph_extractor"] = graph.diagnostics["relation_policy"]
        enriched.append(record)
        print(f"[{index}/{len(records)}] {record['plan_id']}: {graph.diagnostics['nodes']} nodes")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record) + "\n" for record in enriched), encoding="utf-8")
    print(f"Wrote {len(enriched)} records to {output}")


if __name__ == "__main__":
    main()
