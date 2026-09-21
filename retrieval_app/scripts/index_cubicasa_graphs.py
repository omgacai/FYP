from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Direct execution via ``python retrieval_app/scripts/...`` otherwise puts only
# this scripts directory on sys.path, hiding the repository packages.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    if len({record["plan_id"] for record in records}) != len(records):
        raise ValueError("Manifest contains duplicate plan IDs")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial.jsonl")
    completed: dict[str, dict] = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["plan_id"]] = record
        unexpected = set(completed) - {record["plan_id"] for record in records}
        if unexpected:
            raise ValueError(f"Partial index does not match this manifest: {sorted(unexpected)[:3]}")
        print(f"Resuming {len(completed)} indexed plans from {partial}")
    enriched: list[dict] = []

    for index, record in enumerate(records, start=1):
        if record["plan_id"] in completed:
            enriched.append(completed[record["plan_id"]])
            continue
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
        with partial.open("a", encoding="utf-8") as checkpoint:
            checkpoint.write(json.dumps(record) + "\n")
            checkpoint.flush()
        print(f"[{index}/{len(records)}] {record['plan_id']}: {graph.diagnostics['nodes']} nodes")

    output.write_text("".join(json.dumps(record) + "\n" for record in enriched), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(f"Wrote {len(enriched)} records to {output}")


if __name__ == "__main__":
    main()
