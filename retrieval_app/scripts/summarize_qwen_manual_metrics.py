"""Combine Qwen JSON-validity and manual-graph metrics into one comparison file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Merged raw Qwen JSONL.")
    parser.add_argument("--graph-metrics", type=Path, required=True, help="Manual evaluator graph_metrics.json.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, graph = read_jsonl(args.predictions.expanduser().resolve()), json.loads(args.graph_metrics.expanduser().read_text())
    output = {"method": "qwen3vl_image_only_spatial", "prediction_records": len(rows),
              "valid_json": sum(bool(row.get("valid")) for row in rows),
              "valid_json_rate": sum(bool(row.get("valid")) for row in rows) / len(rows) if rows else 0.0,
              "model": sorted({str(row.get("model")) for row in rows}),
              "prompt_system": rows[0].get("prompt_system") if rows else None,
              "manual_graph_metrics": graph}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prediction_records": output["prediction_records"], "valid_json_rate": output["valid_json_rate"],
                      "node_f1": graph.get("micro_nodes", {}).get("f1"), "edge_f1": graph.get("micro_edges", {}).get("f1")}, indent=2))


if __name__ == "__main__":
    main()
