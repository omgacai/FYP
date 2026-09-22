"""Convert a packaged manual-20 manifest.json into Qwen's JSONL input format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-root", type=Path, required=True, help="Folder containing manifest.json, images/, and annotations/.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.manual_root.expanduser().resolve()
    rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("manual manifest must be a non-empty JSON array")
    output = []
    for row in rows:
        plan_id, image_path = str(row["plan_id"]), str(row["image_path"])
        annotation = root / "annotations" / f"{plan_id}.graph.json"
        if not (root / image_path).is_file() or not annotation.is_file():
            raise FileNotFoundError(f"{plan_id}: missing packaged image or annotation")
        output.append({"plan_id": plan_id, "image_path": image_path, "manual_annotation": f"annotations/{plan_id}.graph.json"})
    if len({row["plan_id"] for row in output}) != len(output):
        raise ValueError("manual manifest contains duplicate plan IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"plans": len(output), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
