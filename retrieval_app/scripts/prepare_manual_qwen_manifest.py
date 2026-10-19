"""Convert a manual benchmark bundle into Qwen's JSONL input format.

Both the historic ``manifest.json`` package and the current review-react
``annotations/`` + ``images/`` layout are supported.  The latter is the source
of truth for the frozen Manual20 test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-root", type=Path, required=True, help="Folder containing images/ and annotations/ (or a legacy manifest.json).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.manual_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("manual manifest must be a non-empty JSON array")
    else:
        rows = []
        for annotation_path in sorted((root / "annotations").glob("*.graph.json")):
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            plan_id = annotation.get("plan_id")
            image = annotation.get("image")
            image_path = image.get("local_path") if isinstance(image, dict) else None
            if not isinstance(plan_id, str) or not plan_id:
                raise ValueError(f"{annotation_path}: missing plan_id")
            if annotation.get("status") != "manually_reviewed":
                raise ValueError(f"{annotation_path}: annotation is not manually reviewed")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"{annotation_path}: missing image.local_path")
            rows.append({"plan_id": plan_id, "image_path": f"images/{Path(image_path).name}"})
        if not rows:
            raise ValueError(f"{root}: need manifest.json or reviewed annotations/*.graph.json")
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
