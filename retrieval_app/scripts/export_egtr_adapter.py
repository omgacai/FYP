"""Convert canonical CubiCasa records to the COCO + rel.json layout EGTR expects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical CubiCasa records to an EGTR adapter dataset.")
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="If given, create output-dir/images symlink to this read-only corpus root.")
    parser.add_argument("--relation-mode", choices=("detailed", "primary_access"), default="primary_access")
    parser.add_argument("--max-objects", type=int, default=200)
    args = parser.parse_args()

    rows = read_jsonl(args.canonical.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    categories: dict[int, str] = {}
    coco: dict[str, dict[str, list[dict[str, Any]]]] = {split: {"images": [], "annotations": []} for split in ("train", "val", "test")}
    relations: dict[str, dict[str, list[list[int]]]] = {split: {} for split in ("train", "val", "test")}
    annotation_id = 0
    for image_id, row in enumerate(rows):
        split = row["split"]
        if split not in coco:
            raise ValueError(f"Unsupported split {split!r} for {row['plan_id']}")
        if len(row["rooms"]) > args.max_objects:
            raise ValueError(f"{row['plan_id']} has {len(row['rooms'])} rooms; exceeds --max-objects={args.max_objects}")
        width, height = row["image_size"]
        coco[split]["images"].append({"id": image_id, "file_name": row["image_path"], "width": width, "height": height})
        room_index: dict[str, int] = {}
        for room_index_value, room in enumerate(row["rooms"]):
            room_index[room["room_id"]] = room_index_value
            x0, y0, x1, y1 = room["bbox_xyxy"]
            categories[int(room["category_id"])] = str(room["category_name"])
            coco[split]["annotations"].append({
                "id": annotation_id, "image_id": image_id, "category_id": int(room["category_id"]),
                "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0,
            })
            annotation_id += 1
        triples: list[list[int]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for edge in row["silver_edges"]:
            pair = tuple(sorted((str(edge["room_a"]), str(edge["room_b"]))))
            if pair in seen_pairs:
                raise ValueError(f"{row['plan_id']} has duplicate/conflicting relations for room pair {pair}")
            seen_pairs.add(pair)
            source, target = room_index[edge["room_a"]], room_index[edge["room_b"]]
            if args.relation_mode == "primary_access":
                predicate = 1 if edge["predicate"] == "adjacent_to" else 2
            else:
                predicate = int(edge["predicate_id"])
            triples.extend([[source, target, predicate], [target, source, predicate]])
        relations[split][str(image_id)] = triples
    for split, payload in coco.items():
        payload["categories"] = [{"id": identifier, "name": categories[identifier]} for identifier in sorted(categories)]
        (output / f"{split}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rel_categories = (["no_relation", "adjacent_to", "direct_access"]
                      if args.relation_mode == "primary_access"
                      else ["no_relation", "adjacent_to", "connected_by_door", "open_connected"])
    rel_payload: dict[str, Any] = {"rel_categories": rel_categories, **relations}
    (output / "rel.json").write_text(json.dumps(rel_payload, indent=2) + "\n", encoding="utf-8")
    provenance = {"schema_version": "cubicasa-egtr-adapter/2", "canonical": str(args.canonical.expanduser().resolve()),
                  "canonical_sha256": hashlib.sha256(args.canonical.expanduser().read_bytes()).hexdigest(),
                  "relation_mode": args.relation_mode, "relation_provenance": "silver; source CubiGraph edges",
                  "negative_label_caveat": "Unlisted pairs are not verified negatives; open passages may be missing from silver supervision."}
    (output / "label_map.json").write_text(json.dumps({"categories": categories, "rel_categories": rel_payload["rel_categories"],
                                                         "provenance": provenance}, indent=2) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if args.image_root:
        target = args.image_root.expanduser().resolve()
        link = output / "images"
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target:
                raise FileExistsError(f"Refusing to replace existing {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    print(json.dumps({"plans": len(rows), "splits": {split: len(value["images"]) for split, value in coco.items()},
                      "relation_mode": args.relation_mode, "output": str(output), "images_linked": bool(args.image_root)}, indent=2))


if __name__ == "__main__":
    main()
