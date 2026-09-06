from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def find_image(sample_dir: Path) -> Path | None:
    preferred = sample_dir / "F1_scaled.png"
    if preferred.exists():
        return preferred
    return next(iter(sorted(sample_dir.glob("*.png"))), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CubiCasa image/SVG JSONL manifest.")
    parser.add_argument("--cubicasa-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for a smoke-test corpus.")
    args = parser.parse_args()

    root = args.cubicasa_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for svg_path in sorted(root.rglob("model.svg")):
        image_path = find_image(svg_path.parent)
        if image_path is None:
            continue
        relative_dir = svg_path.parent.relative_to(root).as_posix()
        identifier = hashlib.sha1(relative_dir.encode("utf-8")).hexdigest()[:12]
        records.append({
            "plan_id": f"cubicasa-{identifier}",
            "image_path": image_path.relative_to(root).as_posix(),
            "svg_path": svg_path.relative_to(root).as_posix(),
            "graph_provenance": "unavailable",
            "metadata": {"dataset": "CubiCasa5K", "source_dir": relative_dir},
        })
        if args.limit and len(records) >= args.limit:
            break
    output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
