"""Merge Qwen JSONL shards and verify exactly one record for every target plan."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shards", required=True, help="Quoted glob for shard JSONL files.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = [row["plan_id"] for row in rows(args.manifest)]
    files = [Path(path) for path in sorted(glob.glob(args.shards))]
    if not files:
        raise FileNotFoundError(f"No shard files match {args.shards!r}")
    merged = [row for path in files for row in rows(path)]
    ids = [str(row.get("plan_id")) for row in merged]
    duplicates = sorted({plan_id for plan_id in ids if ids.count(plan_id) > 1})
    missing, unexpected = sorted(set(expected) - set(ids)), sorted(set(ids) - set(expected))
    if duplicates or missing or unexpected:
        raise ValueError(f"Shard coverage mismatch: duplicates={duplicates}, missing={missing}, unexpected={unexpected}")
    by_id = {str(row["plan_id"]): row for row in merged}
    ordered = [by_id[plan_id] for plan_id in expected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in ordered), encoding="utf-8")
    print(json.dumps({"plans": len(ordered), "valid": sum(bool(row.get("valid")) for row in ordered),
                      "invalid": sum(not bool(row.get("valid")) for row in ordered), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
