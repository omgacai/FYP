#!/usr/bin/env python3
"""Create a Stage-2 QA SFT dataset with deterministic Stage-1 replay.

The Stage-2 validation rows stay Codex-only.  Replay is train-only and excludes
every Codex plan identity, so no plan occurs on both sides of the split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True, help="Completed Stage-1 SFT dataset directory")
    parser.add_argument("--stage2", type=Path, required=True, help="Fresh Codex SFT dataset directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-fraction", type=float, default=0.25, help="Fraction of final training rows sourced from Stage 1")
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if not 0 < args.replay_fraction < 0.5:
        parser.error("--replay-fraction must be between 0 and 0.5")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    stage1, stage2 = (args.stage1.expanduser().resolve(), args.stage2.expanduser().resolve())
    s1_train = read_jsonl(stage1 / "train.jsonl")
    s2_train, s2_val = read_jsonl(stage2 / "train.jsonl"), read_jsonl(stage2 / "val.jsonl")
    codex_ids = {row["plan_id"] for row in [*s2_train, *s2_val]}
    if {row["plan_id"] for row in s2_train} & {row["plan_id"] for row in s2_val}:
        raise ValueError("Stage-2 train and validation plans overlap")
    candidates = [row for row in s1_train if row.get("plan_id") not in codex_ids]
    by_plan: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_plan.setdefault(row["plan_id"], []).append(row)
    target = round(len(s2_train) * args.replay_fraction / (1 - args.replay_fraction))
    plan_ids = sorted(by_plan)
    random.Random(args.seed).shuffle(plan_ids)
    replay: list[dict[str, Any]] = []
    for plan_id in plan_ids:
        if len(replay) >= target:
            break
        replay.extend(by_plan[plan_id])
    if len(replay) < target:
        raise ValueError(f"Only {len(replay)} eligible Stage-1 replay rows; need {target}")
    mixed_train = [*s2_train, *replay]
    random.Random(args.seed + 1).shuffle(mixed_train)
    train_ids, val_ids = {row["plan_id"] for row in mixed_train}, {row["plan_id"] for row in s2_val}
    if train_ids & val_ids:
        raise ValueError("Mixed training and Codex validation plans overlap")
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", mixed_train)
    write_jsonl(output / "val.jsonl", s2_val)
    manifest = {
        "schema_version": "qwen-qa-stage2-replay/1",
        "stage1_dataset": str(stage1), "stage1_dataset_sha256": sha256(stage1 / "dataset_manifest.json"),
        "stage2_dataset": str(stage2), "stage2_dataset_sha256": sha256(stage2 / "dataset_manifest.json"),
        "replay_fraction_requested": args.replay_fraction,
        "records": {"train": len(mixed_train), "val": len(s2_val), "stage2_train": len(s2_train), "stage1_replay": len(replay)},
        "plans": {"train": len(train_ids), "val": len(val_ids), "stage1_replay": len({row["plan_id"] for row in replay})},
        "category_counts": {"train": dict(sorted(Counter(row["category"] for row in mixed_train).items())), "val": dict(sorted(Counter(row["category"] for row in s2_val).items()))},
        "validation_policy": "Codex-only, plan-disjoint from all training rows; Manual20 is not present.",
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
