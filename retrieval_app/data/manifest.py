from __future__ import annotations

import json
from pathlib import Path

from retrieval_app.core.models import PlanRecord


def _resolve(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def load_manifest(manifest_path: Path, corpus_root: Path | None = None) -> list[PlanRecord]:
    """Load a JSONL corpus manifest without assuming a specific dataset layout."""
    manifest_path = manifest_path.expanduser().resolve()
    base_dir = corpus_root.expanduser().resolve() if corpus_root else manifest_path.parent
    records: list[PlanRecord] = []
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        raw = json.loads(raw_line)
        image_path = _resolve(raw.get("image_path"), base_dir)
        if image_path is None:
            raise ValueError(f"Manifest line {line_number} is missing image_path.")
        records.append(
            PlanRecord(
                plan_id=str(raw["plan_id"]),
                image_path=image_path,
                svg_path=_resolve(raw.get("svg_path"), base_dir),
                graph_path=_resolve(raw.get("graph_path"), base_dir),
                description=str(raw.get("description", "")),
                room_counts={str(key): int(value) for key, value in raw.get("room_counts", {}).items()},
                geometry={str(key): float(value) for key, value in raw.get("geometry", {}).items()},
                graph_provenance=str(raw.get("graph_provenance", "unavailable")),
                metadata=dict(raw.get("metadata", {})),
            )
        )
    if not records:
        raise ValueError("The manifest has no records.")
    return records


def load_adjacency(graph_path: Path | None) -> dict[str, dict[str, int]]:
    """Read the adjacency JSON emitted by floorplan_app's CubiGraph stage."""
    if graph_path is None or not graph_path.exists():
        return {}
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected adjacency object in {graph_path}")
    return {
        str(node): {str(neighbour): int(relation) for neighbour, relation in neighbours.items()}
        for node, neighbours in raw.items()
        if isinstance(neighbours, dict)
    }
