from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlanRecord:
    """One retrievable plan and the provenance of every representation."""

    plan_id: str
    image_path: Path
    svg_path: Path | None = None
    graph_path: Path | None = None
    description: str = ""
    room_counts: dict[str, int] = field(default_factory=dict)
    geometry: dict[str, float] = field(default_factory=dict)
    graph_provenance: str = "unavailable"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationConstraint:
    source_type: str
    relation: str
    target_type: str


@dataclass(frozen=True)
class QuerySpec:
    text: str
    room_counts: dict[str, int] = field(default_factory=dict)
    relations: tuple[RelationConstraint, ...] = ()
    requires_verified_connectivity: bool = False


@dataclass(frozen=True)
class ModalityScores:
    visual: float | None
    graph: float | None
    geometry: float | None
    text: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    plan: PlanRecord
    scores: ModalityScores
    fused_score: float
    evidence: tuple[str, ...]
