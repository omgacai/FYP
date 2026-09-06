from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from retrieval_app.core.models import ModalityScores, PlanRecord, QuerySpec, RetrievalResult
from retrieval_app.representations.geometry import geometry_score
from retrieval_app.representations.graph import graph_score


class VisualScorer(Protocol):
    def score(self, text: str, image_path): ...


@dataclass(frozen=True)
class FusionWeights:
    visual: float
    graph: float
    geometry: float

    def normalised(self) -> "FusionWeights":
        total = self.visual + self.graph + self.geometry
        if total <= 0:
            raise ValueError("At least one fusion weight must be positive.")
        return FusionWeights(self.visual / total, self.graph / total, self.geometry / total)


def _fuse(scores: ModalityScores, weights: FusionWeights) -> float:
    active = [(scores.visual, weights.visual), (scores.graph, weights.graph), (scores.geometry, weights.geometry)]
    active = [(score, weight) for score, weight in active if score is not None and weight > 0]
    if not active:
        return 0.0
    denominator = sum(weight for _, weight in active)
    return sum(score * weight for score, weight in active) / denominator


def retrieve(
    query: QuerySpec,
    plans: list[PlanRecord],
    weights: FusionWeights,
    visual_scorer: VisualScorer | None = None,
) -> list[RetrievalResult]:
    weights = weights.normalised()
    results: list[RetrievalResult] = []
    for plan in plans:
        visual = visual_scorer.score(query.text, plan.image_path) if visual_scorer else None
        graph, graph_evidence = graph_score(query, plan)
        geometry, geometry_evidence = geometry_score(query, plan)
        scores = ModalityScores(visual=visual, graph=graph, geometry=geometry)
        evidence = tuple(graph_evidence + geometry_evidence)
        results.append(RetrievalResult(plan=plan, scores=scores, fused_score=_fuse(scores, weights), evidence=evidence))
    return sorted(results, key=lambda item: item.fused_score, reverse=True)
