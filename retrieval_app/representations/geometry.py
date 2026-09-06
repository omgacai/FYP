from __future__ import annotations

from retrieval_app.core.models import PlanRecord, QuerySpec
from retrieval_app.representations.normalise import room_type


def geometry_score(query: QuerySpec, plan: PlanRecord) -> tuple[float | None, list[str]]:
    """Transparent count-based geometry score; later replace with a learned geometry encoder."""
    if not query.room_counts:
        return None, ["No count/geometry constraint was extracted from this query."]
    counts = {room_type(key): value for key, value in plan.room_counts.items()}
    if not counts:
        return None, ["Plan has no room-count metadata."]
    satisfied = 0
    evidence: list[str] = []
    for room, required in query.room_counts.items():
        observed = counts.get(room, 0)
        if observed >= required:
            satisfied += 1
            evidence.append(f"{room}: {observed} (requires at least {required})")
        else:
            evidence.append(f"{room}: {observed} (requires at least {required}; unmet)")
    return satisfied / len(query.room_counts), evidence
