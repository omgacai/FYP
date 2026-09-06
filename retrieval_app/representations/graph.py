from __future__ import annotations

from collections import Counter

from retrieval_app.core.models import PlanRecord, QuerySpec
from retrieval_app.data.manifest import load_adjacency
from retrieval_app.representations.normalise import room_type


RELATION_CODES = {"adjacent_to": 1, "connected_by_door": 2}


def graph_summary(plan: PlanRecord) -> dict[str, int]:
    adjacency = load_adjacency(plan.graph_path)
    degrees = [len(neighbours) for neighbours in adjacency.values()]
    edge_count = sum(degrees) // 2
    return {
        "nodes": len(adjacency),
        "edges": edge_count,
        "door_edges": sum(
            1 for source, neighbours in adjacency.items() for target, relation in neighbours.items()
            if source < target and relation == 2
        ),
        "max_degree": max(degrees, default=0),
    }


def graph_score(query: QuerySpec, plan: PlanRecord) -> tuple[float | None, list[str]]:
    """Match explicit query relations against a typed CubiGraph adjacency map."""
    if not query.relations:
        return None, ["No graph relation was extracted from this query."]
    adjacency = load_adjacency(plan.graph_path)
    if not adjacency:
        return None, ["Plan has no readable graph JSON."]
    node_types = {node: room_type(node) for node in adjacency}
    satisfied = 0
    evidence: list[str] = []
    for requirement in query.relations:
        code = RELATION_CODES[requirement.relation]
        matched = any(
            node_types.get(source) == requirement.source_type
            and node_types.get(target) == requirement.target_type
            and relation == code
            for source, neighbours in adjacency.items()
            for target, relation in neighbours.items()
        )
        if matched:
            satisfied += 1
        state = "matched" if matched else "not found"
        evidence.append(f"{requirement.source_type} {requirement.relation} {requirement.target_type}: {state}")
    return satisfied / len(query.relations), evidence
