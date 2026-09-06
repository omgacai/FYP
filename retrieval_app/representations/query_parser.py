from __future__ import annotations

import re

from retrieval_app.core.models import QuerySpec, RelationConstraint
from retrieval_app.representations.normalise import ALIASES, room_type


ROOM_PATTERN = "|".join(sorted((re.escape(name) for name in ALIASES), key=len, reverse=True))
COUNT_PATTERN = re.compile(rf"\b(\d+)\s+({ROOM_PATTERN})s?\b", re.IGNORECASE)
RELATION_PATTERN = re.compile(
    rf"\b({ROOM_PATTERN})\b.*?\b(adjacent to|next to|beside|connected to|access(?:ed)? from|opens? into)\b.*?\b({ROOM_PATTERN})\b",
    re.IGNORECASE,
)


def parse_query(text: str) -> QuerySpec:
    """Conservative, inspectable baseline parser; users can edit its JSON in the UI."""
    counts: dict[str, int] = {}
    for number, label in COUNT_PATTERN.findall(text):
        canonical = room_type(label)
        counts[canonical] = max(counts.get(canonical, 0), int(number))

    relations: list[RelationConstraint] = []
    connectivity = False
    for source, phrase, target in RELATION_PATTERN.findall(text):
        phrase = phrase.lower()
        relation = "connected_by_door" if any(token in phrase for token in ("connected", "access", "open")) else "adjacent_to"
        connectivity = connectivity or relation == "connected_by_door"
        relations.append(RelationConstraint(room_type(source), relation, room_type(target)))
    return QuerySpec(text=text, room_counts=counts, relations=tuple(relations), requires_verified_connectivity=connectivity)
