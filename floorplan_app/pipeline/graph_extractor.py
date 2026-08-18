from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from floorplan_app.core.models import GraphResult, SVGResult


def generate_door_first_relations(plan) -> None:
    """Generate mutually exclusive graph labels with valid-door evidence first.

    CubiGraph's original implementation tests buffered polygon adjacency before
    shared doors.  Since ordinary interior doors lie on a shared boundary, that
    makes a door relation almost impossible to observe.  This owned policy
    preserves the legacy geometry test but gives a shared detected door the
    intended ``via-door`` label.
    """
    plan.relation = []
    for room in plan.rooms:
        room.get_adjacent_doors(plan.doors)
    for index, room1 in enumerate(plan.rooms):
        for room2 in plan.rooms[index + 1:]:
            shared_door = room1.adjacent_doors.intersection(room2.adjacent_doors)
            if shared_door:
                label = 2  # via-door
            elif room1.to_shapely_polygon().buffer(1.0).intersection(
                room2.to_shapely_polygon().buffer(1.0)
            ).area > 5.0:
                label = 1  # adjacent only
            else:
                label = 0
            plan.relation.append((room1.name, label, room2.name))


def extract_graph(svg_result: SVGResult, cubigraph_repo: Path, output_path: Path) -> GraphResult:
    src = cubigraph_repo / 'src'
    if not src.exists():
        raise FileNotFoundError('CubiGraph5K is missing. Run the notebook setup cell first.')
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from plan import Plan

    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(svg_result.svg_text, 'lxml')
    plan = Plan(soup.find('svg'))
    generate_door_first_relations(plan)
    adjacency = plan.get_adjacency_list()
    output_path.write_text(str(plan.generate_relation_svg()), encoding='utf-8')
    relation_counts = {1: 0, 2: 0}
    for neighbours in adjacency.values():
        for relation in neighbours.values():
            if relation in relation_counts:
                relation_counts[relation] += 1
    # adjacency is symmetric, so count undirected edges once.
    relation_counts = {key: value // 2 for key, value in relation_counts.items()}
    return GraphResult(
        svg_text=output_path.read_text(encoding='utf-8'), adjacency=adjacency, path=output_path,
        diagnostics={
            'nodes': len(adjacency), 'adjacent_edges': relation_counts[1],
            'door_connected_edges': relation_counts[2],
            'relation_policy': 'door-first experimental',
            'adjacency_json': json.dumps(adjacency, indent=2),
        },
    )
