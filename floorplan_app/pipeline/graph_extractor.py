from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, FeatureNotFound, XMLParsedAsHTMLWarning

from floorplan_app.core.models import GraphResult, SVGResult


def _repair_geometry(geometry):
    """Return a usable geometry and whether the source polygon needed repair."""
    if geometry.is_valid:
        return geometry, False
    repaired = geometry.buffer(0)
    if repaired.is_empty:
        raise ValueError("Invalid source polygon became empty during repair")
    return repaired, True


def generate_door_first_relations(plan) -> dict[str, int]:
    """Generate mutually exclusive graph labels with valid-door evidence first.

    CubiGraph's original implementation tests buffered polygon adjacency before
    shared doors.  Since ordinary interior doors lie on a shared boundary, that
    makes a door relation almost impossible to observe.  This owned policy
    preserves the legacy geometry test but gives a shared detected door the
    intended ``via-door`` label.
    """
    plan.relation = []
    repairs = {"room_polygons_repaired": 0, "door_polygons_repaired": 0}
    room_geometries = {}
    door_geometries = {}
    for room in plan.rooms:
        geometry, repaired = _repair_geometry(room.to_shapely_polygon())
        room_geometries[room.name] = geometry
        repairs["room_polygons_repaired"] += int(repaired)
        room.adjacent_doors = set()
    for door in plan.doors:
        geometry, repaired = _repair_geometry(door.to_shapely_polygon())
        door_geometries[door.name] = geometry
        repairs["door_polygons_repaired"] += int(repaired)
    for room in plan.rooms:
        for door_name, door_geometry in door_geometries.items():
            if room_geometries[room.name].intersection(door_geometry.buffer(1.0)).area > 10.0:
                room.adjacent_doors.add(door_name)
    for index, room1 in enumerate(plan.rooms):
        for room2 in plan.rooms[index + 1:]:
            shared_door = room1.adjacent_doors.intersection(room2.adjacent_doors)
            if shared_door:
                label = 2  # via-door
            elif room_geometries[room1.name].buffer(1.0).intersection(
                room_geometries[room2.name].buffer(1.0)
            ).area > 5.0:
                label = 1  # adjacent only
            else:
                label = 0
            plan.relation.append((room1.name, label, room2.name))
    return repairs


def extract_graph(svg_result: SVGResult, cubigraph_repo: Path, output_path: Path) -> GraphResult:
    src = cubigraph_repo / 'src'
    if not src.exists():
        raise FileNotFoundError('CubiGraph5K is missing. Run the notebook setup cell first.')
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from plan import Plan

    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
    # The cluster may intentionally keep the login-home environment tiny.  lxml
    # is preferable for SVG/XML, but CubiGraph only needs basic tag traversal;
    # fall back to Python's built-in parser so a small inspection job does not
    # fail solely because the optional C-extension cannot be installed.
    try:
        soup = BeautifulSoup(svg_result.svg_text, 'lxml')
        parser_backend = 'lxml'
    except FeatureNotFound:
        warnings.warn(
            'lxml is unavailable; using Python html.parser for CubiGraph SVG parsing. '
            'Install lxml in a quota-safe environment before large-scale runs.',
            RuntimeWarning,
        )
        soup = BeautifulSoup(svg_result.svg_text, 'html.parser')
        parser_backend = 'html.parser fallback'
    plan = Plan(soup.find('svg'))
    repairs = generate_door_first_relations(plan)
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
            'svg_parser_backend': parser_backend,
            **repairs,
            'adjacency_json': json.dumps(adjacency, indent=2),
        },
    )
