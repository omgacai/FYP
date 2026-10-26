from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, FeatureNotFound, XMLParsedAsHTMLWarning
from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorplan_app.core.models import GraphResult, SVGResult


def _repair_geometry(geometry):
    """Return a usable geometry and whether the source polygon needed repair."""
    if geometry.is_valid:
        return geometry, False
    repaired = geometry.buffer(0)
    if repaired.is_empty:
        raise ValueError("Invalid source polygon became empty during repair")
    return repaired, True


OPEN_MIN_SHARED_BOUNDARY = 20.0
WALL_CLEARANCE = 0.5


def _points(value: str) -> list[tuple[float, float]]:
    """Parse the simple SVG polygon format used by CubiCasa model.svg files."""
    values = [float(number) for number in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", value)]
    if len(values) < 6 or len(values) % 2:
        raise ValueError("polygon needs at least three x,y points")
    return list(zip(values[::2], values[1::2]))


def source_wall_geometries(svg: Any) -> tuple[list[Any], int]:
    """Return valid top-level Wall polygons from a CubiCasa source SVG.

    Door and window polygons are nested inside Wall groups, so only a direct
    child polygon represents the wall body. This avoids treating a threshold
    itself as a blocker for an open-passage candidate.
    """
    walls: list[Any] = []
    skipped = 0
    for group in svg.find_all("g"):
        classes = set(group.get("class", []))
        if "Wall" not in classes:
            continue
        polygon = group.find("polygon", recursive=False)
        if polygon is None or not isinstance(polygon.get("points"), str):
            skipped += 1
            continue
        try:
            geometry, repaired = _repair_geometry(Polygon(_points(polygon["points"])))
            if geometry.is_empty or geometry.area <= 0:
                skipped += 1
                continue
            walls.append(geometry)
            skipped += int(repaired)
        except (TypeError, ValueError):
            skipped += 1
    return walls, skipped


def generate_door_open_relations(plan, wall_geometries: list[Any]) -> dict[str, int]:
    """Generate door, open-passage, then adjacency relations from source SVG.

    CubiGraph's original implementation tests buffered polygon adjacency before
    shared doors.  Since ordinary interior doors lie on a shared boundary, that
    makes a door relation almost impossible to observe.  This owned policy
    preserves the legacy geometry test but gives a shared detected door the
    intended ``via-door`` label. A relation is marked ``open_connected`` only
    when two room polygons share a substantial boundary segment that remains
    unblocked by an explicit source-SVG wall polygon. When the source contains
    no wall polygons, a shared boundary remains ordinary spatial adjacency:
    absence of wall geometry is unknown, never evidence of an opening.
    """
    plan.relation = []
    repairs = {"room_polygons_repaired": 0, "door_polygons_repaired": 0,
               "open_connected_edges": 0}
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
    wall_union = unary_union(wall_geometries) if wall_geometries else None
    for room in plan.rooms:
        for door_name, door_geometry in door_geometries.items():
            if room_geometries[room.name].intersection(door_geometry.buffer(1.0)).area > 10.0:
                room.adjacent_doors.add(door_name)
    for index, room1 in enumerate(plan.rooms):
        for room2 in plan.rooms[index + 1:]:
            shared_door = room1.adjacent_doors.intersection(room2.adjacent_doors)
            if shared_door:
                label = 2  # via-door
            else:
                shared_boundary = room_geometries[room1.name].boundary.intersection(
                    room_geometries[room2.name].boundary
                )
                unblocked_boundary = shared_boundary
                if wall_union is not None and not shared_boundary.is_empty:
                    unblocked_boundary = shared_boundary.difference(wall_union.buffer(WALL_CLEARANCE))
                if wall_union is not None and (
                    shared_boundary.length >= OPEN_MIN_SHARED_BOUNDARY
                    and unblocked_boundary.length >= OPEN_MIN_SHARED_BOUNDARY
                    and unblocked_boundary.length / shared_boundary.length >= 0.8
                ):
                    label = 3  # open passage, supported by an unblocked shared boundary
                    repairs["open_connected_edges"] += 1
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
    wall_geometries, walls_skipped = source_wall_geometries(soup)
    repairs = generate_door_open_relations(plan, wall_geometries)
    adjacency = plan.get_adjacency_list()
    output_path.write_text(str(plan.generate_relation_svg()), encoding='utf-8')
    relation_counts = {1: 0, 2: 0, 3: 0}
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
            'open_connected_edges': relation_counts[3],
            'source_wall_polygons': len(wall_geometries),
            'source_wall_polygons_skipped': walls_skipped,
            'open_connection_inference_available': bool(wall_geometries),
            'relation_policy': 'door-first plus unblocked-shared-boundary open candidates',
            'svg_parser_backend': parser_backend,
            **repairs,
            'adjacency_json': json.dumps(adjacency, indent=2),
        },
    )
