from __future__ import annotations

from pathlib import Path

import numpy as np
import svgwrite
from shapely.geometry import MultiPolygon, Polygon

from floorplan_app.core.models import ParserResult, SVGResult


CUBIGRAPH_NAMES = {
    'Living Room': 'LivingRoom', 'Bed Room': 'Bedroom', 'Bath': 'Bath',
    'Entry': 'Entry', 'Kitchen': 'Kitchen', 'Storage': 'Storage',
    'Garage': 'Garage', 'Outdoor': 'Outdoor', 'Undefined': 'Undefined',
}


def _add_geometry(drawing, group, geometry, label: str) -> None:
    if isinstance(geometry, Polygon):
        group.add(drawing.polygon(points=[(float(x), float(y)) for x, y in geometry.exterior.coords], class_=label))
    elif isinstance(geometry, MultiPolygon):
        for item in geometry.geoms:
            _add_geometry(drawing, group, item, label)


def _polygon_parts(geometry):
    """Yield one polygon per CubiGraph Space group.

    CubiGraph5K's legacy reader calls ``group.find('polygon')`` and would
    silently ignore second and later polygons in the same Space group.
    """
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def extract_svg(result: ParserResult, output_path: Path) -> SVGResult:
    """Create CubiGraph-compatible SVG. Only predicted doors become Thresholds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = result.processed_image.size
    drawing = svgwrite.Drawing(str(output_path), size=(width, height), profile='full')
    room_count = 0
    undefined_count = 0
    for geometry, metadata in zip(result.room_polygons, result.room_metadata):
        raw_name = result.room_class_names[metadata['class']]
        name = CUBIGRAPH_NAMES.get(raw_name, 'Undefined')
        for part in _polygon_parts(geometry):
            if part.is_empty or part.area <= 1e-6:
                continue
            group = drawing.add(drawing.g(class_=f'Space {name}'))
            _add_geometry(drawing, group, part, name)
            room_count += 1
            undefined_count += name == 'Undefined'

    door_count = 0
    window_count = 0
    for polygon, metadata in zip(result.icon_polygons, result.icon_metadata):
        if metadata.get('type') != 'icon':
            continue
        name = result.icon_class_names[metadata['class']]
        if name == 'Window':
            window_count += 1
        # CubiGraph has no window threshold type. Exporting it would lie about connectivity.
        if name != 'Door':
            continue
        group = drawing.add(drawing.g(class_='Threshold'))
        if isinstance(polygon, np.ndarray):
            group.add(drawing.polygon(points=[(float(x), float(y)) for x, y in polygon.reshape(-1, 2)], class_='Door'))
        else:
            _add_geometry(drawing, group, polygon, 'Door')
        door_count += 1
    drawing.save()
    return SVGResult(
        svg_text=output_path.read_text(encoding='utf-8'),
        path=output_path,
        diagnostics={
            'rooms_exported': room_count, 'undefined_rooms': undefined_count,
            'doors_exported': door_count, 'windows_ignored_for_connectivity': window_count,
        },
    )
