from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class ParserResult:
    """Parser-neutral contract consumed by SVG and graph stages."""

    parser_name: str
    source_image: Image.Image
    processed_image: Image.Image
    room_segmentation: np.ndarray
    icon_segmentation: np.ndarray
    room_polygons: list[Any]
    room_metadata: list[dict[str, Any]]
    icon_polygons: list[Any]
    icon_metadata: list[dict[str, Any]]
    room_class_names: list[str]
    icon_class_names: list[str]
    confidence_maps: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SVGResult:
    svg_text: str
    path: Path
    diagnostics: dict[str, Any]


@dataclass
class GraphResult:
    svg_text: str
    adjacency: dict[str, dict[str, int]]
    path: Path
    diagnostics: dict[str, Any]
