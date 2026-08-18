from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from floorplan_app.core.models import ParserResult


class FloorplanParser(ABC):
    """Implement this adapter to plug another model into the same dashboard."""

    display_name: str
    description: str

    @abstractmethod
    def parse(self, image_path: Path, *, max_side: int, postprocess_threshold: float) -> ParserResult:
        """Return the parser-neutral representation used by later stages."""
