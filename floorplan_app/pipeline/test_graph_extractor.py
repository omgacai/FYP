from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from floorplan_app.core.models import SVGResult
from floorplan_app.pipeline.graph_extractor import extract_graph


REPO_ROOT = Path(__file__).resolve().parents[2]
CUBIGRAPH_ROOT = REPO_ROOT / "third_party" / "CubiGraph5K"


def svg(*, wall: bool = False, outer_wall: bool = False, door: bool = False) -> str:
    wall_svg = (
        '<g class="Wall"><polygon points="98,0 102,0 102,100 98,100"/></g>' if wall else ""
    )
    outer_wall_svg = (
        '<g class="Wall"><polygon points="0,0 4,0 4,100 0,100"/></g>' if outer_wall else ""
    )
    door_svg = (
        '<g class="Threshold"><polygon points="98,40 102,40 102,60 98,60"/></g>' if door else ""
    )
    return f'''<svg width="200" height="100">
      <g class="Space LivingRoom"><polygon points="0,0 100,0 100,100 0,100"/></g>
      <g class="Space Kitchen"><polygon points="100,0 200,0 200,100 100,100"/></g>
      {wall_svg}{outer_wall_svg}{door_svg}
    </svg>'''


class OpenConnectionExtractionTests(unittest.TestCase):
    def extract(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "relations.svg"
            return extract_graph(
                SVGResult(svg_text=svg(**kwargs), path=Path("synthetic.svg"), diagnostics={}),
                CUBIGRAPH_ROOT,
                output,
            )

    def test_unblocked_shared_boundary_is_open_connected_when_walls_are_available(self):
        graph = self.extract(outer_wall=True)
        self.assertEqual(graph.adjacency["LivingRoom_1"]["Kitchen_1"], 3)
        self.assertEqual(graph.diagnostics["open_connected_edges"], 1)

    def test_missing_wall_geometry_keeps_boundary_as_unknown_adjacency(self):
        graph = self.extract()
        self.assertEqual(graph.adjacency["LivingRoom_1"]["Kitchen_1"], 1)
        self.assertFalse(graph.diagnostics["open_connection_inference_available"])

    def test_wall_blocks_open_connection(self):
        graph = self.extract(wall=True)
        self.assertEqual(graph.adjacency["LivingRoom_1"]["Kitchen_1"], 1)
        self.assertEqual(graph.diagnostics["open_connected_edges"], 0)

    def test_detected_door_has_priority_over_open_connection(self):
        graph = self.extract(door=True)
        self.assertEqual(graph.adjacency["LivingRoom_1"]["Kitchen_1"], 2)
        self.assertEqual(graph.diagnostics["open_connected_edges"], 0)


if __name__ == "__main__":
    unittest.main()
