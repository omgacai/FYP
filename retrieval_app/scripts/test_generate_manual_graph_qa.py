from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from retrieval_app.scripts.generate_manual_graph_qa import generate


def annotation(*, all_pairs_reviewed: bool = False) -> dict:
    return {
        "schema_version": "floorplan-manual-graph/2",
        "plan_id": "demo_plan",
        "status": "manually_reviewed",
        "all_pairs_reviewed": all_pairs_reviewed,
        "connectivity_check": {"status": "pass"},
        "nodes": [
            {"id": "entry", "type": "Entry", "label": "Entry 1", "x": 0.1, "y": 0.5, "polygon": [[0, 0], [0.1, 0], [0.1, 1]]},
            {"id": "corridor", "type": "Corridor", "label": "Corridor 1", "x": 0.35, "y": 0.5, "polygon": [[0.2, 0], [0.4, 0], [0.4, 1]]},
            {"id": "bedroom", "type": "Bedroom", "label": "Bedroom 1", "x": 0.7, "y": 0.3, "polygon": [[0.5, 0], [1, 0], [1, 0.6]]},
            {"id": "kitchen", "type": "Kitchen", "label": "Kitchen 1", "x": 0.7, "y": 0.8, "polygon": [[0.5, 0.6], [1, 0.6], [1, 1]]},
        ],
        "edges": [
            {"a": "entry", "b": "corridor", "relation": "connected_by_door"},
            {"a": "corridor", "b": "bedroom", "relation": "connected_by_door"},
            {"a": "corridor", "b": "kitchen", "relation": "adjacent_to"},
        ],
    }


class GenerateManualGraphQATest(unittest.TestCase):
    def test_generates_diverse_rows_and_never_invents_negative_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path = root / "demo.graph.json"
            annotation_path.write_text(json.dumps(annotation()), encoding="utf-8")
            output = root / "questions.csv"
            count, warnings = generate(root, output, per_plan=5, include_unknown=False)
            self.assertEqual(count, 5)
            self.assertEqual(warnings, [])
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual({row["difficulty"] for row in rows}, {"D0", "D1", "D2", "D3", "D4"})
        self.assertNotIn("no", {row["gold_answer"] for row in rows})
        path_row = next(row for row in rows if row["difficulty"] == "D4")
        self.assertEqual(path_row["gold_answer"], "2")
        self.assertEqual(json.loads(path_row["supporting_path_ids"]), ["entry", "corridor", "bedroom"])

    def test_unknown_is_explicit_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "demo.graph.json").write_text(json.dumps(annotation()), encoding="utf-8")
            output = root / "questions.csv"
            generate(root, output, per_plan=20, include_unknown=True)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertTrue(any(row["gold_answer"] == "unknown" for row in rows))


if __name__ == "__main__":
    unittest.main()
