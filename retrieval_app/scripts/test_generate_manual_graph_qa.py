from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from retrieval_app.scripts.generate_manual_graph_qa import generate, generate_candidates, load_plan


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
        self.assertEqual(
            {row["category"] for row in rows},
            {"count", "direct_access", "relative_position", "compound_access", "shortest_access_path"},
        )
        self.assertNotIn("no", {row["gold_answer"] for row in rows})
        path_row = next(row for row in rows if row["difficulty"] == "D4")
        self.assertEqual(path_row["gold_answer"], "2")
        self.assertEqual(json.loads(path_row["supporting_path_ids"]), ["entry", "corridor", "bedroom"])

    def test_unknown_is_explicit_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path = root / "demo.graph.json"
            annotation_path.write_text(json.dumps(annotation()), encoding="utf-8")
            buckets = generate_candidates(load_plan(annotation_path), include_unknown=True)
        self.assertTrue(any(
            row["gold_answer"] == "unknown"
            for rows in buckets.values()
            for row in rows
        ))

    def test_seeded_selection_is_reproducible_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "demo.graph.json").write_text(json.dumps(annotation()), encoding="utf-8")
            first, second = root / "first.csv", root / "second.csv"
            generate(root, first, per_plan=5, include_unknown=False, selection_seed=42)
            generate(root, second, per_plan=5, include_unknown=False, selection_seed=42)
            self.assertEqual(first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8"))
            with first.open(encoding="utf-8", newline="") as handle:
                self.assertEqual({row["selection_seed"] for row in csv.DictReader(handle)}, {"42"})

    def test_adjacency_is_not_substituted_for_direct_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = annotation()
            source["edges"] = [{"a": "corridor", "b": "kitchen", "relation": "adjacent_to"}]
            (root / "demo.graph.json").write_text(json.dumps(source), encoding="utf-8")
            output = root / "questions.csv"
            count, warnings = generate(root, output, per_plan=5, include_unknown=False)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(count, 2)
        self.assertEqual({row["category"] for row in rows}, {"count", "relative_position"})
        self.assertEqual(warnings, ["demo_plan: generated 2/5 candidates"])

    def test_direct_path_is_a_valid_d4_fallback_when_connectivity_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = annotation()
            source["connectivity_check"] = {"status": "fail"}
            source["edges"] = [{"a": "entry", "b": "bedroom", "relation": "connected_by_door"}]
            annotation_path = root / "demo.graph.json"
            annotation_path.write_text(json.dumps(source), encoding="utf-8")
            buckets = generate_candidates(load_plan(annotation_path), include_unknown=False)
        self.assertEqual(len(buckets["D4"]), 1)
        self.assertEqual(buckets["D4"][0]["gold_answer"], "1")

    def test_rejects_a_non_five_question_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "demo.graph.json").write_text(json.dumps(annotation()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be 5"):
                generate(root, root / "questions.csv", per_plan=4, include_unknown=False)


if __name__ == "__main__":
    unittest.main()
