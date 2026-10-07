import json
import tempfile
import unittest
from pathlib import Path

from retrieval_app.scripts.evaluate_qa import bootstrap_delta, summary
from retrieval_app.scripts.run_qwen_qa import graph, make_messages, parse, parse_response, validate_evidence


class QaEvaluationTests(unittest.TestCase):
    def test_strict_answer_parser(self):
        self.assertEqual(parse('{"answer":" YES ","evidence":["A visible door joins the rooms."],"reasoning":"The door is visible."}', "yes_no_unknown"), ("yes", None))
        self.assertEqual(parse('{"answer":2,"evidence":["Two bathrooms are visible."],"reasoning":"Two bathroom labels are visible."}', "integer"), ("2", None))
        self.assertIsNotNone(parse('answer: yes', "yes_no_unknown")[1])

    def test_evidence_reasoning_parser_and_graph_validation(self):
        raw = ('{"answer":"yes","evidence":{"node_ids":["a","b"],'
               '"edge_ids":["a--b:direct_access"]},"reasoning":"The edge joins the rooms."}')
        answer, evidence, reasoning, error = parse_response(raw, "yes_no_unknown", "graph_only")
        self.assertEqual((answer, reasoning, error), ("yes", "The edge joins the rooms.", None))
        self.assertEqual(validate_evidence(evidence, {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"a": "a", "b": "b", "relation": "direct_access"}]}), [])

    def test_compact_evidence_aliases_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.graph.json"
            path.write_text(json.dumps({"nodes": [{"id": "a", "label": "Entry 1", "type": "Entry", "x": 0, "y": 0}, {"id": "b", "label": "Bedroom 1", "type": "Bedroom", "x": 1, "y": 0}], "edges": [{"a": "a", "b": "b", "relation": "connected_by_door"}]}), encoding="utf-8")
            compact = graph(path, compact_evidence_ids=True)
        self.assertEqual([node["id"] for node in compact["nodes"]], ["R1", "R2"])
        self.assertEqual(compact["edges"][0]["edge_id"], "E1")
        self.assertEqual(validate_evidence({"node_ids": ["R1", "R2"], "edge_ids": ["E1"]}, compact), [])

    def test_condition_specific_prompts_and_shortest_path_note(self):
        row = {"question": "How many bathrooms are there?", "answer_format": "integer", "category": "count"}
        image_messages = make_messages(row, "image_only", Path("plan.png"), None)
        self.assertNotIn("graph", image_messages[0]["content"].lower())
        self.assertIn("visual observations", image_messages[0]["content"])
        graph_data = {"plan_id": "p", "nodes": [], "edges": []}
        graph_messages = make_messages(row, "graph_only", Path("plan.png"), graph_data)
        self.assertIn("R-number aliases", graph_messages[0]["content"])
        path_row = {"question": "How far?", "answer_format": "integer", "category": "shortest_access_path"}
        path_messages = make_messages(path_row, "graph_only", Path("plan.png"), graph_data)
        self.assertIn("door or a direct open connection", path_messages[1]["content"][-1]["text"])
        self.assertIn("path length 2 from A to C", path_messages[1]["content"][-1]["text"])

    def test_plan_macro_and_unknown_metrics(self):
        rows = [
            {"question_id": "a", "plan_id": "p1", "difficulty": "D1", "category": "access", "answer_format": "yes_no_unknown", "answerability": "answerable", "gold": "yes", "pred": "yes", "valid": True, "correct": True},
            {"question_id": "b", "plan_id": "p1", "difficulty": "D1", "category": "access", "answer_format": "yes_no_unknown", "answerability": "unknown", "gold": "unknown", "pred": "yes", "valid": True, "correct": False},
            {"question_id": "c", "plan_id": "p2", "difficulty": "D0", "category": "count", "answer_format": "integer", "answerability": "answerable", "gold": "2", "pred": "2", "valid": True, "correct": True},
        ]
        result = summary(rows)
        self.assertAlmostEqual(result["micro_exact_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["plan_macro_accuracy"], 0.75)
        self.assertEqual(result["unknown"]["unsupported_answer_rate"], 1.0)

    def test_paired_bootstrap_is_in_percentage_points(self):
        base = [{"question_id": "a", "plan_id": "p1", "correct": False}, {"question_id": "b", "plan_id": "p2", "correct": False}]
        candidate = [{"question_id": "a", "plan_id": "p1", "correct": True}, {"question_id": "b", "plan_id": "p2", "correct": True}]
        result = bootstrap_delta(base, candidate, 100, 1)
        self.assertEqual(result["point_estimate_pp"], 100.0)


if __name__ == "__main__":
    unittest.main()
