import unittest

from retrieval_app.scripts.evaluate_qa import bootstrap_delta, summary
from retrieval_app.scripts.run_qwen_qa import parse, parse_response, validate_evidence


class QaEvaluationTests(unittest.TestCase):
    def test_strict_answer_parser(self):
        self.assertEqual(parse('{"answer":" YES "}', "yes_no_unknown"), ("yes", None))
        self.assertEqual(parse('{"answer":2}', "integer"), ("2", None))
        self.assertIsNotNone(parse('answer: yes', "yes_no_unknown")[1])

    def test_evidence_reasoning_parser_and_graph_validation(self):
        raw = ('{"answer":"yes","evidence":{"node_ids":["a","b"],'
               '"edge_ids":["a--b:direct_access"]},"reasoning":"The edge joins the rooms."}')
        answer, evidence, reasoning, error = parse_response(raw, "yes_no_unknown", "evidence_reasoning_v1")
        self.assertEqual((answer, reasoning, error), ("yes", "The edge joins the rooms.", None))
        self.assertEqual(validate_evidence(evidence, {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"a": "a", "b": "b", "relation": "direct_access"}]}), [])

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
