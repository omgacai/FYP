from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from retrieval_app.scripts.build_qwen_qa_sft_dataset import main, prepare_plan


def plan(*, plan_id: str = "plan-a", split: str = "train", edges: list[dict] | None = None) -> dict:
    return {
        "plan_id": plan_id, "split": split, "source_identity": f"colorful/{plan_id}",
        "image_path": f"colorful/{plan_id}/F1_scaled.png", "svg_path": f"colorful/{plan_id}/model.svg", "image_size": [200, 100],
        "rooms": [
            {"room_id": "entry", "category_name": "Entry", "bbox_xyxy": [0, 20, 40, 90]},
            {"room_id": "corridor", "category_name": "Corridor", "bbox_xyxy": [50, 20, 100, 90]},
            {"room_id": "bedroom", "category_name": "Bedroom", "bbox_xyxy": [110, 0, 190, 50]},
        ],
        "silver_edges": edges if edges is not None else [
            {"room_a": "entry", "room_b": "corridor", "predicate": "connected_by_door"},
            {"room_a": "corridor", "room_b": "bedroom", "predicate": "connected_by_door"},
        ],
    }


class BuildQwenQaSftDatasetTest(unittest.TestCase):
    def test_no_edge_plan_is_excluded(self):
        rooms, edges, reason = prepare_plan(plan(edges=[]))
        self.assertEqual((rooms, edges, reason), ([], [], "no_cubigraph_edges"))

    def test_builder_keeps_splits_disjoint_and_emits_five_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "canonical.jsonl"
            corpus.write_text("\n".join(json.dumps(item) for item in [plan(), plan(plan_id="plan-v", split="val")]) + "\n", encoding="utf-8")
            exclusions = root / "excluded.json"
            exclusions.write_text("[]", encoding="utf-8")
            output = root / "qa"
            old_argv = sys.argv
            try:
                sys.argv = ["builder", "--canonical-corpus", str(corpus), "--exclude-identities", str(exclusions), "--output-dir", str(output)]
                main()
            finally:
                sys.argv = old_argv
            train = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            val = [json.loads(line) for line in (output / "val.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(train), 5)
        self.assertEqual(len(val), 5)
        self.assertEqual({row["plan_id"] for row in train}, {"plan-a"})
        self.assertEqual({row["category"] for row in train}, {"count", "direct_access", "relative_position", "compound_access", "shortest_access_path"})
        self.assertTrue(all(len(row["messages"]) == 3 for row in train))


if __name__ == "__main__":
    unittest.main()
