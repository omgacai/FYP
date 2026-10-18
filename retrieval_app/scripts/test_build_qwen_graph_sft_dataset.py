from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from retrieval_app.scripts.build_qwen_graph_sft_dataset import canonical_graph, main


def source_record(*, plan_id: str = "plan-a", split: str = "train", identity: str = "colorful/1") -> dict:
    return {
        "plan_id": plan_id, "source_identity": identity, "split": split,
        "image_path": "colorful/1/F1_scaled.png", "svg_path": "colorful/1/model.svg", "image_size": [200, 100],
        "rooms": [
            {"room_id": "bed", "category_name": "Bedroom", "bbox_xyxy": [100, 10, 180, 80]},
            {"room_id": "entry", "category_name": "Entry", "bbox_xyxy": [10, 20, 80, 90]},
        ],
        "silver_edges": [{"room_a": "bed", "room_b": "entry", "predicate": "connected_by_door"}],
        "provenance": {"graph_path": "derived/plan-a.json"},
    }


class BuildQwenGraphSftDatasetTest(unittest.TestCase):
    def test_canonical_graph_is_stably_relabelled_and_normalised(self):
        graph = canonical_graph(source_record())
        self.assertEqual([room["id"] for room in graph["rooms"]], ["R1", "R2"])
        self.assertEqual(graph["rooms"][0]["type"], "bedroom")
        self.assertEqual(graph["rooms"][0]["bbox"], [500.0, 100.0, 900.0, 800.0])
        self.assertEqual(graph["edges"], [{"source": "R1", "target": "R2", "type": "connected_by_door", "confidence": 1.0}])

    def test_builder_excludes_manual_identity_and_writes_split_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "canonical.jsonl"
            val = source_record(plan_id="plan-b", split="val", identity="high_quality/2")
            train = source_record(plan_id="plan-c", split="train", identity="colorful/3")
            corpus.write_text("\n".join(json.dumps(item) for item in [source_record(identity="colorful/1"), train, val]) + "\n", encoding="utf-8")
            exclusions = root / "exclude.json"
            exclusions.write_text(json.dumps([
                "/Users/example/Downloads/cubicasa5k/cubicasa5k/colorful/1/"
            ]), encoding="utf-8")
            output = root / "sft"
            import sys
            previous = sys.argv
            try:
                sys.argv = ["builder", "--canonical-corpus", str(corpus), "--exclude-identities", str(exclusions), "--output-dir", str(output)]
                main()
            finally:
                sys.argv = previous
            self.assertEqual(len((output / "train.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len((output / "val.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len((output / "excluded.jsonl").read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
