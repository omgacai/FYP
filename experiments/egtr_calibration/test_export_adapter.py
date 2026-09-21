"""Regression test for the portable CubiCasa-to-EGTR adapter export."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "retrieval_app/scripts/export_egtr_adapter.py"


class AdapterExportTest(unittest.TestCase):
    def test_primary_access_collapses_door_and_open_edges(self):
        records = [
            {"plan_id": "p-train", "split": "train", "image_path": "colorful/1/F1_scaled.png", "image_size": [100, 80],
             "rooms": [{"room_id": "Kitchen_1", "category_id": 1, "category_name": "Kitchen", "bbox_xyxy": [0, 0, 30, 30]},
                       {"room_id": "LivingRoom_1", "category_id": 2, "category_name": "LivingRoom", "bbox_xyxy": [30, 0, 90, 70]}],
             "silver_edges": [{"room_a": "Kitchen_1", "room_b": "LivingRoom_1", "predicate_id": 3, "predicate": "open_connected"}]},
            {"plan_id": "p-val", "split": "val", "image_path": "high_quality/2/F1_scaled.png", "image_size": [90, 90],
             "rooms": [{"room_id": "Bath_1", "category_id": 3, "category_name": "Bath", "bbox_xyxy": [10, 10, 60, 60]}],
             "silver_edges": []},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            canonical = temporary_path / "canonical.jsonl"
            canonical.write_text("".join(json.dumps(record) + "\n" for record in records))
            output = temporary_path / "cubicasa_visual_genome"
            subprocess.run([sys.executable, str(EXPORTER), "--canonical", str(canonical), "--output-dir", str(output)], check=True)
            relations = json.loads((output / "rel.json").read_text())
            self.assertEqual(relations["rel_categories"], ["no_relation", "adjacent_to", "direct_access"])
            self.assertEqual(relations["train"]["0"], [[0, 1, 2], [1, 0, 2]])
            self.assertEqual(len(json.loads((output / "test.json").read_text())["images"]), 0)


if __name__ == "__main__":
    unittest.main()
