import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "retrieval_app/scripts/build_cubicasa_error_review.py"


class ErrorReviewTest(unittest.TestCase):
    def test_builds_visual_bundle_for_skipped_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "high_quality/1"
            sample.mkdir(parents=True)
            Image.new("RGB", (80, 40), "white").save(sample / "F1_scaled.png")
            (sample / "model.svg").write_text("<svg/>")
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"plan_id": "p1", "image_path": "high_quality/1/F1_scaled.png", "svg_path": "high_quality/1/model.svg", "metadata": {"source_dir": "high_quality/1"}}) + "\n")
            errors = root / "errors.jsonl"
            errors.write_text(json.dumps({"plan_id": "p1", "error": "Degenerate room box"}) + "\n")
            output = root / "review"
            subprocess.run([sys.executable, str(SCRIPT), "--errors", str(errors), "--manifest", str(manifest), "--corpus-root", str(root), "--output", str(output)], check=True)
            self.assertIn("Degenerate room box", (output / "index.html").read_text())
            self.assertTrue(any((output / "assets").glob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
