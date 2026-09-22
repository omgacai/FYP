import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'experiments/egtr_calibration/audit_egtr_adapter.py'


class AdapterAuditTest(unittest.TestCase):
    def test_validates_small_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, images = root / 'data', root / 'source/high_quality/1'
            images.mkdir(parents=True)
            (images / 'F1_scaled.png').write_bytes(b'fixture')
            data.mkdir(); (data / 'images').symlink_to(root / 'source', target_is_directory=True)
            categories = [{'id': 1, 'name': 'Kitchen'}]
            for split, image_rows, annotations in [('train', [{'id': 0, 'file_name': 'high_quality/1/F1_scaled.png', 'width': 20, 'height': 10}], [{'id': 0, 'image_id': 0, 'category_id': 1, 'bbox': [0, 0, 10, 10]}]), ('val', [], []), ('test', [], [])]:
                (data / f'{split}.json').write_text(json.dumps({'images': image_rows, 'annotations': annotations, 'categories': categories}))
            (data / 'rel.json').write_text(json.dumps({'rel_categories': ['no_relation', 'adjacent_to', 'direct_access'], 'train': {'0': []}, 'val': {}, 'test': {}}))
            canonical = root / 'canonical.jsonl'; canonical.write_text(json.dumps({'source_identity': 'high_quality/1'}) + '\n')
            exclusions = root / 'exclude.json'; exclusions.write_text('[]')
            subprocess.run([sys.executable, str(SCRIPT), '--data', str(data), '--canonical', str(canonical), '--exclusions', str(exclusions)], check=True)


if __name__ == '__main__':
    unittest.main()
