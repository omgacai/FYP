import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from experiments.egtr_calibration.adapt import adapt
from experiments.egtr_calibration.prepare import identity


class AdapterTests(unittest.TestCase):
    def fixture(self):
        raw = {'status': 'ok', 'plan_id': 'p', 'objects': [
            {'query': i, 'label': 'room', 'score': 1., 'bbox_cxcywh': [0.25+i*0.5, 0.5, 0.4, 0.8]}
            for i in range(2)], 'provenance': {'labels': {'relations': ['door']}}}
        tensors = {'pred_rel': torch.ones(1, 2, 2, 1), 'pred_connectivity': torch.ones(1, 2, 2, 1)}
        mapping = {'objects': {'room': {'target': 'Bedroom', 'justification': 'Synthetic test only'}},
                   'relations': {'door': {'target': 'connected_by_door', 'justification': 'Synthetic test only'}}}
        return raw, tensors, mapping

    def test_cubicasa_coordinates_do_not_scale_from_svg_viewport(self):
        from retrieval_app.scripts.build_cubicasa_egtr_corpus import source_rooms
        room = SimpleNamespace(name='Bedroom_1', type='Bedroom', points=[(100, 100), (300, 200)])
        fake_plan = SimpleNamespace(Plan=lambda svg: SimpleNamespace(rooms=[room]))
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules', {'plan': fake_plan}):
            svg = Path(folder) / 'model.svg'
            svg.write_text('<svg width="500" height="400" viewBox="0 0 500 400"></svg>')
            result = source_rooms(svg, (1000, 800), Path(folder), svg_coordinate_size=(1000, 800))
            self.assertEqual(result[0]['bbox_xyxy'], [100, 100, 300, 200])
            smaller = source_rooms(svg, (500, 400), Path(folder), svg_coordinate_size=(1000, 800))
            self.assertEqual(smaller[0]['bbox_xyxy'], [50, 50, 150, 100])

    def test_symmetric_predictions_are_one_edge_without_self_loops(self):
        raw, tensors, mapping = self.fixture()
        result = adapt(raw, tensors, mapping, .3, .01)
        self.assertEqual(len(result['graph']['edges']), 1)
        self.assertEqual(result['graph']['edges'][0]['a'], 'q0')
        self.assertEqual(result['graph']['edges'][0]['b'], 'q1')

    def test_unsupported_ontology_is_not_an_empty_success(self):
        raw, tensors, _ = self.fixture()
        result = adapt(raw, tensors, {'objects': {}, 'relations': {}}, .3, .01)
        self.assertFalse(result['valid'])
        self.assertEqual(len(result['coverage']['unsupported_objects']), 2)

    def test_exclusions_survive_cluster_path_changes(self):
        self.assertEqual(identity('/Users/someone/data/colorful/803/'), identity('/cluster/data/colorful/803'))

    def test_mapping_needs_evidence(self):
        raw, tensors, mapping = self.fixture()
        mapping['relations']['door']['justification'] = ''
        with self.assertRaises(ValueError):
            adapt(raw, tensors, mapping, .3, .01)


if __name__ == '__main__':
    unittest.main()
