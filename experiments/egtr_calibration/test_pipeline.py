import unittest
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
