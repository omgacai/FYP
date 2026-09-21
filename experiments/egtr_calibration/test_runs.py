import argparse
import json
import tempfile
import unittest
from pathlib import Path
from experiments.egtr_calibration.runs import digest, initialize, verify_inputs, execute, refresh_summary
from experiments.egtr_calibration.qa import graph_context


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base/'source'
        (self.source/'images').mkdir(parents=True)
        (self.source/'annotations').mkdir()
        image=self.source/'images/p.png'
        image.write_bytes(b'fixture-image-hash')
        self.reference={'plan_id':'p','status':'manually_reviewed','all_pairs_reviewed':True,
                        'nodes':[{'id':'a','type':'Bedroom','bbox_xyxy':[0,0,.4,.4]},
                                 {'id':'b','type':'Kitchen','bbox_xyxy':[.6,.6,1,1]}],
                        'edges':[{'a':'a','b':'b','relation':'open_connected'}]}
        annotation=self.source/'annotations/p.graph.json'
        annotation.write_text(json.dumps(self.reference))
        (self.source/'manifest.json').write_text(json.dumps([{'plan_id':'p','image_path':'images/p.png',
            'image_sha256':digest(image),'annotation_sha256':digest(annotation)}]))

    def init(self):
        return initialize(self.source,self.base/'runs','test','synthetic software test')

    def test_unique_run_and_input_snapshot(self):
        one,two=self.init(),self.init()
        self.assertNotEqual(one,two)
        (self.source/'images/p.png').write_bytes(b'changed original')
        verify_inputs(one)
        (one/'images/p.png').write_bytes(b'changed frozen snapshot')
        with self.assertRaises(ValueError): verify_inputs(one)

    def test_bad_manifest_hash_rejected_before_run_creation(self):
        (self.source/'images/p.png').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): self.init()
        self.assertFalse((self.base/'runs').exists())

    def test_evaluation_provenance_logs_and_no_overwrite(self):
        root=self.init()
        out=root/'predictions/egtr_frozen'
        out.mkdir()
        prediction=json.loads(json.dumps(self.reference))
        prediction['edges'][0]['relation']='connected_by_door'
        (out/'p.json').write_text(json.dumps(prediction))
        args=argparse.Namespace(stage='evaluate',run=root,min_iou=.3)
        self.assertEqual(execute(args),0)
        summary=refresh_summary(root)
        self.assertEqual(summary['edge_f1'],1.)
        self.assertEqual(summary['reference_kind'],'manual_reviewed')
        self.assertTrue((root/'logs/evaluate.log').exists())
        self.assertTrue((root/'provenance/evaluate/execution.json').exists())
        with self.assertRaises(ValueError): execute(args)
        self.assertFalse((root/'.stage-lock').exists())

    def test_qa_uses_same_access_collapse_without_mutation(self):
        context=graph_context(self.reference)
        self.assertEqual(context['edges'][0]['relation'],'direct_access')
        self.assertEqual(self.reference['edges'][0]['relation'],'open_connected')

    def test_qa_delta_uses_paired_questions(self):
        root=self.init()
        rows=[{'question_id':'one','arm':a,'status':'ok','correct':a!='image_only'}
              for a in ('image_only','reference_graph','egtr_graph')]
        rows += [{'question_id':'two','arm':'reference_graph','status':'ok','correct':False},
                 {'question_id':'two','arm':'egtr_graph','status':'unavailable'}]
        (root/'results/qa.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        qa=refresh_summary(root)['qa']
        self.assertEqual(qa['paired_questions'],1)
        self.assertEqual(qa['qa_gain_pp'],100)
        self.assertEqual(qa['qa_gap_pp'],0)


if __name__=='__main__': unittest.main()
