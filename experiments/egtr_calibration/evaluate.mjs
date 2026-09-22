// Calibration/reference evaluation; raw fine-grained labels remain unchanged on disk.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {evaluateGraph,metrics} from '../../review_react/src/graphEvaluation.js';
const root = path.resolve(process.argv[2] || 'cubicasa_eval/calibration/egtr_v1');
const iouArg = process.argv.indexOf('--min-iou');
const minIoU = iouArg < 0 ? 0.3 : Number(process.argv[iouArg + 1]);
const predictionArg = process.argv.indexOf('--prediction-dir');
const predictionRelative = predictionArg < 0 ? 'predictions/egtr_frozen' : process.argv[predictionArg + 1];
if (!predictionRelative) throw Error('--prediction-dir needs a relative path');
const predictionDir = path.resolve(root, predictionRelative);
if (!predictionDir.startsWith(root + path.sep)) throw Error('--prediction-dir must stay inside the run root');
if (!(minIoU > 0 && minIoU <= 1)) throw Error('min-iou must be in (0,1]');
const output = path.join(root, 'results/graph_metrics.json');
try {await fs.access(output); throw Error(`Refusing to overwrite ${output}; create a new run`);}
catch(error) {if(error.code !== 'ENOENT') throw error;}
const manifest = JSON.parse(await fs.readFile(path.join(root, 'manifest.json')));
const results = [], kinds = new Set();
for (const row of manifest) {
  const referenceBytes = await fs.readFile(path.join(root, 'annotations', `${row.plan_id}.graph.json`));
  const imageBytes = await fs.readFile(path.join(root, row.image_path));
  for (const [bytes, expected] of [[referenceBytes, row.annotation_sha256], [imageBytes, row.image_sha256]]) {
    if (createHash('sha256').update(bytes).digest('hex') !== expected) throw Error(`Snapshot changed: ${row.plan_id}`);
  }
  const reference = JSON.parse(referenceBytes);
  if (!['pipeline_reference','manually_reviewed'].includes(reference.status)) throw Error('Unreviewed/unknown reference status');
  kinds.add(reference.status === 'manually_reviewed' ? 'manual_reviewed' : 'pipeline_silver');
  try {
    const prediction = JSON.parse(await fs.readFile(path.join(predictionDir, `${row.plan_id}.json`)));
    if (prediction.valid === false) {
      results.push({plan_id: row.plan_id, status: 'unavailable', reason: prediction.error});
      continue;
    }
    results.push({plan_id: row.plan_id, status: 'evaluated', reference_provenance: reference.provenance,
      unsupported_reference_relations: reference.unsupported_reference_relations,
      ...evaluateGraph(reference, prediction, {minIoU, completePairs: false, relationMode: 'access'})});
  } catch (error) {
    results.push({plan_id: row.plan_id, status: error.code === 'ENOENT' ? 'missing_prediction' : 'invalid', error: error.message});
  }
}
const scored = results.filter(r => r.status === 'evaluated');
const sum = (key, metric) => scored.reduce((s, r) => s+r[key][metric], 0);
const mean = values => values.length ? values.reduce((a,b)=>a+b,0)/values.length : null;
const matched = scored.flatMap(r => r.matches);
const report = {
  schema_version: 'egtr-graph-report/2', reference_kind: [...kinds].sort().join('+'),
  settings: {minIoU, relationMode: 'access', collapse: ['connected_by_door','open_connected'], target: 'direct_access', predictionDir: predictionRelative},
  note: 'Metrics cover evaluated plans only; inspect coverage. Door-only silver rules miss open-passage access. Absent unreviewed pairs are not verified negatives.',
  coverage: Object.fromEntries(['total','evaluated','unavailable','missing_prediction','invalid'].map(s => [s, s==='total'?results.length:results.filter(r=>r.status===s).length])),
  micro_nodes: metrics(sum('node_metrics','tp'),sum('node_metrics','fp'),sum('node_metrics','fn')),
  micro_edges: metrics(sum('edge_metrics','tp'),sum('edge_metrics','fp'),sum('edge_metrics','fn')),
  macro_node_f1: mean(scored.map(r=>r.node_metrics.f1).filter(x=>x!==null)),
  macro_edge_f1: mean(scored.map(r=>r.edge_metrics.f1).filter(x=>x!==null)),
  macro_matched_type_accuracy: mean(scored.map(r=>r.type_accuracy).filter(x=>x!==null)),
  matched_nodes: matched.length,
  mean_matched_iou: mean(matched.map(match=>match.iou)),
  micro_per_relation: Object.fromEntries(['direct_access','adjacent_to'].map(rel => [rel, metrics(...['tp','fp','fn'].map(k=>scored.reduce((s,r)=>s+r.per_relation[rel][k],0)))])),
  mean_aligned_edit_cost: mean(scored.map(r=>r.aligned_edit_cost)),
  excluded_pairs: scored.reduce((s,r)=>s+r.excluded.length,0), per_plan: results
};
await fs.mkdir(path.join(root, 'results'), {recursive: true});
await fs.writeFile(output, JSON.stringify(report, null, 2) + '\n', {flag:'wx'});
console.log(JSON.stringify({coverage: report.coverage, node_f1: report.micro_nodes.f1, edge_f1: report.micro_edges.f1}));
