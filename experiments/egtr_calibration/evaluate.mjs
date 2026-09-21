// Explicit silver-reference evaluator. Does not mark pipeline references manually reviewed.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {evaluateGraph} from '../../review_react/src/graphEvaluation.js';
const root = path.resolve(process.argv[2] || 'cubicasa_eval/calibration/egtr_v1');
const manifest = JSON.parse(await fs.readFile(path.join(root, 'manifest.json')));
const results = [];
for (const row of manifest) {
  const referenceBytes = await fs.readFile(path.join(root, 'annotations', `${row.plan_id}.graph.json`));
  const imageBytes = await fs.readFile(path.join(root, row.image_path));
  for (const [bytes, expected] of [[referenceBytes, row.annotation_sha256], [imageBytes, row.image_sha256]]) {
    if (createHash('sha256').update(bytes).digest('hex') !== expected) throw Error(`Snapshot changed: ${row.plan_id}`);
  }
  const reference = JSON.parse(referenceBytes);
  try {
    const prediction = JSON.parse(await fs.readFile(path.join(root, 'predictions/egtr_frozen', `${row.plan_id}.json`)));
    if (prediction.valid === false) {
      results.push({plan_id: row.plan_id, status: 'unavailable', reason: prediction.error});
      continue;
    }
    results.push({plan_id: row.plan_id, status: 'evaluated', reference_provenance: reference.provenance,
      unsupported_reference_relations: reference.unsupported_reference_relations,
      ...evaluateGraph(reference, prediction, {minIoU: 0.3, completePairs: false})});
  } catch (error) {
    results.push({plan_id: row.plan_id, status: error.code === 'ENOENT' ? 'missing_prediction' : 'invalid', error: error.message});
  }
}
const report = {reference_kind: 'pipeline_silver', note: 'Provisional IoU=0.3. Source rules do not detect open passages; absent pairs are not verified negatives. Not manual-gold accuracy.',
  coverage: {total: results.length, evaluated: results.filter(r => r.status === 'evaluated').length}, per_plan: results};
await fs.mkdir(path.join(root, 'results'), {recursive: true});
await fs.writeFile(path.join(root, 'results/graph_metrics.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report.coverage));
