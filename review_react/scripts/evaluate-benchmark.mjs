import {promises as fs} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {loadBenchmark,loadPredictions} from '../evaluationStore.js';
import {evaluateGraph,metrics,VERSION} from '../src/graphEvaluation.js';
const args=process.argv.slice(2),get=(name,def)=>{const i=args.indexOf(name);return i<0?def:args[i+1];};
const root=path.resolve(get('--root',fileURLToPath(new URL('../../cubicasa_eval',import.meta.url))));
const minIoU=Number(get('--min-iou','0.3')),completePairs=args.includes('--complete-pairs'),relationMode=get('--relation-mode','access');
const plans=await loadBenchmark(root),methods=await loadPredictions(root),results={};
for(const [method,predictions] of Object.entries(methods)){
 const per_plan=plans.map(({annotation,imageError,annotation_sha256})=>{const base={plan_id:annotation.plan_id,annotation_sha256};try{if(imageError)throw Error(imageError);if(annotation.status!=='manually_reviewed')throw Error('Reference is not reviewed');if(!predictions[annotation.plan_id])return {...base,status:'missing_prediction'};return {...base,status:'evaluated',...evaluateGraph(annotation,predictions[annotation.plan_id],{minIoU,completePairs,relationMode})};}catch(e){return {...base,status:'invalid',error:e.message};}});
 const scored=per_plan.filter(p=>p.status==='evaluated'),sum=(key,metric)=>scored.reduce((s,p)=>s+p[key][metric],0);
 results[method]={coverage:{total:plans.length,evaluated:scored.length,missing:per_plan.filter(p=>p.status==='missing_prediction').length,invalid:per_plan.filter(p=>p.status==='invalid').length},micro_nodes:metrics(sum('node_metrics','tp'),sum('node_metrics','fp'),sum('node_metrics','fn')),micro_edges:metrics(sum('edge_metrics','tp'),sum('edge_metrics','fp'),sum('edge_metrics','fn')),per_plan};
}
const report={schema_version:VERSION,generated_at:new Date().toISOString(),settings:{minIoU,completePairs,relationMode},reference_count:plans.length,reference_snapshot:plans.map(p=>({plan_id:p.annotation.plan_id,sha256:p.annotation_sha256,image_error:p.imageError})),methods:results};
const output=path.resolve(get('--output',path.join(root,'results','evaluation.json')));await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({output,reference_count:plans.length,methods:Object.fromEntries(Object.entries(results).map(([k,v])=>[k,v.coverage]))},null,2));
