// Shared by the read-only browser evaluator and the batch CLI. No label-based matching.
export const VERSION='floorplan-evaluation/1';
export const RELATIONS=['connected_by_door','open_connected','adjacent_to','uncertain'];
const aliases={bedroom:'Bedroom',bathroom:'Bath',bath:'Bath',kitchen:'Kitchen',living_room:'LivingRoom',livingroom:'LivingRoom',dining_room:'Dining',dining:'Dining',corridor:'Corridor',storage:'Storage',entrance:'Entry',entry:'Entry',garage:'Garage',balcony:'Outdoor',outdoor:'Outdoor',other:'Other'};
const pair=(a,b)=>JSON.stringify([a,b].sort());
export function normalizeGraph(record,{relationMode='fine'}={}){
  if(!['fine','access'].includes(relationMode))throw Error('relationMode must be fine or access');
  if(record.valid===false)throw Error(record.error||'Prediction marked invalid');
  const g=record.graph||record;
  if(!Array.isArray(g.nodes||g.rooms)||!Array.isArray(g.edges))throw Error('Expected nodes/rooms and edges arrays');
  const vlm=!g.nodes;
  const nodes=(g.nodes||g.rooms).map(n=>{
    if(typeof n.id!=='string'||!n.id||typeof n.type!=='string'||!n.type)throw Error('Every room needs a string id and type');
    let box=n.bbox_xyxy;
    if(vlm&&n.bbox)box=n.bbox.map(v=>v/1000); // Existing Qwen spatial schema uses 0..1000, not pixels.
    if(!box&&n.polygon?.length)box=[Math.min(...n.polygon.map(p=>p[0])),Math.min(...n.polygon.map(p=>p[1])),Math.max(...n.polygon.map(p=>p[0])),Math.max(...n.polygon.map(p=>p[1]))];
    if(!Array.isArray(box)||box.length!==4||!box.every(v=>Number.isFinite(v)&&v>=0&&v<=1)||box[0]>=box[2]||box[1]>=box[3])throw Error(`Room ${n.id}: valid normalized bbox_xyxy required (Qwen bbox uses 0..1000)`);
    return {...n,type:aliases[n.type.toLowerCase()]||n.type,bbox_xyxy:box,x:(box[0]+box[2])/2,y:(box[1]+box[3])/2};
  });
  const ids=new Set(nodes.map(n=>n.id));if(ids.size!==nodes.length)throw Error('Duplicate room ids');
  const seen=new Map();
  for(const e of g.edges){const a=e.a??e.source,b=e.b??e.target;let relation=e.relation??e.type;
    if(relationMode==='access'&&['connected_by_door','open_connected'].includes(relation))relation='direct_access';
    if(!ids.has(a)||!ids.has(b)||a===b||![...RELATIONS,'direct_access'].includes(relation))throw Error('Invalid edge endpoint or unsupported relation');
    const key=pair(a,b);if(seen.has(key)&&seen.get(key).relation!==relation)throw Error('Conflicting relations on the same undirected pair');
    seen.set(key,{a,b,relation});
  }
  return {plan_id:record.plan_id||g.plan_id,nodes,edges:[...seen.values()]};
}
export function iou(a,b){const area=x=>(x[2]-x[0])*(x[3]-x[1]);const overlap=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));return overlap/(area(a)+area(b)-overlap);}
// Rectangular Hungarian minimization, columns >= rows. Dummy columns allow unmatched rooms.
function assignment(cost){const n=cost.length;if(!n)return [];const m=cost[0].length,u=Array(n+1).fill(0),v=Array(m+1).fill(0),p=Array(m+1).fill(0),way=Array(m+1).fill(0);
  for(let i=1;i<=n;i++){p[0]=i;let j0=0;const min=Array(m+1).fill(Infinity),used=Array(m+1).fill(false);do{used[j0]=true;const i0=p[j0];let delta=Infinity,j1=0;for(let j=1;j<=m;j++)if(!used[j]){const cur=cost[i0-1][j-1]-u[i0]-v[j];if(cur<min[j]){min[j]=cur;way[j]=j0;}if(min[j]<delta){delta=min[j];j1=j;}}for(let j=0;j<=m;j++){if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else min[j]-=delta;}j0=j1;}while(p[j0]!==0);do{const j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0);}
  const result=Array(n).fill(-1);for(let j=1;j<=m;j++)if(p[j])result[p[j]-1]=j-1;return result;
}
export function metrics(tp,fp,fn){return {tp,fp,fn,precision:tp+fp?tp/(tp+fp):null,recall:tp+fn?tp/(tp+fn):null,f1:2*tp+fp+fn?2*tp/(2*tp+fp+fn):null};}
export function evaluateGraph(reference,prediction,{minIoU=0.3,completePairs=false,relationMode='fine'}={}){
  if(!(minIoU>0&&minIoU<=1))throw Error('minIoU must be in (0,1]');
  const gold=normalizeGraph(reference,{relationMode}),pred=normalizeGraph(prediction,{relationMode});
  if(gold.plan_id&&pred.plan_id&&gold.plan_id!==pred.plan_id)throw Error('Plan IDs do not match');
  const weights=pred.nodes.map(p=>gold.nodes.map(g=>iou(p.bbox_xyxy,g.bbox_xyxy)));
  const assign=assignment(weights.map(row=>[...row.map(w=>w>=minIoU?-(1+w/(Math.min(gold.nodes.length,pred.nodes.length)+1)):1),...pred.nodes.map(()=>0)]));
  const mapping=new Map(),used=new Set(),matches=[];
  assign.forEach((j,i)=>{if(j<gold.nodes.length&&j>=0&&weights[i][j]>=minIoU){mapping.set(pred.nodes[i].id,gold.nodes[j].id);used.add(gold.nodes[j].id);matches.push({pred_id:pred.nodes[i].id,gold_id:gold.nodes[j].id,iou:weights[i][j]});}});
  const nodeErrors=[];for(const p of pred.nodes){const g=gold.nodes.find(g=>g.id===mapping.get(p.id));if(!g)nodeErrors.push({kind:'extra',pred:p});else if(g.type!==p.type)nodeErrors.push({kind:'wrong_type',gold:g,pred:p});}for(const g of gold.nodes)if(!used.has(g.id))nodeErrors.push({kind:'missing',gold:g});
  const ge=new Map(gold.edges.map(e=>[pair(e.a,e.b),e])),pe=new Map(),edgeErrors=[],excluded=[],correct=[];
  for(const e of pred.edges){const a=mapping.get(e.a),b=mapping.get(e.b);if(!a||!b){edgeErrors.push({kind:'extra',pred:e,incident_unmatched:true});continue;}pe.set(pair(a,b),e);}
  // Old saves with all_pairs_reviewed=false do not establish negative labels unless caller explicitly opts in.
  const exhaustive=completePairs||reference.all_pairs_reviewed===true;
  for(const [key,g] of ge){const p=pe.get(key);if(g.relation==='uncertain'){excluded.push({gold:g,pred:p,reason:'uncertain reference'});pe.delete(key);continue;}if(!p)edgeErrors.push({kind:'missing',gold:g,incident_unmatched:!used.has(g.a)||!used.has(g.b)});else if(p.relation!==g.relation)edgeErrors.push({kind:'wrong_type',gold:g,pred:p});else correct.push({gold:g,pred:p});pe.delete(key);}
  for(const p of pe.values()){if(exhaustive)edgeErrors.push({kind:'extra',pred:p});else excluded.push({pred:p,reason:'unreviewed absent pair'});}
  const counts=Object.fromEntries(['missing','extra','wrong_type'].map(k=>[k,{nodes:nodeErrors.filter(e=>e.kind===k).length,edges:edgeErrors.filter(e=>e.kind===k).length}]));
  const perRelation=Object.fromEntries((relationMode==='access'?['direct_access','adjacent_to']:['connected_by_door','open_connected','direct_access','adjacent_to']).map(r=>[r,metrics(correct.filter(e=>e.gold.relation===r).length,edgeErrors.filter(e=>e.pred?.relation===r).length,edgeErrors.filter(e=>e.gold?.relation===r).length)]));
  const edgeMetric=metrics(correct.length,edgeErrors.filter(e=>e.pred).length,edgeErrors.filter(e=>e.gold).length);
  const edit=nodeErrors.length+edgeErrors.filter(e=>!e.incident_unmatched).length;
  return {schema_version:VERSION,plan_id:gold.plan_id,settings:{minIoU,completePairs,relationMode,matching:'maximum-cardinality then maximum-IoU, class-blind',exhaustive},gold,pred,matches,nodeErrors,edgeErrors,excluded,correct,counts,node_metrics:metrics(matches.length,counts.extra.nodes,counts.missing.nodes),type_accuracy:matches.length?(matches.length-counts.wrong_type.nodes)/matches.length:null,edge_metrics:edgeMetric,per_relation:perRelation,aligned_edit_cost:edit,normalized_edit_cost:edit/Math.max(1,gold.nodes.length+gold.edges.filter(e=>e.relation!=='uncertain').length),warnings:exhaustive?[]:['Absent reference pairs are not confirmed negatives; extra-edge scores have partial coverage.']};
}
export function demoPrediction(reference){const p=structuredClone(reference);p.demo=true;p.status='synthetic_demo';if(p.nodes.length)p.nodes[0].type=p.nodes[0].type==='Kitchen'?'Bedroom':'Kitchen';if(p.edges.length)p.edges[0].relation=p.edges[0].relation==='adjacent_to'?'connected_by_door':'adjacent_to';if(p.edges.length>1)p.edges.splice(1,1);if(p.nodes.length>2){const removed=p.nodes.pop().id;p.edges=p.edges.filter(e=>e.a!==removed&&e.b!==removed);}p.nodes.push({id:'synthetic-extra',type:'Other',label:'Synthetic extra',bbox_xyxy:[0.01,0.01,0.09,0.09]});return p;}
