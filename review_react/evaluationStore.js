import {promises as fs} from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
export async function loadBenchmark(root){
 const names=(await fs.readdir(path.join(root,'annotations'))).filter(n=>n.endsWith('.graph.json')).sort();
 const plans=[];for(const name of names){const raw=await fs.readFile(path.join(root,'annotations',name));const annotation=JSON.parse(raw);const imageName=path.basename(annotation.image.local_path);let imageError=null;try{const bytes=await fs.readFile(path.join(root,'images',imageName));if(createHash('sha256').update(bytes).digest('hex')!==annotation.image.sha256)imageError='Image hash mismatch';}catch(e){imageError=e.message;}
 plans.push({annotation,imageName,imageError,annotation_sha256:createHash('sha256').update(raw).digest('hex')});}
 const ids=plans.map(p=>p.annotation.plan_id);if(new Set(ids).size!==ids.length)throw Error('Duplicate benchmark plan IDs');return plans;
}
export async function loadPredictions(root){
 const methods={};let dirs=[];try{dirs=await fs.readdir(path.join(root,'predictions'),{withFileTypes:true});}catch(e){if(e.code!=='ENOENT')throw e;}
 for(const dir of dirs.filter(d=>d.isDirectory())){const records={};for(const file of (await fs.readdir(path.join(root,'predictions',dir.name))).filter(n=>n.endsWith('.json')).sort()){
 const fallback=file.replace(/(?:\.graph)?\.json$/,'');try{const record=JSON.parse(await fs.readFile(path.join(root,'predictions',dir.name,file),'utf8'));const id=record.plan_id||fallback;if(records[id])throw Error(`Duplicate prediction for ${id}`);records[id]={...record,plan_id:id};}catch(e){records[fallback]={plan_id:fallback,valid:false,error:e.message};}}
 methods[dir.name]=records;}return methods;
}
export function evaluationStore(root){return async(req,res,next)=>{const url=new URL(req.url,'http://localhost');if(!url.pathname.startsWith('/api/evaluation/'))return next();try{if(req.method!=='GET'){res.statusCode=405;return res.end('GET required');}
 if(url.pathname==='/api/evaluation/benchmark'){res.setHeader('Content-Type','application/json');return res.end(JSON.stringify({plans:await loadBenchmark(root),methods:await loadPredictions(root)}));}
 if(url.pathname==='/api/evaluation/image'){const name=url.searchParams.get('name');if(!name||path.basename(name)!==name||! /\.(png|jpe?g)$/i.test(name))throw Error('Invalid image name');res.setHeader('Content-Type',/\.png$/i.test(name)?'image/png':'image/jpeg');return res.end(await fs.readFile(path.join(root,'images',name)));}
 res.statusCode=404;res.end();}catch(e){res.statusCode=400;res.setHeader('Content-Type','application/json');res.end(JSON.stringify({error:e.message}));}};}
