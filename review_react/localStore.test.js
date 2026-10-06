import test from 'node:test';
import assert from 'node:assert/strict';
import {promises as fs} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {localStore} from './localStore.js';
test('image upload, durable annotation, reopening and overwrite protection',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'fyp-store-test-'));
 const dataset=path.join(root,'dataset');
 await fs.mkdir(path.join(dataset,'colorful','803'),{recursive:true});
 const middleware=localStore(root,dataset),server=http.createServer((q,s)=>middleware(q,s,()=>{s.statusCode=404;s.end();}));
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${server.address().port}`;
 const post=async(url,body,headers={})=>{const r=await fetch(base+url,{method:'POST',body,headers});return {status:r.status,data:await r.json()};};
 try{
 const bytes=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jS1kAAAAASUVORK5CYII=','base64');
 await fs.writeFile(path.join(dataset,'colorful','803','F1_original.png'),bytes);
 const upload=await post('/api/local-plan/lookup',bytes);assert.equal(upload.status,200);
 await assert.rejects(fs.access(path.join(root,'images')));
 const annotation={plan_id:'test',reviewer:'tester',status:'manually_reviewed',all_pairs_reviewed:false,image:{sha256:upload.data.sha256},nodes:[{id:'test',label:'Room',polygon:[[0,0],[1,0],[0,1]]}],edges:[]};
 annotation.connectivity_check={start_node:'test',status:'pass',separate_components:[{room_ids:['garage','storage'],valid_separate_access:true}]};
 const payload={image_base64:bytes.toString('base64'),revision:0,annotation};
 const saved=await post('/api/local-plan/complete',JSON.stringify(payload));assert.equal(saved.data.revision,1);
 assert.equal(saved.data.annotation_path,'cubicasa_eval/annotations/803_colorful.graph.json');
 assert.deepEqual(JSON.parse(await fs.readFile(path.join(root,'exclusion_manifest.json'),'utf8')),[path.join(dataset,'colorful','803')+'/']);
 const reopened=await post('/api/local-plan/lookup',bytes);assert.deepEqual(reopened.data.annotation.nodes,annotation.nodes);
 assert.deepEqual(reopened.data.annotation.connectivity_check,annotation.connectivity_check);
 assert.equal((await fs.readdir(path.join(root,'images'))).length,1);
 assert.equal((await post('/api/local-plan/complete',JSON.stringify(payload))).status,409);
 assert.equal((await post('/api/local-plan/complete',JSON.stringify({...payload,annotation:{...annotation,status:'draft'}}))).status,400);
 assert.equal((await post('/api/local-plan/lookup',bytes,{Origin:'https://unrelated.example'})).status,403);
 }finally{server.closeAllConnections();await new Promise(r=>server.close(r));await fs.rm(root,{recursive:true,force:true});}
});
