import test from 'node:test';
import assert from 'node:assert/strict';
import {promises as fs} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {localStore} from './localStore.js';
test('image upload, durable annotation, reopening and overwrite protection',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'fyp-store-test-'));
 const middleware=localStore(root),server=http.createServer((q,s)=>middleware(q,s,()=>{s.statusCode=404;s.end();}));
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${server.address().port}`;
 const post=async(url,body,headers={})=>{const r=await fetch(base+url,{method:'POST',body,headers});return {status:r.status,data:await r.json()};};
 try{
 const bytes=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jS1kAAAAASUVORK5CYII=','base64');
 const upload=await post('/api/local-plan/upload',bytes);assert.equal(upload.status,200);
 const annotation={image:{sha256:upload.data.sha256},nodes:[{id:'test'}],edges:[]};
 const payload={key:upload.data.key,revision:0,annotation};
 const saved=await post('/api/local-plan/save',JSON.stringify(payload));assert.equal(saved.data.revision,1);
 const reopened=await post('/api/local-plan/upload',bytes);assert.deepEqual(reopened.data.annotation.nodes,annotation.nodes);
 assert.equal((await fs.readdir(path.join(root,'images'))).length,1);
 assert.equal((await post('/api/local-plan/save',JSON.stringify(payload))).status,409);
 assert.equal((await post('/api/local-plan/save',JSON.stringify({...payload,key:'../../escape'}))).status,400);
 assert.equal((await post('/api/local-plan/upload',bytes,{Origin:'https://unrelated.example'})).status,403);
 }finally{server.closeAllConnections();await new Promise(r=>server.close(r));await fs.rm(root,{recursive:true,force:true});}
});
