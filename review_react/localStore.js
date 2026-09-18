import {promises as fs} from 'node:fs';
import path from 'node:path';
import {createHash, randomUUID} from 'node:crypto';
export function localStore(root) {
  let queue=Promise.resolve();
  return async (req,res,next)=>{
    if(!req.url.startsWith('/api/local-plan'))return next();
    const reply=(status,data)=>{res.statusCode=status;res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));};
    const origin=req.headers.origin;
    if(origin&&origin!==`http://${req.headers.host}`)return reply(403,{error:'Cross-origin writes are not allowed.'});
    if(req.method!=='POST')return reply(405,{error:'POST required.'});
    try {
      let size=0;const chunks=[];for await(const chunk of req){size+=chunk.length;if(size>40*1024*1024)throw Error('File exceeds 40 MB.');chunks.push(chunk);}
      const body=Buffer.concat(chunks),url=new URL(req.url,'http://localhost');
      const job=queue.then(async()=>{
        await fs.mkdir(path.join(root,'images'),{recursive:true});await fs.mkdir(path.join(root,'annotations'),{recursive:true});
        const read=async file=>{try{return JSON.parse(await fs.readFile(file,'utf8'));}catch(e){if(e.code==='ENOENT')return null;throw e;}};
        const atomic=async(file,data)=>{const tmp=file+'.'+randomUUID()+'.tmp';await fs.writeFile(tmp,JSON.stringify(data,null,2)+'\n');await fs.rename(tmp,file);};
        if(url.pathname==='/api/local-plan/upload') {
          const png=body.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])),jpg=body[0]===255&&body[1]===216;
          if(!png&&!jpg)throw Error('Only PNG and JPEG images are supported.');
          const hash=createHash('sha256').update(body).digest('hex');
          const manifest=await read(path.join(root,'exclusion_manifest.json'));
          const match=manifest?.plans?.find(p=>p.image_sha256===hash);
          const prefix=match?.source_dir?.replace(/[^a-zA-Z0-9_-]/g,'_')||'plan';
          const key=`${prefix}--${hash}`,imageName=key+(png?'.png':'.jpg');
          const imagePath=path.join(root,'images',imageName);
          try{await fs.writeFile(imagePath,body,{flag:'wx'});}catch(e){if(e.code!=='EEXIST')throw e;}
          const annotation=await read(path.join(root,'annotations',key+'.graph.json'));
          return {key,sha256:hash,plan_id:match?.plan_id||`local-${hash.slice(0,16)}`,source_dir:match?.source_dir||null,image_path:`cubicasa_eval/images/${imageName}`,annotation_path:`cubicasa_eval/annotations/${key}.graph.json`,annotation,revision:annotation?.storage_revision||0};
        }
        if(url.pathname==='/api/local-plan/save') {
          const {key,revision,annotation}=JSON.parse(body.toString());
          if(typeof key!=='string'||!/^[-a-zA-Z0-9_]+--[a-f0-9]{64}$/.test(key))throw Error('Invalid storage key.');
          const hash=key.slice(-64);
          if(annotation?.image?.sha256!==hash||!Array.isArray(annotation.nodes)||!Array.isArray(annotation.edges))throw Error('Invalid annotation or image identity.');
          const files=await fs.readdir(path.join(root,'images'));if(!files.some(f=>f===key+'.png'||f===key+'.jpg'))throw Error('Upload the original image first.');
          const target=path.join(root,'annotations',key+'.graph.json'),previous=await read(target);
          if(revision!==(previous?.storage_revision||0)){const e=Error('Another tab changed this annotation. Download your draft, then reopen the image to reload the saved version.');e.status=409;throw e;}
          const saved={...annotation,storage_revision:revision+1,updated_at:new Date().toISOString()};await atomic(target,saved);
          return {revision:saved.storage_revision,annotation_path:`cubicasa_eval/annotations/${key}.graph.json`};
        }
        throw Error('Unknown endpoint.');
      });
      queue=job.catch(()=>{});reply(200,await job);
    } catch(e){reply(e.status||400,{error:e.message});}
  };
}
