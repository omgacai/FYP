import {promises as fs} from 'node:fs';
import path from 'node:path';
import {createHash, randomUUID} from 'node:crypto';
export function localStore(root, datasetRoot='/Users/numkoos/Downloads/cubicasa/cubicasa5k/cubicasa5k') {
  const matches=new Map();
  const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
  async function locate(hash,size) {
    if(matches.has(hash))return matches.get(hash);
    const found=[];
    for(const category of ['colorful','high_quality','high_quality_architectural']) {
      let folders;try{folders=await fs.readdir(path.join(datasetRoot,category),{withFileTypes:true});}catch(e){if(e.code==='ENOENT')continue;throw e;}
      for(const folder of folders.filter(d=>d.isDirectory()))for(const name of ['F1_original.png','F1_scaled.png']){
        const file=path.join(datasetRoot,category,folder.name,name);
        try{if((await fs.stat(file)).size===size && digest(await fs.readFile(file))===hash){found.push(`${category}/${folder.name}`);break;}}catch(e){if(e.code!=='ENOENT')throw e;}
      }
    }
    if(found.length>1)throw Error('Image matches multiple dataset plans; source identity is ambiguous.');
    const result=found[0]||null;matches.set(hash,result);return result;
  }
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

        const read=async file=>{try{return JSON.parse(await fs.readFile(file,'utf8'));}catch(e){if(e.code==='ENOENT')return null;throw e;}};
        const atomic=async(file,data)=>{const tmp=file+'.'+randomUUID()+'.tmp';await fs.writeFile(tmp,JSON.stringify(data,null,2)+'\n');await fs.rename(tmp,file);};
        if(url.pathname==='/api/local-plan/lookup' || url.pathname==='/api/local-plan/complete') {
          const completing=url.pathname.endsWith('/complete');
          const payload=completing?JSON.parse(body.toString()):null;
          const imageBytes=completing?Buffer.from(payload.image_base64,'base64'):body;
          const png=imageBytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])),jpg=imageBytes[0]===255&&imageBytes[1]===216;
          if(!png&&!jpg)throw Error('Only PNG and JPEG images are supported.');
          const hash=createHash('sha256').update(imageBytes).digest('hex');
          const sourceDir=await locate(hash,imageBytes.length);
          if(completing&&!sourceDir)throw Error('Cannot identify this image in the local CubiCasa dataset. Use its original PNG so the plan number and exclusion folder are correct.');
          const [category,number]=(sourceDir||'unknown/unknown').split('/');
          const key=sourceDir?`${number}_${category}`:`unmatched-${hash}`;
          const imageName=key+(png?'.png':'.jpg');
          const imagePath=path.join(root,'images',imageName);
          const annotation=await read(path.join(root,'annotations',key+'.graph.json'));
          if(completing){
            const record=payload.annotation;
            if(record?.image?.sha256!==hash || record.status!=='manually_reviewed' || !record.reviewer?.trim() || !record.plan_id?.trim() || !record.nodes?.length || record.nodes.some(n=>!n.label?.trim()||!n.polygon) || !Array.isArray(record.edges))throw Error('Complete room outlines and annotator details before saving.');
            if(annotation&&annotation.image?.sha256!==hash)throw Error('This plan already has an annotation for a different image version. Reopen that image to avoid overwriting it.');
            if(payload.revision!==(annotation?.storage_revision||0)){const e=Error('Another tab changed this annotation. Reopen the saved version before overwriting.');e.status=409;throw e;}
            await fs.mkdir(path.join(root,'images'),{recursive:true});await fs.mkdir(path.join(root,'annotations'),{recursive:true});
            try{await fs.writeFile(imagePath,imageBytes,{flag:'wx'});}catch(e){if(e.code!=='EEXIST')throw e;if(digest(await fs.readFile(imagePath))!==hash)throw Error('An image with this plan name already exists with different bytes.');}
            await atomic(path.join(root,'annotations',key+'.graph.json'),{...record,source_dir:sourceDir,image:{...record.image,local_path:`cubicasa_eval/images/${imageName}`},storage_revision:payload.revision+1,updated_at:new Date().toISOString()});
            const exclusionPath=path.join(root,'exclusion_manifest.json');
            const previousList=await read(exclusionPath);
            const folders=Array.isArray(previousList)?previousList:(previousList?.plans||[]).map(p=>path.dirname(p.local_image_path)+'/');
            const sourceFolder=path.join(datasetRoot,sourceDir)+'/';
            await atomic(exclusionPath,[...new Set([...folders,sourceFolder])]);
            return {revision:payload.revision+1,annotation_path:`cubicasa_eval/annotations/${key}.graph.json`};
          }
          return {key,sha256:hash,plan_id:sourceDir?key:`local-${hash.slice(0,16)}`,source_dir:sourceDir,image_path:`cubicasa_eval/images/${imageName}`,annotation_path:`cubicasa_eval/annotations/${key}.graph.json`,annotation,revision:annotation?.storage_revision||0};
        }
        throw Error('Unknown endpoint.');
      });
      queue=job.catch(()=>{});reply(200,await job);
    } catch(e){reply(e.status||400,{error:e.message});}
  };
}
