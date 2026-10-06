import {promises as fs} from 'node:fs';
import path from 'node:path';

const V1_CONDITION_DIRS={image_only:'image_only',image_and_graph:'image_and_graph',graph_only:'graph_only'};
const V2_SCORE_LABELS={image_only:'image_only','image_graph:gold_manual':'image_graph','graph_only:gold_manual':'graph_only'};

function parseCsv(text){
  const rows=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const char=text[i],next=text[i+1];
    if(char==='"'&&quoted&&next==='"'){field+='"';i++;continue;}
    if(char==='"'){quoted=!quoted;continue;}
    if(char===','&&!quoted){row.push(field);field='';continue;}
    if((char==='\n'||char==='\r')&&!quoted){if(char==='\r'&&next==='\n')i++;row.push(field);field='';if(row.some(value=>value!==''))rows.push(row);row=[];continue;}
    field+=char;
  }
  if(field||row.length){row.push(field);rows.push(row);}
  const [header,...values]=rows;
  return values.map(values=>Object.fromEntries(header.map((key,index)=>[key,values[index]??''])));
}

function questions(rows){return rows.map(row=>({...row,supporting_node_ids:JSON.parse(row.supporting_node_ids||'[]'),supporting_edge_ids:JSON.parse(row.supporting_edge_ids||'[]'),supporting_path_ids:JSON.parse(row.supporting_path_ids||'[]')}));}

async function jsonl(directory){
  let names=[];try{names=(await fs.readdir(directory)).filter(name=>name.endsWith('.jsonl')).sort();}catch(error){if(error.code==='ENOENT')return [];throw error;}
  const records=[];
  for(const name of names){const text=await fs.readFile(path.join(directory,name),'utf8');for(const line of text.split(/\r?\n/))if(line.trim())records.push(JSON.parse(line));}
  return records;
}

function rawAnswer(record){return {prediction:record.parsed_answer??null,valid:Boolean(record.valid_output),raw_output:record.raw_output??'',model:record.model,graph_source:record.graph_source??null,evidence:record.evidence??null,reasoning:record.reasoning??null,evidence_valid:record.evidence_valid??null,evidence_errors:record.evidence_errors??null,score_only:false};}
function scoreAnswer(row){return {prediction:row.pred||null,valid:row.valid==='True',raw_output:'',model:null,graph_source:row.label.includes('gold_manual')?'gold_manual':null,evidence:null,reasoning:null,evidence_valid:null,evidence_errors:null,score_only:true};}

async function loadV1(qaRoot,annotationRoot){
  const questionRows=parseCsv(await fs.readFile(path.join(annotationRoot,'questions','manual20_qa_v1.csv'),'utf8'));
  const answers={};
  for(const [condition,directory] of Object.entries(V1_CONDITION_DIRS)){
    const records=await jsonl(path.join(qaRoot,directory));
    answers[condition]=Object.fromEntries(records.map(record=>[record.question_id,rawAnswer(record)]));
  }
  return {title:'V1 answer-only prompt',description:'Original 100-question Manual20 run with raw model outputs.',questions:questions(questionRows),answers,conditions:['image_only','image_and_graph','graph_only'],hasRawOutputs:true};
}

async function loadV2(resultBase,annotationRoot){
  const runDirectory=path.join(resultBase,'manual20_qa_v2_evidence');
  const resultDirectory=path.join(runDirectory,'results');
  const questionFile=path.join(annotationRoot,'questions','manual20_qa_v2_candidate.csv');
  const questionText=await fs.readFile(questionFile,'utf8');
  const rawConditions={image_only:'image_only',image_graph:'image_graph',graph_only:'graph_only'};
  const answers={image_only:{},image_graph:{},graph_only:{}};
  let rawCount=0;
  for(const [condition,directory] of Object.entries(rawConditions)){
    const records=await jsonl(path.join(runDirectory,directory));
    rawCount+=records.length;
    answers[condition]=Object.fromEntries(records.map(record=>[record.question_id,rawAnswer(record)]));
  }
  if(rawCount>0){
    const rawRecords=Object.values(answers).flatMap(records=>Object.values(records));
    const hasEvidenceReasoning=rawRecords.some(record=>record.evidence||record.reasoning);
    return hasEvidenceReasoning
      ?{title:'V2 evidence + reasoning prompt',description:'Five-category v2 run with raw answers, cited evidence, and reasoning.',questions:questions(parseCsv(questionText)),answers,conditions:['image_only','image_graph','graph_only'],hasRawOutputs:true}
      :{title:'V2 answer-only run',description:'Five-category v2 question set, but the SOC run used the earlier answer-only prompt. No evidence or reasoning was requested; rerun after syncing the updated QA runner.',questions:questions(parseCsv(questionText)),answers,conditions:['image_only','image_graph','graph_only'],hasRawOutputs:true};
  }

  const scoreText=await fs.readFile(path.join(resultDirectory,'qa_question_scores.csv'),'utf8');
  for(const row of parseCsv(scoreText)){const condition=V2_SCORE_LABELS[row.label];if(condition)answers[condition][row.question_id]=scoreAnswer(row);}
  return {title:'V2 evidence + reasoning prompt',description:'Five-category v2 run; evaluator scores are loaded locally. Download raw JSONL shards to inspect citations and reasoning.',questions:questions(parseCsv(questionText)),answers,conditions:['image_only','image_graph','graph_only'],hasRawOutputs:false};
}

async function annotations(annotationRoot){return Promise.all((await fs.readdir(path.join(annotationRoot,'annotations'))).filter(name=>name.endsWith('.graph.json')).sort().map(async name=>{const annotation=JSON.parse(await fs.readFile(path.join(annotationRoot,'annotations',name),'utf8'));return {...annotation,imageName:annotation.image?.local_path?path.basename(annotation.image.local_path):annotation.image?.name};}));}

export function qaStore(annotationRoot,qaRoot){
  return async(req,res,next)=>{
    const url=new URL(req.url,'http://localhost');
    if(url.pathname!=='/api/qa/data')return next();
    try{
      if(req.method!=='GET'){res.statusCode=405;return res.end('GET required');}
      const datasets={manual20_v1:await loadV1(qaRoot,annotationRoot)};
      try{datasets.manual20_v2_evidence=await loadV2(path.dirname(qaRoot),annotationRoot);}catch(error){if(error.code!=='ENOENT')throw error;}
      res.setHeader('Content-Type','application/json');
      res.end(JSON.stringify({annotations:await annotations(annotationRoot),datasets}));
    }catch(error){res.statusCode=400;res.setHeader('Content-Type','application/json');res.end(JSON.stringify({error:error.message}));}
  };
}
