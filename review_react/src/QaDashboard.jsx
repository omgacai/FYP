import React,{useEffect,useMemo,useState} from 'react';
import './qa.css';

const conditionLabels={image_only:'Image only',image_and_graph:'Image + manual graph',image_graph:'Image + manual graph',graph_only:'Manual graph only'};
const relationStyle={direct_access:{color:'#c026d3',dash:null,label:'direct access'},connected_by_door:{color:'#c026d3',dash:null,label:'door'},open_connected:{color:'#f97316',dash:'8 5',label:'open access'},adjacent_to:{color:'#2563eb',dash:'2 6',label:'adjacent'},uncertain:{color:'#64748b',dash:'8 4 2 4',label:'uncertain'}};
const answerText=value=>Array.isArray(value)?value.join(', '):value==null?'Invalid output':String(value);

function GraphOverlay({annotation,question,imageUrl}){
  const [showAll,setShowAll]=useState(true);
  const width=annotation.image.width,height=annotation.image.height;
  const supported=new Set(question.supporting_node_ids||[]),edgeIds=new Set(question.supporting_edge_ids||[]);
  const edgeId=edge=>`${[edge.a,edge.b].sort().join('--')}:${edge.relation}`;
  return <section className="qa-plan"><div className="qa-plan-title"><div><p>Manual graph overlay</p><small>Gold graph; purple = direct access, blue = adjacency</small></div><label><input type="checkbox" checked={showAll} onChange={event=>setShowAll(event.target.checked)}/> Show all graph relations</label></div><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Floor plan ${annotation.plan_id} with manual graph`}>
    <image href={imageUrl} width={width} height={height}/>
    {annotation.edges.map((edge,index)=>{const a=annotation.nodes.find(node=>node.id===edge.a),b=annotation.nodes.find(node=>node.id===edge.b);if(!a||!b)return null;const style=relationStyle[edge.relation]||relationStyle.uncertain,highlight=edgeIds.has(edgeId(edge));if(!showAll&&!highlight)return null;return <g key={index}><line x1={a.x*width} y1={a.y*height} x2={b.x*width} y2={b.y*height} stroke={style.color} strokeWidth={highlight?7:3} strokeDasharray={style.dash||undefined} opacity={highlight?1:.6}/><title>{`${a.label} ↔ ${b.label}: ${style.label}`}</title></g>;})}
    {annotation.nodes.map(node=>{const selected=supported.has(node.id);return <g key={node.id}><circle cx={node.x*width} cy={node.y*height} r={selected?13:9} fill={selected?'#f59e0b':'#0f766e'} stroke="white" strokeWidth="3"/><text x={node.x*width} y={node.y*height-16} textAnchor="middle" className="qa-node-label">{node.label}</text><title>{`${node.label}: ${node.type}`}</title></g>;})}
  </svg></section>;
}

function AnswerCard({name,answer,gold}){
  const correct=answer?.valid&&String(answer.prediction).toLowerCase()===String(gold).toLowerCase();
  const evidence=answer?.evidence;
  const readableEvidence=answer?.evidence_display||evidence;
  const visualEvidence=Array.isArray(evidence);
  return <article className={`qa-answer ${correct?'correct':'incorrect'}`}><p>{name}</p><strong>{answerText(answer?.prediction)}</strong><span>{correct?'✓ Correct':answer?.valid?'✕ Incorrect':'✕ Invalid output'}</span>{answer?.score_only?<small className="qa-answer-note">Evaluator score only — raw output, citations, and reasoning were not downloaded.</small>:<>{answer?.reasoning&&<div className="qa-model-reasoning"><b>Model reasoning</b><span>{answer.reasoning}</span></div>}{evidence&&<div className="qa-model-evidence"><b>{visualEvidence?'Model visual evidence':'Model citations'}</b>{visualEvidence?<span>{evidence.join(' ')}</span>:<span>Rooms: {readableEvidence.node_ids?.length?readableEvidence.node_ids.join(', '):'None'}<br/>Relations: {readableEvidence.edge_ids?.length?readableEvidence.edge_ids.join(', '):'None'}</span>}</div>}{answer?.evidence_valid===false&&<small className="qa-answer-note">Citation validation: {answer.evidence_errors?.join(', ')||'invalid citation'}</small>}<details><summary>Raw model output</summary><pre>{answer?.raw_output||'No answer record found.'}</pre></details></>}</article>;
}

export default function QaDashboard(){
  const [data,setData]=useState(null),[error,setError]=useState(''),[datasetId,setDatasetId]=useState('manual20_v1'),[planId,setPlanId]=useState(''),[questionId,setQuestionId]=useState(''),[onlyInteresting,setOnlyInteresting]=useState(false);
  useEffect(()=>{fetch('/api/qa/data').then(async response=>{const body=await response.json();if(!response.ok)throw Error(body.error||'Could not load QA data');setData(body);const preferred=body.datasets.manual20_v2_evidence_aliases_v2?'manual20_v2_evidence_aliases_v2':body.datasets.manual20_v2_evidence?'manual20_v2_evidence':Object.keys(body.datasets)[0];setDatasetId(preferred);const first=body.annotations[0]?.plan_id;setPlanId(first);setQuestionId(body.datasets[preferred].questions.find(question=>question.plan_id===first)?.question_id||'');}).catch(issue=>setError(issue.message));},[]);
  const dataset=data?.datasets[datasetId];
  const annotation=data?.annotations.find(item=>item.plan_id===planId);
  const planQuestions=useMemo(()=>{let rows=(dataset?.questions||[]).filter(question=>question.plan_id===planId);if(onlyInteresting&&dataset){rows=rows.filter(question=>dataset.conditions.some(condition=>{const answer=dataset.answers[condition]?.[question.question_id];return !answer?.valid||String(answer.prediction).toLowerCase()!==String(question.gold_answer).toLowerCase();}));}return rows;},[dataset,planId,onlyInteresting]);
  useEffect(()=>{if(planQuestions.length&&!planQuestions.some(question=>question.question_id===questionId))setQuestionId(planQuestions[0].question_id);},[planQuestions,questionId]);
  const question=dataset?.questions.find(item=>item.question_id===questionId);
  const answers=dataset?.answers;
  if(error)return <main className="qa-dashboard"><p className="qa-error">{error}</p></main>;
  if(!data||!dataset||!annotation||!question)return <main className="qa-dashboard"><p>Loading Manual20 QA records…</p></main>;
  const imageUrl=`/api/evaluation/image?name=${encodeURIComponent(annotation.imageName)}`;
  return <main className="qa-dashboard"><header><div><p className="qa-kicker">MANUAL20 · QA ERROR REVIEW</p><h1>Image, graph, and answer comparison</h1><span>Manual graph is gold/oracle context, not an extracted graph prediction.</span></div><a href="/?mode=evaluate">Graph extraction review ↗</a></header>
    <section className="qa-controls"><label>Evaluation set<select value={datasetId} onChange={event=>{const id=event.target.value;setDatasetId(id);const first=data.annotations[0]?.plan_id;setPlanId(first);setQuestionId(data.datasets[id].questions.find(question=>question.plan_id===first)?.question_id||'');}}>{Object.entries(data.datasets).map(([id,item])=><option key={id} value={id}>{item.title}</option>)}</select></label><label>Floor plan<select value={planId} onChange={event=>setPlanId(event.target.value)}>{data.annotations.map(item=><option key={item.plan_id} value={item.plan_id}>{item.plan_id}</option>)}</select></label><label>Question<select value={questionId} onChange={event=>setQuestionId(event.target.value)}>{planQuestions.map(item=><option key={item.question_id} value={item.question_id}>{item.question_id} · {item.difficulty} · {item.category}</option>)}</select></label><label className="qa-checkbox"><input type="checkbox" checked={onlyInteresting} onChange={event=>setOnlyInteresting(event.target.checked)}/> Only errors / disagreements</label></section>
    <p className="qa-dataset-note">{dataset.description}</p>
    <section className="qa-question"><div><p>{question.question_id} · {question.difficulty} · {question.category}</p><h2>{question.question}</h2></div><div className="qa-gold"><span>Gold answer</span><strong>{question.gold_answer}</strong><small>{question.answer_format}</small></div></section>
    <div className="qa-layout"><GraphOverlay annotation={annotation} question={question} imageUrl={imageUrl}/><section className="qa-answers"><h2>Condition answers</h2>{dataset.conditions.map(key=><AnswerCard key={key} name={conditionLabels[key]} answer={answers[key]?.[question.question_id]} gold={question.gold_answer}/>)}</section></div>
    <section className="qa-evidence"><h2>Gold supporting evidence</h2><div><strong>Rooms:</strong> {question.supporting_node_ids.length?question.supporting_node_ids.map(id=>annotation.nodes.find(node=>node.id===id)?.label||id).join(', '):'Not required for this question.'}</div><div><strong>Relations:</strong> {question.supporting_edge_ids.length?question.supporting_edge_ids.join(', '):'Not required for this question.'}</div><div><strong>Path:</strong> {question.supporting_path_ids.length?question.supporting_path_ids.map(id=>annotation.nodes.find(node=>node.id===id)?.label||id).join(' → '):'Not required for this question.'}</div></section>
  </main>;
}
