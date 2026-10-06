import React, {useEffect, useRef, useState} from 'react';
import './manual.css';
import {checkReachability, componentKey, componentAccepted} from './reachability.js';
import {SCHEMA, polygonError, contains, withPolygon, exportGraph} from './roomGeometry.js';
const TYPES = ['LivingRoom','Bedroom','Kitchen','Dining','Bath','Storage','Entry','Corridor','Garage','Outdoor','Other'];
const RELATIONS = {connected_by_door:'Door', open_connected:'Open passage', adjacent_to:'Shared wall only', uncertain:'Uncertain'};
const COLOURS = {connected_by_door:'#a21caf',open_connected:'#059669',adjacent_to:'#2563eb',uncertain:'#d97706'};
const EDGE_DASHES = {connected_by_door:undefined,open_connected:'10 6',adjacent_to:'1 6',uncertain:'10 5 1 5'};
const EDGE_STYLE_NAMES = {connected_by_door:'solid',open_connected:'dashed',adjacent_to:'dotted',uncertain:'dash-dot'};
const pairKey = (a,b) => JSON.stringify([a,b].sort());
const blank = () => ({schema_version:SCHEMA,plan_id:'',reviewer:'',status:'draft',provenance:'manual_from_raster',notes:'',image:null,nodes:[],edges:[],all_pairs_reviewed:false});
function validate(d) {
  if(!['floorplan-manual-graph/1',SCHEMA].includes(d?.schema_version) || !Array.isArray(d.nodes) || !Array.isArray(d.edges) || typeof d.plan_id !== 'string' || typeof d.reviewer !== 'string' || typeof d.notes !== 'string') throw Error('Not a supported manual graph JSON.');
  if(!d.image || typeof d.image.name !== 'string' || !Number.isFinite(d.image.width) || !Number.isFinite(d.image.height) || d.image.width<=0 || d.image.height<=0 || !/^[a-f0-9]{64}$/.test(d.image.sha256)) throw Error('Missing image identity.');
  const ids = new Set();
  for(const n of d.nodes) {if(typeof n.id !== 'string' || ids.has(n.id) || !TYPES.includes(n.type) || typeof n.label !== 'string' || !Number.isFinite(n.x) || !Number.isFinite(n.y) || n.x<0 || n.x>1 || n.y<0 || n.y>1) throw Error('Invalid or duplicate room node.'); ids.add(n.id);}
  const pairs=new Set();
  for(const e of d.edges) {const key=pairKey(e.a,e.b); if(!ids.has(e.a)||!ids.has(e.b)||e.a===e.b||!Object.hasOwn(RELATIONS,e.relation)||pairs.has(key)) throw Error('Invalid or duplicate edge.'); pairs.add(key);}
  return exportGraph(d);
}
export default function ManualAnnotator() {
  const [doc,setDoc]=useState(blank),[image,setImage]=useState(null),[selected,setSelected]=useState(null),[mode,setMode]=useState('add'),[kind,setKind]=useState('Bedroom'),[relation,setRelation]=useState('connected_by_door'),[pending,setPending]=useState(null),[message,setMessage]=useState(''),[zoom,setZoom]=useState(1),[history,setHistory]=useState([]);
  const [showReachability,setShowReachability]=useState(false);
  const [selectedEdge,setSelectedEdge]=useState(null);
  const [nodePicker,setNodePicker]=useState(null),[pickerName,setPickerName]=useState('');
  const pickerDialog=useRef();
  useEffect(()=>{if(nodePicker)pickerDialog.current?.showModal();else pickerDialog.current?.close();},[nodePicker]);
  const [outline,setOutline]=useState(null),[outlineTarget,setOutlineTarget]=useState(null),[showBoxes,setShowBoxes]=useState(true);
  const [diskStatus,setDiskStatus]=useState('Not saved — finish annotating, then save.');
  const storage=useRef(null),sourceFile=useRef(null);
  const svg=useRef(),drag=useRef(null),imageRef=useRef(null),docRef=useRef(doc),fileRef=useRef(),[cache,setCache]=useState(false);
  docRef.current=doc;
  useEffect(()=>{try{setCache(!!localStorage.getItem('fyp-manual-draft-v1'));}catch{} return ()=>{if(imageRef.current) URL.revokeObjectURL(imageRef.current);};},[]);
  useEffect(()=>{const warn=e=>{if(docRef.current.image){e.preventDefault();e.returnValue='';}};window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);},[]);
  const [saving,setSaving]=useState(false);
  const saveCompleted=async()=>{
    if(saving||!sourceFile.current)return;
    setSaving(true);setDiskStatus('Saving completed annotation…');
    const snapshot=doc;
    const dfsResult=checkReachability(snapshot.nodes,snapshot.edges,accessStart,snapshot.connectivity_check);
    setShowReachability(true);
    const connectivity_check={algorithm:'DFS',start_node:accessStart,allowed_relations:['connected_by_door','open_connected'],status:dfsResult?.status||'not_run',separate_components:dfsResult?.separate_components||[],visited:dfsResult?.order||[],unreachable:dfsResult?.unreachable||[],checked_at:new Date().toISOString()};
    try{
      const image_base64=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(sourceFile.current);});
      const response=await fetch('/api/local-plan/complete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64,revision:storage.current?.revision||0,annotation:{...exportGraph(snapshot),status:'manually_reviewed',connectivity_check}})});
      const result=await response.json();if(!response.ok)throw Error(result.error||'Save failed.');
      storage.current.revision=result.revision;
      if(docRef.current===snapshot)setDoc({...snapshot,status:'manually_reviewed',connectivity_check});
      setDiskStatus(`Saved completed image and graph: ${result.annotation_path}`);
    }catch(e){setDiskStatus(`Not saved: ${e.message}`);}finally{setSaving(false);}
  };
  const change = fn => {setHistory(h=>[...h.slice(-49),doc]);setDoc(d=>({...fn(d),status:'draft',all_pairs_reviewed:false}));};
  const replace = d => {setShowReachability(false);setSelectedEdge(null);setNodePicker(null);storage.current=null;sourceFile.current=null;setDiskStatus('Not saved — finish annotating, then save.');setOutline(null);setOutlineTarget(null);setHistory([]);setDoc(d);setSelected(null);setPending(null);setMode('add');setZoom(1);if(imageRef.current)URL.revokeObjectURL(imageRef.current);imageRef.current=null;setImage(null);};
  const loadImage=async f=>{if(!f)return;try{
    const bytes=await f.arrayBuffer();const sha256=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
    if(doc.image && doc.image.sha256!==sha256)throw Error('This image differs from the current annotation. Download your JSON, then choose New plan first.');
    const url=URL.createObjectURL(f),im=new Image(); await new Promise((resolve,reject)=>{im.onload=resolve;im.onerror=()=>reject(Error('Cannot read this image.'));im.src=url;});
    if(imageRef.current)URL.revokeObjectURL(imageRef.current);imageRef.current=url;setImage(url);
    const response=await fetch('/api/local-plan/lookup?name='+encodeURIComponent(f.name),{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:bytes});
    const saved=await response.json();if(!response.ok)throw Error(saved.error||'Could not store image.');
    storage.current={...saved,blocked:false};sourceFile.current=f;
    const restored=saved.annotation?validate(saved.annotation):null;
    if(restored&&doc.image&&(doc.nodes.length||doc.edges.length)){if(!confirm('Keep your current draft? Cancel loads the stored version. Nothing will be saved until you finish.')){setDoc(restored);setMessage('Stored annotation reopened.');return;}setDoc(d=>({...d,image:{...d.image,local_path:saved.image_path}}));setMessage('Current draft kept. Save when finished.');return;}
    setDoc(d=>restored||({...d,plan_id:saved.source_dir?saved.plan_id:(d.plan_id||saved.plan_id),source_dir:saved.source_dir,image:{name:f.name,width:im.naturalWidth,height:im.naturalHeight,sha256,local_path:saved.image_path}}));
    setMessage(restored?'Saved annotation reopened with its image.':'Image opened. Nothing is saved until you click Save completed annotation.');
  }catch(e){setMessage(e.message);} };
  const point=e=>{const p=svg.current.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const q=p.matrixTransform(svg.current.getScreenCTM().inverse());return {x:Math.max(0,Math.min(1,q.x/doc.image.width)),y:Math.max(0,Math.min(1,q.y/doc.image.height))};};
  const nodeClick=(e,n)=>{e.stopPropagation();setSelectedEdge(null);setSelected(n.id);if(mode==='add'){openNodePicker({id:n.id});return;}if(mode==='connect'){if(!pending){setPending(n.id);return;}if(pending===n.id){setPending(null);return;}change(d=>({...d,edges:[...d.edges.filter(x=>pairKey(x.a,x.b)!==pairKey(pending,n.id)),{a:pending,b:n.id,relation}]}));setSelectedEdge(pairKey(pending,n.id));setPending(null);}};
  const save=()=>{const data={...exportGraph(doc),updated_at:new Date().toISOString()};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`${doc.plan_id.replace(/[^a-zA-Z0-9_-]/g,'_')||'plan'}.graph.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);setMessage('JSON downloaded. Keep it alongside the original image.');};
  const startOutline = target => {setOutline([]);setOutlineTarget(target);setMode('outline');setPending(null);setMessage('Click each room corner in order, then Finish outline. The last edge closes automatically.');};
  const cancelOutline = () => {setOutline(null);setOutlineTarget(null);setMode('move');};
  const finishOutline = () => {const error=polygonError(outline);if(error){setMessage(error);return;}
    const id=outlineTarget||crypto.randomUUID();
    change(d=>{const base=d.nodes.find(n=>n.id===id)||{id,type:kind,label:`${kind} ${d.nodes.filter(n=>n.type===kind).length+1}`,x:-1,y:-1};const room=withPolygon(base,outline);return {...d,nodes:outlineTarget?d.nodes.map(n=>n.id===id?room:n):[...d.nodes,room]};});
    setSelected(null);cancelOutline();setMode('add');setSelectedEdge(null);setMessage('Outline complete. Click inside the next room to choose its type and trace it.');
  };
  useEffect(()=>{
    if(outline===null||nodePicker)return;
    const onKeyDown=event=>{
      if(event.key!=='Enter'||event.repeat||event.isComposing||event.ctrlKey||event.metaKey||event.altKey)return;
      if(event.target instanceof Element && event.target.closest('input,textarea,select,button,[contenteditable="true"],dialog'))return;
      event.preventDefault();finishOutline();
    };
    window.addEventListener('keydown',onKeyDown);
    return ()=>window.removeEventListener('keydown',onKeyDown);
  },[outline,outlineTarget,doc,nodePicker]);
  const editVertex = (index,remove=false) => {if(!active?.polygon)return;const points=[...active.polygon];if(remove)points.splice(index,1);else {const a=points[index],b=points[(index+1)%points.length];points.splice(index+1,0,[(a[0]+b[0])/2,(a[1]+b[1])/2]);}try{const room=withPolygon(active,points);change(d=>({...d,nodes:d.nodes.map(n=>n.id===active.id?room:n)}));}catch(e){setMessage(e.message);}};
  const openNodePicker = value => {setPickerName(value.id ? doc.nodes.find(n=>n.id===value.id)?.label||'' : '');setNodePicker(value);};
  const chooseNodeType = type => {
    if(!nodePicker)return;
    const id=nodePicker.id||crypto.randomUUID();
    change(d=>{
      const existing=d.nodes.find(n=>n.id===id);
      const names=new Set(d.nodes.filter(n=>n.id!==id).map(n=>n.label));let number=1;while(names.has(`${type} ${number}`))number++;
      const oldAutomatic=existing && pickerName===existing.label && existing.label.startsWith(existing.type+' ') && /^\d+$/.test(existing.label.slice(existing.type.length+1));
      const label=pickerName.trim()&&!oldAutomatic?pickerName.trim():`${type} ${number}`;
      return {...d,nodes:existing?d.nodes.map(n=>n.id===id?{...n,type,label}:n):[...d.nodes,{id,type,label,x:nodePicker.x,y:nodePicker.y}]};
    });
    setSelected(id);setKind(type);setNodePicker(null);
    if(!doc.nodes.find(n=>n.id===id)?.polygon)startOutline(id);
    else setMessage('Room type updated. Its existing outline is preserved.');
  };
  const deleteSelection=()=>{
    if(saving||outline!==null||nodePicker)return;
    if(selectedEdge&&doc.edges.some(e=>pairKey(e.a,e.b)===selectedEdge)){
      change(d=>({...d,edges:d.edges.filter(e=>pairKey(e.a,e.b)!==selectedEdge)}));
      setSelectedEdge(null);setMessage('Connection deleted. Use Undo to restore it.');
    }else if(selected){
      change(d=>({...d,nodes:d.nodes.filter(n=>n.id!==selected),edges:d.edges.filter(e=>e.a!==selected&&e.b!==selected)}));
      setSelected(null);setMessage('Room and its connections deleted. Use Undo to restore them.');
    }
    setPending(null);
  };
  useEffect(()=>{
    const keydown=event=>{
      if(!['Delete','Backspace'].includes(event.key)||event.repeat||event.isComposing||event.metaKey||event.ctrlKey||event.altKey)return;
      if(event.target instanceof Element&&event.target.closest('input,textarea,select,button,[contenteditable],dialog'))return;
      if(outline!==null||nodePicker||saving||(!selected&&!selectedEdge))return;
      event.preventDefault();deleteSelection();
    };
    window.addEventListener('keydown',keydown);return()=>window.removeEventListener('keydown',keydown);
  },[doc,selected,selectedEdge,outline,nodePicker,saving]);
  const saveRequirements = [
    !image && 'Open the original floor-plan image.',
    !doc.plan_id.trim() && 'Enter a plan ID.',
    !doc.reviewer.trim() && 'Enter your name or ID in the Annotator field above.',
    !doc.nodes.length && 'Add at least one room.',
    doc.nodes.some(n=>!n.label.trim()) && 'Give every room a label.',
    doc.nodes.some(n=>!n.polygon) && 'Finish the outline for every room.',
    outline!==null && 'Finish or cancel the current outline.',
  ].filter(Boolean);
  const savedSuccessfully = !saving && doc.status==='manually_reviewed' && diskStatus.startsWith('Saved completed');
  const active=doc.nodes.find(n=>n.id===selected), label=id=>doc.nodes.find(n=>n.id===id)?.label||id;
  const accessStart=doc.nodes.some(n=>n.id===doc.connectivity_check?.start_node)?doc.connectivity_check.start_node:(doc.nodes.find(n=>n.type==='Entry')?.id||doc.nodes[0]?.id||'');
  const accessCheck=showReachability?checkReachability(doc.nodes,doc.edges,accessStart,doc.connectivity_check):null;
  // Drop obsolete approvals when a component splits, merges, disappears, or entry changes.
  useEffect(()=>{
    if(!doc.connectivity_check)return;
    const result=checkReachability(doc.nodes,doc.edges,accessStart,doc.connectivity_check);
    const components=result?.separate_components||[];
    if(JSON.stringify(components)!==JSON.stringify(doc.connectivity_check.separate_components||[]))setDoc(d=>({...d,connectivity_check:{...d.connectivity_check,start_node:accessStart,separate_components:components,status:result?.status||'not_run'}}));
  },[doc.nodes,doc.edges,accessStart,doc.connectivity_check]);
  const reviewComponent=(roomIds,reason)=>{
    const result=checkReachability(doc.nodes,doc.edges,accessStart,doc.connectivity_check);
    const separate_components=result.separate_components.map(c=>componentKey(c.room_ids)===componentKey(roomIds)?{...c,valid_separate_access:reason==='separate_access',valid_separate_level:reason==='separate_level'}:c);
    setDoc(d=>({...d,status:'draft',connectivity_check:{...d.connectivity_check,start_node:accessStart,separate_components,status:separate_components.some(c=>!componentAccepted(c))?'review':'pass'}}));
  };
  return <main className="manual"><dialog ref={pickerDialog} className="node-picker" aria-labelledby="node-picker-title" onCancel={()=>setNodePicker(null)} ><div className="picker-heading"><div><p className="eyebrow">Room node</p><h2 id="node-picker-title">{nodePicker?.id?'Change this room':'What room is this?'}</h2></div><button aria-label="Close room picker" onClick={()=>setNodePicker(null)}>✕</button></div><p>Choose a room type, then trace its corners on the image. Existing outlines are kept when relabelling.</p><label>Optional room name<input value={pickerName} placeholder="Auto-name from room type" onChange={e=>setPickerName(e.target.value)}/></label><div className="node-type-grid">{TYPES.map(type=><button key={type} onClick={()=>chooseNodeType(type)}><strong>{({LivingRoom:'Living room',Bath:'Bathroom',Entry:'Entrance',Other:'Other / sauna'})[type]||type}</strong></button>)}</div><button className="picker-cancel" onClick={()=>setNodePicker(null)}>Cancel</button></dialog><header><div><p className="eyebrow">CubiCasa · 25-plan benchmark</p><h1>Floor-plan annotator</h1><p className="muted">1. Choose room type → 2. Trace its outline → 3. Connect rooms</p></div><div className="manual-actions"><a href="?mode=review">SVG reviewer ↗</a><button disabled={saving} onClick={()=>{if(doc.image&&!confirm('Start a new plan? Unsaved edits will be lost.'))return;replace(blank());setMessage('Choose your next image.');}}>New plan</button><button className="primary" disabled={!doc.image||outline!==null} onClick={save}>Download JSON</button></div></header>
  <details className="file-panel" open={!image}><summary>{image?`Current image: ${doc.image?.name||'floor plan'} · Open / restore files`:'Open a floor plan'}</summary><section className="uploads"><label>Floor-plan image<input ref={fileRef} type="file" accept="image/png,image/jpeg" onChange={e=>{loadImage(e.target.files[0]);e.target.value='';}}/></label><label>Reopen saved JSON<input type="file" accept=".json" onChange={async e=>{const f=e.target.files[0];e.target.value='';if(!f)return;try{const d=validate(JSON.parse(await f.text()));if(doc.image&&!confirm('Replace current annotation? Download its JSON first.'))return;replace(d);setMessage('Annotation restored. Select its original image to continue.');}catch(err){setMessage(err.message);}}}/></label><button disabled={!cache} onClick={()=>{try{const d=validate(JSON.parse(localStorage.getItem('fyp-manual-draft-v1')));replace(d);setMessage('Browser draft restored. Select its original image.');}catch(e){setMessage(e.message);}}}>Restore browser draft</button></section></details>
  <div className="manual-meta"><label>Unique plan ID<input value={doc.plan_id} placeholder="high_quality/1234" onChange={e=>setDoc({...doc,plan_id:e.target.value,status:'draft'})}/></label><label>Annotator<input value={doc.reviewer} placeholder="Your name or ID" onChange={e=>setDoc({...doc,reviewer:e.target.value,status:'draft'})}/></label><span>{doc.nodes.length} rooms · {doc.edges.length} edges · {doc.status}</span></div>
  <p role="status" className="manual-message">{message||'Nothing saves automatically. Complete the annotation, then save the image and graph together.'}</p>
  <p role="status" className={`manual-status ${diskStatus.startsWith('Not saved')?'save-error':''}`} title={diskStatus}>{doc.status==='draft'&&diskStatus.startsWith('Saved completed')?'Unsaved changes — save again when finished.':diskStatus} {outline!==null&&'Finish this outline before saving.'}</p>
  <div className="manual-tools">{[['add','＋ Add room'],['move','Move / edit'],['connect','Connect rooms']].map(([id,title])=><button key={id} disabled={outline!==null} aria-pressed={mode===id} onClick={()=>{setMode(id);setPending(null);setSelectedEdge(null);}}>{title}</button>)}<button disabled={!image||!active||outline!==null} onClick={()=>startOutline(active.id)}>Trace selected room</button>{mode==='connect'&&<label>{selectedEdge?'Selected edge type':'New edge type'} <select value={doc.edges.find(e=>pairKey(e.a,e.b)===selectedEdge)?.relation||relation} onChange={e=>{const value=e.target.value;setRelation(value);if(selectedEdge)change(d=>({...d,edges:d.edges.map(edge=>pairKey(edge.a,edge.b)===selectedEdge?{...edge,relation:value}:edge)}));}}>{Object.entries(RELATIONS).map(([id,name])=><option key={id} value={id}>{name}</option>)}</select></label>}<button className="danger-action" disabled={saving||outline!==null||(!selected&&!selectedEdge)} onClick={deleteSelection}>{selectedEdge?'Delete selected edge':'Delete selected room'}</button><button disabled={!history.length||outline!==null} onClick={()=>{setDoc(history.at(-1));setHistory(h=>h.slice(0,-1));setSelected(null);setPending(null);}}>Undo</button><label className="zoom-label">Zoom <input type="range" min="1" max="3" step=".1" value={zoom} onChange={e=>setZoom(Number(e.target.value))}/></label><label><input type="checkbox" checked={showBoxes} disabled={mode==='connect'} onChange={e=>setShowBoxes(e.target.checked)}/>{mode==='connect'?'Room outlines hidden while connecting':'Show derived boxes'}</label></div>
  {outline!==null&&<div className="outline-tools"><b>{outlineTarget?`Outline for ${label(outlineTarget)}`:"New room outline"} · {outline.length} corners</b><button disabled={outline.length<3} onClick={finishOutline}>Finish outline ↵</button><button disabled={!outline.length} onClick={()=>setOutline(p=>p.slice(0,-1))}>Undo corner</button><button onClick={cancelOutline}>Cancel outline</button><span>Click corners in order, then press Enter to finish and add the next room.</span></div>}
  <p className="help mode-help">{mode==='outline'?'Click around the inside boundary of the room; do not repeat the first corner.':mode==='connect'?(pending?`Now select the room connected to ${label(pending)}.`:'Click two nodes to add an edge, then change its type if needed. The newest edge stays selected.') :mode==='move'?'Drag a node within its room or drag the selected outline’s white corner handles.':'Click inside a room, choose its type, then trace its corners. Finish the outline before adding another room.'}</p>
  <div className="edge-editor-slot">{selectedEdge&&doc.edges.some(e=>pairKey(e.a,e.b)===selectedEdge)?<div className="selected-edge-bar"><strong>{(()=>{const edge=doc.edges.find(e=>pairKey(e.a,e.b)===selectedEdge);return `${label(edge.a)} ↔ ${label(edge.b)}`;})()}</strong><span>Choose a type to update this edge:</span>{Object.entries(RELATIONS).map(([type,name])=><button key={type} style={{borderColor:COLOURS[type]}} onClick={()=>{setRelation(type);change(d=>({...d,edges:d.edges.map(e=>pairKey(e.a,e.b)===selectedEdge?{...e,relation:type}:e)}));}}>{name}</button>)}<button className="danger-action" onClick={deleteSelection}>Delete edge</button><button onClick={()=>setSelectedEdge(null)}>Done</button></div>:<div className="edge-editor-placeholder">Select an edge to change its type, or connect two rooms.</div>}</div>
  <div className="edge-legend" aria-label="Connection line styles">{Object.entries(RELATIONS).map(([type,name])=><span key={type}><svg width="42" height="14" aria-hidden="true"><line x1="3" y1="7" x2="39" y2="7" stroke={COLOURS[type]} strokeWidth="3" strokeDasharray={EDGE_DASHES[type]} strokeLinecap="round"/></svg>{name}<small>{EDGE_STYLE_NAMES[type]}</small></span>)}</div>
  <section className="workspace manual-workspace"><div className="manual-canvas">{image&&doc.image?<svg ref={svg} aria-label="Floor-plan graph canvas" viewBox={`0 0 ${doc.image.width} ${doc.image.height}`} style={{width:`${zoom*100}%`,maxWidth:zoom===1?`calc((100vh - 290px) * ${doc.image.width/doc.image.height})`:'none',touchAction:'none'}} onClick={e=>{if(e.target!==e.currentTarget)return;if(mode==='outline'){const p=point(e);setOutline(points=>[...points,[p.x,p.y]]);return;}if(mode!=='add')return;openNodePicker(point(e));}}
    onPointerMove={e=>{if(!drag.current)return;const p=point(e),action=drag.current;setDoc(d=>({...d,status:'draft',all_pairs_reviewed:false,nodes:d.nodes.map(n=>{if(n.id!==action.id)return n;if(action.vertex===undefined)return !n.polygon||contains(n.polygon,[p.x,p.y])?{...n,...p}:n;const points=n.polygon.map((v,i)=>i===action.vertex?[p.x,p.y]:v);return polygonError(points)?n:withPolygon(n,points);})}));}}
    onPointerUp={()=>{drag.current=null;}} onPointerCancel={()=>{drag.current=null;}}>
    <rect width={doc.image.width} height={doc.image.height} fill="transparent" style={{pointerEvents:'none'}}/>
    <image href={image} width={doc.image.width} height={doc.image.height} style={{pointerEvents:'none'}}/>
    {mode!=='connect'&&doc.nodes.filter(n=>n.polygon).map(n=>{const [x0,y0,x1,y1]=n.bbox_xyxy;return <g key={`shape-${n.id}`} style={{pointerEvents:'none'}}><polygon points={n.polygon.map(p=>`${p[0]*doc.image.width},${p[1]*doc.image.height}`).join(' ')} fill={n.id===selected?'#f9731622':'#2563eb12'} stroke={n.id===selected?'#ea580c':'#2563eb'} strokeWidth="2" vectorEffect="non-scaling-stroke"/>{showBoxes&&<rect x={x0*doc.image.width} y={y0*doc.image.height} width={(x1-x0)*doc.image.width} height={(y1-y0)*doc.image.height} fill="none" stroke="#64748b" strokeDasharray="6 5" strokeWidth="1.5" vectorEffect="non-scaling-stroke"/>}</g>;})}
    {outline!==null&&<g style={{pointerEvents:'none'}}><polyline points={outline.map(p=>`${p[0]*doc.image.width},${p[1]*doc.image.height}`).join(' ')} fill="#f59e0b22" stroke="#ea580c" strokeWidth="2" vectorEffect="non-scaling-stroke"/>{outline.map((p,i)=><circle key={i} cx={p[0]*doc.image.width} cy={p[1]*doc.image.height} r={doc.image.width*.005} fill="#ea580c"/>)}</g>}
    {doc.edges.map(edge=>{const a=doc.nodes.find(n=>n.id===edge.a),b=doc.nodes.find(n=>n.id===edge.b),key=pairKey(edge.a,edge.b),coords={x1:a.x*doc.image.width,y1:a.y*doc.image.height,x2:b.x*doc.image.width,y2:b.y*doc.image.height};const select=()=>{setSelectedEdge(key);setMode('connect');setPending(null);};return <g key={key} role="button" tabIndex={mode==='outline'?-1:0} aria-label={`Edit ${RELATIONS[edge.relation]} edge: ${label(edge.a)} to ${label(edge.b)}`} onClick={e=>{e.stopPropagation();if(mode!=='outline')select();}} onKeyDown={e=>{if(mode!=='outline'&&(e.key==='Enter'||e.key===' ')){e.preventDefault();select();}}} style={{pointerEvents:mode==='outline'?'none':'auto',cursor:'pointer'}}>
      <title>{label(edge.a)} ↔ {label(edge.b)}: {RELATIONS[edge.relation]} — click to change</title>
      <line {...coords} stroke={selectedEdge===key?'#fbbf2466':'transparent'} strokeWidth="14" vectorEffect="non-scaling-stroke" style={{pointerEvents:mode==='outline'?'none':'stroke'}}/>
      <line {...coords} stroke={COLOURS[edge.relation]} strokeWidth="3" vectorEffect="non-scaling-stroke" strokeDasharray={EDGE_DASHES[edge.relation]} strokeLinecap="round" style={{pointerEvents:'none'}}/>
    </g>;})}
    {doc.nodes.map(n=><g key={n.id} transform={`translate(${n.x*doc.image.width},${n.y*doc.image.height})`} onDoubleClick={e=>{e.stopPropagation();if(mode!=='connect')openNodePicker({id:n.id});}} onClick={e=>nodeClick(e,n)} onPointerDown={e=>{e.stopPropagation();if(mode!=='move')return;setSelected(n.id);setHistory(h=>[...h.slice(-49),doc]);drag.current={id:n.id};svg.current.setPointerCapture(e.pointerId);}} style={{pointerEvents:mode==='outline'?'none':'auto',cursor:mode==='move'?'grab':'pointer'}}>{accessCheck&&<circle r={doc.image.width*.017} fill="none" stroke={accessCheck.unreachable.includes(n.id)?(componentAccepted(accessCheck.separate_components.find(c=>c.room_ids.includes(n.id))||{})?'#0891b2':'#d97706'):'#16a34a'} strokeWidth="3" vectorEffect="non-scaling-stroke" style={{pointerEvents:'none'}}/>}<circle r={doc.image.width*.011} fill={pending===n.id?'#d97706':selected===n.id?'#ea580c':'#172554'} stroke="white" strokeWidth="2" vectorEffect="non-scaling-stroke"/><text y={-doc.image.width*.018} textAnchor="middle" fontSize={doc.image.width*.015} paintOrder="stroke" stroke="white" strokeWidth={doc.image.width*.004} fill="#172554">{n.label}</text></g>)}
    {mode==='move'&&active?.polygon?.map((p,i)=><circle key={`handle-${i}`} cx={p[0]*doc.image.width} cy={p[1]*doc.image.height} r={doc.image.width*.006} fill="white" stroke="#ea580c" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{cursor:'grab'}} onPointerDown={e=>{e.stopPropagation();setHistory(h=>[...h.slice(-49),doc]);drag.current={id:active.id,vertex:i};svg.current.setPointerCapture(e.pointerId);}}><title>Corner {i+1}: drag to move</title></circle>)}
    </svg>:<div className="empty">{doc.image?'Select the original image to resume your saved graph.':'Upload a PNG or JPG to begin a blank graph.'}</div>}</div>
    <aside>
  <section className="reachability-panel" aria-label="DFS connectivity test"><div className="dfs-heading"><strong>Connectivity test</strong><span className={`dfs-badge ${!accessCheck?'not-run':accessCheck.status}`} role="status">{!accessCheck?'RUNS ON SUBMIT':accessCheck.status==='review'?'REVIEW — separate connectivity components detected':'PASS — connectivity reviewed'}</span></div>
  <div className="reachability-controls"><label>Start from <select value={accessStart} disabled={!doc.nodes.length||saving} onChange={e=>{const start_node=e.target.value;setDoc(d=>({...d,status:'draft',connectivity_check:{start_node,separate_components:[]}}));}}>{doc.nodes.map(n=><option key={n.id} value={n.id}>{n.label}</option>)}</select></label><button disabled={!doc.nodes.length} onClick={()=>setShowReachability(true)}>{showReachability?'Run DFS test again':'Run DFS test'}</button>{showReachability&&<button onClick={()=>setShowReachability(false)}>Reset test</button>}</div>
  {accessCheck&&<><div role="status" className={accessCheck.status==='review'?'access-warning':'access-pass'}><strong>Main-entry component</strong><p>{accessCheck.order.length}/{accessCheck.total} rooms reachable from {label(accessCheck.start)}</p>{accessCheck.status==='pass'&&accessCheck.separate_components.length>0&&<p>{accessCheck.unreachable.length} rooms in explicitly reviewed separate components. This does not establish a traversable route between components.</p>}<details><summary>DFS visit order</summary><p>{accessCheck.order.map(label).join(' → ')}</p><small>Traversal order, not a route between consecutive rooms.</small></details></div>
  {accessCheck.separate_components.map((component,index)=><div className="separate-component" key={componentKey(component.room_ids)}><strong>Separate component {index+1}</strong><div className="unreachable-rooms">{component.room_ids.map(id=><button key={id} disabled={outline!==null} onClick={()=>{setSelected(id);setSelectedEdge(null);setPending(null);setMode('move');}}>{label(id)}</button>)}</div><label>Review this component<select disabled={saving} value={component.valid_separate_level?'separate_level':component.valid_separate_access?'separate_access':''} onChange={e=>reviewComponent(component.room_ids,e.target.value)}><option value="">Needs review</option><option value="separate_access">Valid separate-access area</option><option value="separate_level">Valid separate floor / level</option></select></label>{component.valid_separate_level&&<p>Confirmed as another level. Stair connectivity is not represented by this check; no door or open-passage edge is invented between floors.</p>}</div>)}
  </>}
  <p className="help">Runs on submit using doors and open passages only. Amber rings mark unreviewed separate areas; teal rings mark accepted separate areas or levels. Confirm the reason from the image; disconnected components alone do not prove multiple levels. Results update live; reset hides the test without erasing decisions. Changing entry or component membership requires fresh review. This check does not block saving.</p></section>
<details className="inspector-section"><summary>Rooms ({doc.nodes.length})</summary><p className="help">{doc.nodes.filter(n=>n.polygon).length}/{doc.nodes.length} room outlines complete. Every room needs an outline before marking reviewed.</p><div className="manual-room-list">{doc.nodes.map(n=><button key={n.id} disabled={outline!==null} aria-pressed={selected===n.id} onClick={()=>{setSelected(n.id);setSelectedEdge(null);setMode('move');setPending(null);}}>{n.label}</button>)}</div>{active&&<fieldset disabled={outline!==null} className="manual-editor"><button onClick={()=>openNodePicker({id:active.id})}>Choose room type…</button><label>Room label<input value={active.label} onChange={e=>change(d=>({...d,nodes:d.nodes.map(n=>n.id===selected?{...n,label:e.target.value}:n)}))}/></label><label>Room type<select value={active.type} onChange={e=>change(d=>({...d,nodes:d.nodes.map(n=>n.id===selected?{...n,type:e.target.value}:n)}))}>{TYPES.map(t=><option key={t}>{t}</option>)}</select></label><button onClick={()=>{change(d=>({...d,nodes:d.nodes.filter(n=>n.id!==selected),edges:d.edges.filter(e=>e.a!==selected&&e.b!==selected)}));setSelected(null);setPending(null);}}>Delete room + its edges</button><button disabled={!image} onClick={()=>startOutline(active.id)}>{active.polygon?'Redraw outline':'Draw outline for selected room'}</button>{active.polygon?<><p className="help">{active.polygon.length} corners · box [x₀, y₀, x₁, y₁]: {active.bbox_xyxy.map(v=>v.toFixed(3)).join(', ')}. Box and outline use normalized image coordinates.</p><details><summary>Edit outline corners</summary>{active.polygon.map((p,i)=><div className="corner-row" key={i}><span>Corner {i+1}</span><button onClick={()=>editVertex(i)}>Insert after</button><button disabled={active.polygon.length<=3} onClick={()=>editVertex(i,true)}>Remove</button></div>)}</details><button onClick={()=>change(d=>({...d,nodes:d.nodes.map(n=>n.id===active.id?{...n,polygon:null,bbox_xyxy:null}:n)}))}>Remove outline, keep node</button></>:<p className="help">No outline yet. Add one to include this room in box-based evaluation.</p>}</fieldset>}
    </details><details className="inspector-section"><summary>Connections ({doc.edges.length})</summary><p className="help">Door and open passage mean direct access. Shared wall only means no direct access. Use uncertain when the image is unclear.</p><div className="manual-edges">{doc.edges.map(e=><div key={pairKey(e.a,e.b)}><span>{label(e.a)} ↔ {label(e.b)}</span><select aria-label={`Relation ${label(e.a)} to ${label(e.b)}`} value={e.relation} onChange={ev=>change(d=>({...d,edges:d.edges.map(x=>x===e?{...x,relation:ev.target.value}:x)}))}>{Object.entries(RELATIONS).map(([id,name])=><option key={id} value={id}>{name}</option>)}</select><button aria-label={`Delete edge ${label(e.a)} to ${label(e.b)}`} onClick={()=>change(d=>({...d,edges:d.edges.filter(x=>x!==e)}))}>Remove</button></div>)}</div>
    </details><details className="inspector-section"><summary>Notes / ambiguities</summary><label className="notes">Notes / ambiguities<textarea value={doc.notes} onChange={e=>setDoc({...doc,notes:e.target.value,status:'draft'})}/></label></details><div className="completion-panel" id="completion-panel"><h2>Finish annotation</h2>{saveRequirements.length>0?<div role="status"><p>Before saving:</p><ul>{saveRequirements.map(item=><li key={item}>{item}</li>)}</ul>{!doc.reviewer.trim()&&<label>Annotator name or ID<input value={doc.reviewer} placeholder="Enter your name or ID" onChange={e=>setDoc({...doc,reviewer:e.target.value,status:'draft'})}/></label>}</div>:<p>Ready to save the image, graph and exclusion folder.</p>}<div className="save-button-stack"><button className="primary" disabled={saving||saveRequirements.length>0} onClick={saveCompleted}>{saving?'Saving…':'Confirm & save annotation'}</button>{savedSuccessfully&&<p className="save-success" role="status">✓ Saved successfully</p>}</div></div></aside></section>
  <footer className="save-dock">
    <div><strong>{saving?'Saving…':savedSuccessfully?'Annotation complete':'Finish & save'}</strong><span>{savedSuccessfully?'Image, graph and exclusion entry saved in cubicasa_eval.':saveRequirements.length?saveRequirements[0]:'Ready to save your completed annotation.'}</span></div>
    {saveRequirements.length>0&&<button onClick={()=>document.getElementById('completion-panel')?.scrollIntoView({behavior:'smooth',block:'center'})}>Review requirements ({saveRequirements.length})</button>}
    <div className="save-button-stack"><button className="primary" disabled={saving||saveRequirements.length>0} onClick={saveCompleted}>{saving?'Saving…':'Confirm & save annotation'}</button>{savedSuccessfully&&<p className="save-success" role="status">✓ Saved successfully</p>}</div>
  </footer>
  </main>;
}
