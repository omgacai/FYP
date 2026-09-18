import React, {useMemo, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";
import "./handles.css";
import ManualAnnotator from "./ManualAnnotator.jsx";

const TYPES = ["LivingRoom", "Bedroom", "Kitchen", "Dining", "Bath", "Storage", "Entry", "Garage", "Other", "Outdoor"];
// Colour identifies a room *instance*, not its predicted type. Most CubiGraph
// SVGs initially call several rooms "Other", so type-colouring made them grey.
const COLOURS = ["#e11d48", "#2563eb", "#d97706", "#7c3aed", "#059669", "#db2777", "#0891b2", "#ea580c", "#9333ea", "#16a34a"];
const roomColour = room => COLOURS[[...room.id].reduce((total, char) => total + char.charCodeAt(0), 0) % COLOURS.length];
const TYPE_COLOURS = {LivingRoom:"#2563eb", Bedroom:"#7c3aed", Kitchen:"#ea580c", Dining:"#0891b2", Bath:"#db2777", Storage:"#a16207", Entry:"#0f766e", Garage:"#4f46e5", Other:"#64748b", Outdoor:"#16a34a"};
const typeColour = type => TYPE_COLOURS[type] || "#64748b";
const EDGE_PASSES = [
  {id:"door", predicate:"connected_by_door", title:"1. Door connections", short:"door connection", help:"Add only pairs separated by a real, traversable drawn door."},
  {id:"open", predicate:"open_connected", title:"2. Open connections", short:"open connection", help:"Add only distinct room zones with a direct, unobstructed opening and no door."},
  {id:"adjacent", predicate:"adjacent_to", title:"3. Adjacency", short:"adjacent", help:"Add shared-boundary pairs only when there is neither a door nor an open connection."}
];
const PREDICATE_IDS = {adjacent_to:1, connected_by_door:2, open_connected:3};
const edgePassFor = predicate => EDGE_PASSES.find(pass => pass.predicate === predicate) || EDGE_PASSES[2];
const edgePairKey = ({a, b}) => [a, b].sort().join("|");
const uniqueEdges = edges => [...new Map(edges.map(edge => [edgePairKey(edge), {...edge, a:[edge.a, edge.b].sort()[0], b:[edge.a, edge.b].sort()[1]}])).values()];
const edgeRoute = (edge, first, second) => {
  const [x1, y1] = centre(first), [x2, y2] = centre(second);
  const dx = x2 - x1, dy = y2 - y1, length = Math.hypot(dx, dy) || 1;
  const hash = [...edgePairKey(edge)].reduce((total, char) => total + char.charCodeAt(0), 0);
  const bend = ((hash % 5) - 2) * 18;
  const cx = (x1 + x2) / 2 - (dy / length) * bend, cy = (y1 + y2) / 2 + (dx / length) * bend;
  return {d:`M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`, x:(x1 + 2 * cx + x2) / 4, y:(y1 + 2 * cy + y2) / 4};
};
const pointList = text => text.trim().split(/\s+/).map(p => p.split(",").map(Number));
const bounds = points => [Math.min(...points.map(p => p[0])), Math.min(...points.map(p => p[1])), Math.max(...points.map(p => p[0])), Math.max(...points.map(p => p[1]))];
const centre = room => [(room.box[0] + room.box[2]) / 2, (room.box[1] + room.box[3]) / 2];
// Stable source IDs keep graph endpoints reliable; reviewer-facing labels use
// the corrected semantic type and a per-type instance number instead.
const roomLabel = (room, rooms) => `${room.type}_${rooms.filter(candidate => candidate.type === room.type).indexOf(room) + 1}`;
const distanceToSegment = (point, start, end) => {
  const dx = end[0] - start[0], dy = end[1] - start[1];
  const t = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy || 1)));
  return (point[0] - (start[0] + t * dx)) ** 2 + (point[1] - (start[1] + t * dy)) ** 2;
};

function parseSvg(text) {
  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  const svg = doc.documentElement;
  const width = Number(svg.getAttribute("width")) || 1;
  const height = Number(svg.getAttribute("height")) || 1;
  const polygons = [...svg.children].filter(e => e.localName === "polygon" && e.getAttribute("stroke") !== "black");
  const circles = [...svg.children].filter(e => e.localName === "circle");
  const labels = [...svg.children].filter(e => e.localName === "text").map(e => e.textContent.trim());
  const rooms = polygons.map((polygon, index) => {
    const id = labels[index] || `Room_${index + 1}`;
    const type = TYPES.includes(id.replace(/_\d+$/, "")) ? id.replace(/_\d+$/, "") : "Other";
    const points = pointList(polygon.getAttribute("points"));
    return {id, type, points, box: bounds(points), sourcePoints: points, sourceBox: bounds(points)};
  });
  const dots = circles.map((circle, index) => ({id: rooms[index]?.id, x: Number(circle.getAttribute("cx")), y: Number(circle.getAttribute("cy"))}));
  const closest = (x, y) => dots.reduce((best, dot) => !best || (dot.x-x)**2+(dot.y-y)**2 < (best.x-x)**2+(best.y-y)**2 ? dot : best, null)?.id;
  const edges = uniqueEdges([...svg.children].filter(e => e.localName === "line").map(line => ({
    a: closest(Number(line.getAttribute("x1")), Number(line.getAttribute("y1"))),
    b: closest(Number(line.getAttribute("x2")), Number(line.getAttribute("y2"))),
    predicate: line.hasAttribute("stroke-dasharray") ? "connected_by_door" : "adjacent_to"
  })).filter(e => e.a && e.b && e.a !== e.b));
  return {width, height, rooms, edges};
}

function EditHandles({room, onDrag, onRemoveVertex}) {
  const [x0, y0, x1, y1] = room.box;
  const boxHandles = [
    [x0, y0, "left top"], [(x0 + x1) / 2, y0, "top"], [x1, y0, "right top"],
    [x0, (y0 + y1) / 2, "left"], [x1, (y0 + y1) / 2, "right"],
    [x0, y1, "left bottom"], [(x0 + x1) / 2, y1, "bottom"], [x1, y1, "right bottom"]
  ];
  return <g className="edit-handles">
    <rect x={x0} y={y0} width={x1-x0} height={y1-y0}/>
    {boxHandles.map(([x, y, kind]) => <rect key={kind} x={x-7} y={y-7} width="14" height="14" onPointerDown={event=>onDrag(event, kind)}/>) }
    {room.points.map((point, index) => <circle key={index} cx={point[0]} cy={point[1]} r="7" onPointerDown={event=>onDrag(event, "vertex", index)} onDoubleClick={event=>onRemoveVertex(event, index)}><title>Double-click to remove this vertex</title></circle>) }
  </g>;
}

function App() {
  const [plan, setPlan] = useState(null), [imageUrl, setImageUrl] = useState(null), [imageSize, setImageSize] = useState(null);
  const [selected, setSelected] = useState(null), [reviewer, setReviewer] = useState(""), [notes, setNotes] = useState("");
  const [graphSelection, setGraphSelection] = useState([]);
  const [focusedPair, setFocusedPair] = useState(null), [showAllRelations, setShowAllRelations] = useState(false);
  const [reviewedPairs, setReviewedPairs] = useState(new Set()), [pairDecisions, setPairDecisions] = useState({});
  const [step, setStep] = useState("rooms"), [confirmedRooms, setConfirmedRooms] = useState(new Set()), [edgesConfirmed, setEdgesConfirmed] = useState(false);
  const [showOverlay, setShowOverlay] = useState(false), [dragging, setDragging] = useState(null);
  const [draftPoints, setDraftPoints] = useState(null);
  const [zoom, setZoom] = useState(1);
  const imageInput = useRef();
  const selectRoom = plan?.rooms.find(r => r.id === selected);
  const displayedRooms = !plan || !selected ? plan?.rooms : [...plan.rooms.filter(room => room.id !== selected), plan.rooms.find(room => room.id === selected)];
  const labelForId = id => { const room = plan?.rooms.find(candidate => candidate.id === id); return room ? roomLabel(room, plan.rooms) : id; };
  const scale = useMemo(() => plan && imageSize ? imageSize.width / plan.width : 1, [plan, imageSize]);
  // A reviewer should see a complete plan first; zoom is relative to this fit scale.
  const fitScale = imageSize ? Math.min(1, 1100 / imageSize.width, 620 / imageSize.height) : 1;
  const renderScale = fitScale * zoom;
  const roomsComplete = !!plan && confirmedRooms.size === plan.rooms.length;
  // A pair is owned by the earlier room in this fixed source-SVG order. That
  // makes review exhaustive without asking two volunteers the same question.
  const edgePairs = plan ? plan.rooms.flatMap((room, index) => plan.rooms.slice(index + 1).map(target => ({a:room.id, b:target.id}))) : [];
  const currentPair = edgePairs.find(pair => !reviewedPairs.has(edgePairKey(pair)));
  const currentSourceIndex = currentPair ? plan.rooms.findIndex(room => room.id === currentPair.a) : -1;
  const currentExistingEdge = currentPair ? plan.edges.find(edge => edgePairKey(edge) === edgePairKey(currentPair)) : null;
  const allPairsReviewed = !!plan && reviewedPairs.size === edgePairs.length;
  const updateRoom = patch => {
    setPlan(p => ({...p, rooms: p.rooms.map(r => r.id === selected ? {...r, ...patch} : r)}));
    setConfirmedRooms(current => { const next = new Set(current); next.delete(selected); return next; });
    setEdgesConfirmed(false);
  };
  const updateSelectedPoints = points => updateRoom({points, box: bounds(points)});
  const pointerPoint = event => {
    const rect = (event.currentTarget.ownerSVGElement || event.currentTarget).getBoundingClientRect();
    return [(event.clientX - rect.left) / (scale * renderScale), (event.clientY - rect.top) / (scale * renderScale)];
  };
  const moveHandle = event => {
    if (!dragging) return;
    const room = plan.rooms.find(item => item.id === dragging.roomId);
    if (!room) return;
    const point = pointerPoint(event);
    if (dragging.kind === "move") {
      const dx = point[0] - dragging.start[0], dy = point[1] - dragging.start[1];
      const points = dragging.basePoints.map(([x, y]) => [x + dx, y + dy]);
      const box = dragging.baseBox.map((value, index) => value + (index % 2 === 0 ? dx : dy));
      updateRoom({points, box});
    } else if (dragging.kind === "vertex") {
      const points = room.points.map((item, index) => index === dragging.index ? point : item);
      updateSelectedPoints(points);
    } else {
      const box = [...room.box];
      if (dragging.kind.includes("left")) box[0] = Math.min(point[0], box[2] - 1);
      if (dragging.kind.includes("right")) box[2] = Math.max(point[0], box[0] + 1);
      if (dragging.kind.includes("top")) box[1] = Math.min(point[1], box[3] - 1);
      if (dragging.kind.includes("bottom")) box[3] = Math.max(point[1], box[1] + 1);
      // Resize the visible outline and fill with the box.  Keeping only the
      // numeric box in sync made a dragged edge look detached from the room.
      const [oldX0, oldY0, oldX1, oldY1] = room.box;
      const oldWidth = oldX1 - oldX0 || 1, oldHeight = oldY1 - oldY0 || 1;
      const [newX0, newY0, newX1, newY1] = box;
      const points = room.points.map(([x, y]) => [
        newX0 + ((x - oldX0) / oldWidth) * (newX1 - newX0),
        newY0 + ((y - oldY0) / oldHeight) * (newY1 - newY0)
      ]);
      updateRoom({box, points});
    }
  };
  const addVertex = event => {
    if (step !== "rooms" || !selected) return;
    event.stopPropagation();
    const room = plan.rooms.find(item => item.id === selected);
    const point = pointerPoint(event);
    let insertAfter = 0, shortest = Infinity;
    room.points.forEach((start, index) => {
      const distance = distanceToSegment(point, start, room.points[(index + 1) % room.points.length]);
      if (distance < shortest) { shortest = distance; insertAfter = index; }
    });
    // Ignore accidental double-clicks in the middle of a room; a point belongs on its boundary.
    if (shortest > 900) return;
    const points = [...room.points]; points.splice(insertAfter + 1, 0, point); updateSelectedPoints(points);
  };
  const removeVertex = (event, index) => {
    if (step !== "rooms" || !selected || selectRoom.points.length <= 3) return;
    event.stopPropagation();
    updateSelectedPoints(selectRoom.points.filter((_, pointIndex) => pointIndex !== index));
  };
  const addEdge = candidate => {
    if (!candidate.a || !candidate.b || candidate.a === candidate.b) return;
    const canonical = {...candidate, a:[candidate.a, candidate.b].sort()[0], b:[candidate.a, candidate.b].sort()[1]};
    // One unordered room pair has exactly one relation. Setting a new class
    // deliberately replaces its old class instead of creating a duplicate.
    setPlan(current => {
      const previous = current.edges.find(edge => edgePairKey(edge) === edgePairKey(canonical));
      if (previous?.predicate === canonical.predicate) return current;
      return {...current, edges:[...current.edges.filter(edge => edgePairKey(edge) !== edgePairKey(canonical)), canonical]};
    });
    setEdgesConfirmed(false);
  };
  const download = () => {
    // A volunteer submission is evidence for adjudication, not automatic gold truth.
    const record = {plan_id: "reviewed_plan", reviewer, review_status: "submitted_for_adjudication", source_graph_provenance: "silver", reviewer_notes: notes,
      room_types_confirmed: true, graph_edges_confirmed: true,
      rooms: plan.rooms.map(r => ({room_id:r.id, category_id:TYPES.indexOf(r.type)+1, category_name:r.type, bbox_xyxy:r.box.map(n => Math.round(n*100)/100), polygon_svg_points:r.points.map(point => point.map(n => Math.round(n*100)/100))})),
      pair_review: {pair_count:edgePairs.length, reviewed_pair_count:reviewedPairs.size, decisions:pairDecisions},
      edges: uniqueEdges(plan.edges).map(e => ({room_a:e.a, room_b:e.b, predicate_id:PREDICATE_IDS[e.predicate], predicate:e.predicate}))};
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], {type:"application/json"}));
    const a = document.createElement("a"); a.href=url; a.download="cubicasa_review.json"; a.click(); URL.revokeObjectURL(url);
  };
  const updateEdges = change => { setPlan(change); setEdgesConfirmed(false); };
  const chooseRoom = id => { setSelected(id); setShowOverlay(true); if (step === "edges") setStep("rooms"); };
  const selectGraphNode = id => setGraphSelection(current => {
    const next = current.includes(id) ? current.filter(nodeId => nodeId !== id) : current.length < 2 ? [...current, id] : [current[1], id];
    setFocusedPair(next.length === 2 ? edgePairKey({a:next[0], b:next[1]}) : null);
    return next;
  });
  const focusRelation = edge => { setGraphSelection([edge.a, edge.b]); setFocusedPair(edgePairKey(edge)); setShowAllRelations(false); };
  const addSelectedGraphEdge = predicate => {
    if (graphSelection.length !== 2) return;
    addEdge({a:graphSelection[0], b:graphSelection[1], predicate});
    setGraphSelection([]);
  };
  const reviewCurrentPair = predicate => {
    if (!currentPair) return;
    const key = edgePairKey(currentPair), suggested = currentExistingEdge?.predicate || null;
    setReviewedPairs(current => new Set([...current, key]));
    setPairDecisions(current => ({...current, [key]: {room_a:currentPair.a, room_b:currentPair.b, suggested_predicate:suggested, decision:predicate ? suggested === predicate ? "accepted" : suggested ? "relabelled" : "added" : suggested ? "removed" : "no_relation", final_predicate:predicate || null}}));
    if (predicate) addEdge({...currentPair, predicate});
    else updateEdges(current => ({...current, edges:current.edges.filter(edge => edgePairKey(edge) !== key)}));
    setGraphSelection([]); setFocusedPair(null); setShowAllRelations(false);
  };
  const beginEdgeReview = () => {
    setStep("edges"); setGraphSelection([]); setFocusedPair(null); setShowAllRelations(false);
  };
  const showAnnotations = () => {
    if (!showOverlay && !selected) setSelected(plan.rooms.find(room => !confirmedRooms.has(room.id))?.id || plan.rooms[0]?.id || null);
    setShowOverlay(value => !value);
  };
  const confirmRoom = roomId => {
    if (!roomId) return;
    const currentIndex = plan.rooms.findIndex(room => room.id === roomId);
    const orderedCandidates = [...plan.rooms.slice(currentIndex + 1), ...plan.rooms.slice(0, currentIndex)];
    const nextRoom = orderedCandidates.find(room => room.id !== roomId && !confirmedRooms.has(room.id));
    setConfirmedRooms(current => new Set([...current, roomId]));
    setSelected(nextRoom?.id || null);
  };
  const confirmCurrentRoom = () => confirmRoom(selected);
  const resetSelectedOverlay = () => {
    if (!selectRoom?.sourcePoints) return;
    updateRoom({points: selectRoom.sourcePoints, box: selectRoom.sourceBox});
  };
  const deleteRoom = roomId => {
    if (!roomId || !window.confirm(`Delete ${roomId} and all of its graph edges?`)) return;
    const deleted = roomId;
    setPlan(current => ({...current, rooms: current.rooms.filter(room => room.id !== deleted), edges: current.edges.filter(edge => edge.a !== deleted && edge.b !== deleted)}));
    setConfirmedRooms(current => { const next = new Set(current); next.delete(deleted); return next; });
    setSelected(null); setEdgesConfirmed(false);
  };
  const startRedraw = () => { setDraftPoints([]); setSelected(null); setShowOverlay(true); };
  const beginMove = (event, room) => {
    if (step !== "rooms" || selected !== room.id || draftPoints !== null) return;
    event.stopPropagation(); event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
    setDragging({roomId:room.id, kind:"move", start:pointerPoint(event), basePoints:room.points, baseBox:room.box});
  };
  const addDraftPoint = event => {
    if (draftPoints === null) return;
    event.stopPropagation();
    setDraftPoints(points => [...points, pointerPoint(event)]);
  };
  const finishRedraw = () => {
    if (!draftPoints || draftPoints.length < 3) return;
    const used = new Set(plan.rooms.map(room => room.id));
    let index = 1; while (used.has(`NewRoom_${index}`)) index += 1;
    const id = `NewRoom_${index}`, box = bounds(draftPoints);
    const room = {id, type:"Other", points:draftPoints, box, sourcePoints:draftPoints, sourceBox:box};
    setPlan(current => ({...current, rooms:[...current.rooms, room]}));
    setDraftPoints(null); setSelected(id); setEdgesConfirmed(false);
  };
  return <main>
    <header><div><p className="eyebrow">CubiCasa5K · human annotation</p><h1>Room graph reviewer</h1><p className="muted">Confirm semantic room types first. Then independently validate the silver graph edges.</p></div><button className="primary" disabled={!edgesConfirmed} onClick={download}>Download review JSON</button></header>
    <section className="uploads">
      <label>1. CubiGraph relation SVG<input type="file" accept=".svg,image/svg+xml" onChange={async e => { const f=e.target.files[0]; if (!f) return; const loaded=parseSvg(await f.text()); setPlan(loaded); setReviewedPairs(new Set()); setPairDecisions({}); setSelected(null); setStep("rooms"); setGraphSelection([]); setFocusedPair(null); setShowAllRelations(false); setConfirmedRooms(new Set()); setEdgesConfirmed(false); setShowOverlay(false); }}/></label>
      <label>2. Matching floor-plan PNG/JPG<input ref={imageInput} type="file" accept="image/png,image/jpeg" onChange={e => { const f=e.target.files[0]; if (!f) return; const url=URL.createObjectURL(f); const img=new Image(); img.onload=()=>{setImageSize({width:img.width,height:img.height});setImageUrl(url);setZoom(1)}; img.src=url; }}/></label>
      <label>Reviewer ID<input value={reviewer} placeholder="e.g. volunteer_04" onChange={e=>setReviewer(e.target.value)}/></label>
    </section>
    {!plan || !imageUrl ? <div className="empty">Upload the CubiGraph SVG and its matching plan image to begin.</div> : <>
      <nav className="steps"><button className={step === "rooms" ? "current" : ""} onClick={()=>setStep("rooms")}><strong>1</strong><span>Confirm room types<small>{confirmedRooms.size}/{plan.rooms.length} confirmed</small></span></button><i/><button disabled={!roomsComplete} className={step === "edges" ? "current" : ""} onClick={beginEdgeReview}><strong>2</strong><span>Confirm graph edges<small>{edgesConfirmed ? "confirmed" : "locked until reviewed"}</small></span></button></nav>
      <section className="workspace">
      <div className="canvas-panel"><div className="legend"><button className="overlay-toggle" onClick={showAnnotations}>{showOverlay ? "Hide annotations" : "Show annotations"}</button><div className="zoom-controls"><button aria-label="Zoom out" onClick={()=>setZoom(value=>Math.max(0.5, Number((value-0.25).toFixed(2))))}>−</button><label>Zoom <input aria-label="Zoom" type="range" min="0.5" max="3" step="0.05" value={zoom} onChange={event=>setZoom(Number(event.target.value))}/>{Math.round(zoom*100)}%</label><button aria-label="Zoom in" onClick={()=>setZoom(value=>Math.min(3, Number((value+0.25).toFixed(2))))}>+</button><button onClick={()=>setZoom(1)}>Fit</button></div>{showOverlay && (step === "rooms" ? <><span className="room-step">Room type pass</span><span>Only the current room is highlighted. Drag square handles to resize the box.</span></> : <><button className="relation-view-toggle" onClick={()=>setShowAllRelations(value=>!value)}>{showAllRelations ? "Focus current pair" : `Show all ${plan.edges.length} links`}</button><span><i className="dashed"/> adjacency</span><span><i className="door-key"/> door</span><span><i className="open-key"/> open connection</span></>)}</div>
        <div className="plan" style={{width:imageSize.width * renderScale, height:imageSize.height * renderScale}}>
          <img src={imageUrl}/>{showOverlay && <svg className={`overlay ${step === "edges" ? "graph-mode" : ""}`} viewBox={`0 0 ${plan.width} ${plan.height}`} style={{width:plan.width, height:plan.height, transform:`scale(${scale * renderScale})`, transformOrigin:"top left"}} onClickCapture={addDraftPoint} onPointerMove={moveHandle} onPointerUp={()=>setDragging(null)} onPointerCancel={()=>setDragging(null)}>
            {displayedRooms?.map(room => {const semanticColour = typeColour(room.type), graphFocus=step === "edges" && (currentPair?.a === room.id || currentPair?.b === room.id); return <g key={room.id} onClick={()=>step === "rooms" && chooseRoom(room.id)} className={`${selected===room.id ? "selected" : ""} ${graphFocus ? "graph-focus" : ""} ${selected && selected !== room.id ? "muted-room" : ""} ${step === "rooms" && confirmedRooms.has(room.id) ? "confirmed" : ""}`}><polygon points={room.points.map(p=>p.join(",")).join(" ")} fill={step === "edges" ? semanticColour+"2e" : roomColour(room)+"24"} stroke={step === "edges" ? semanticColour : roomColour(room)} style={step === "edges" ? {stroke:semanticColour, strokeWidth:6, fill:semanticColour+"2e"} : selected === room.id ? {stroke:roomColour(room), strokeWidth:8, fill:roomColour(room)+"40"} : undefined} onPointerDown={event=>beginMove(event, room)} onDoubleClick={addVertex}/>{step === "rooms" && <text x={centre(room)[0]} y={centre(room)[1]}>{roomLabel(room, plan.rooms)}</text>}{step === "rooms" && selected === room.id && <EditHandles room={room} onDrag={(event, kind, index) => {event.stopPropagation(); event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); setDragging({roomId:room.id, kind, index});}} onRemoveVertex={removeVertex}/>}</g>})}
            {step === "edges" && plan.edges.filter(edge=>showAllRelations || (currentPair && edgePairKey(edge) === edgePairKey(currentPair))).map(edge => {const a=plan.rooms.find(r=>r.id===edge.a), b=plan.rooms.find(r=>r.id===edge.b); if(!a||!b)return null; const route=edgeRoute(edge, a, b), name=edgePassFor(edge.predicate).short; return <g key={edgePairKey(edge)} className="graph-edge"><path d={route.d} className={edge.predicate === "connected_by_door" ? "door" : edge.predicate === "open_connected" ? "open" : "adjacent"}/><g transform={`translate(${route.x} ${route.y})`}><rect x={-42} y={-14} width="84" height="28" rx="14"/><text y="5">{name}</text></g></g>})}
            {step === "edges" && plan.rooms.map(room => {const [x,y]=centre(room), label=roomLabel(room, plan.rooms), width=Math.max(72, label.length*8+20), colour=typeColour(room.type); return <g key={`node:${room.id}`} className="graph-node"><circle cx={x} cy={y} r="13" fill={colour}/><rect x={x+16} y={y-15} width={width} height="30" rx="8" fill="#ffffff" stroke={colour}/><text x={x+25} y={y+5}>{label}</text></g>})}
            {draftPoints && <g className="draft-room"><polyline points={draftPoints.map(point=>point.join(",")).join(" ")}/>{draftPoints.map((point, index)=><circle key={index} cx={point[0]} cy={point[1]} r="7"/>)}</g>}
          </svg>}
        </div>
      </div>
      <aside>{step === "rooms" ? <><div className="side-title"><div><p className="eyebrow">Step 1</p><h2>Confirm each room</h2></div><b>{confirmedRooms.size}/{plan.rooms.length}</b></div><div className="room-list">{plan.rooms.map(room=>{const isConfirmed=confirmedRooms.has(room.id); const isActive=selected===room.id; return <div className="room-entry" key={room.id}><button className={`room-card ${isActive?"active":""} ${isConfirmed?"done":""}`} onClick={()=>!isConfirmed && chooseRoom(room.id)}><i style={{background:roomColour(room)}}/>{roomLabel(room, plan.rooms)}<small>{isConfirmed ? "Confirmed · " : "Needs review · "}{room.type}</small></button><div className="room-card-actions">{isConfirmed && <button className="edit-confirmed" onClick={()=>chooseRoom(room.id)}>Edit</button>}{isActive && !isConfirmed && <button className="confirm-inline" title="Confirm this room and continue" onClick={()=>confirmRoom(room.id)}>✓</button>}<button className="delete-inline" onClick={()=>deleteRoom(room.id)}>Delete</button></div></div>})}</div>
        {draftPoints ? <div className="redraw-controls"><button onClick={()=>setDraftPoints(points=>points.slice(0,-1))}>Undo point</button><button disabled={draftPoints.length < 3} onClick={finishRedraw}>Finish new room</button><button onClick={()=>setDraftPoints(null)}>Cancel</button></div> : <button className="redraw" onClick={startRedraw}>+ Add room</button>}
        {selectRoom ? <div className="editor"><h3>{roomLabel(selectRoom, plan.rooms)}</h3><label>Room type<select value={selectRoom.type} onChange={e=>updateRoom({type:e.target.value})}>{TYPES.map(t=><option key={t}>{t}</option>)}</select></label><p className="help">Drag the highlighted room directly on the plan; use its edge handles to resize it.</p><div className="room-actions"><button onClick={resetSelectedOverlay}>Reset SVG shape</button></div><button className="confirm-room" onClick={confirmCurrentRoom}>✓ Confirm and continue</button></div> : <p className="muted">{draftPoints ? `Click ${Math.max(0, 3-draftPoints.length)} more point(s) on the plan, then finish.` : "Select a room or add a missing one."}</p>}
        <button className="next" disabled={!roomsComplete} onClick={beginEdgeReview}>{roomsComplete ? "Continue to graph edges →" : `Confirm ${plan.rooms.length-confirmedRooms.size} more room(s)`}</button></> : <>
          <div className="side-title"><div><p className="eyebrow">Step 2</p><h2>Check room connections</h2></div><b>{reviewedPairs.size}/{edgePairs.length}</b></div>
          {currentPair ? <><div className="source-progress"><p className="eyebrow">Room {currentSourceIndex + 1} of {Math.max(1, plan.rooms.length - 1)}</p><h3>From {labelForId(currentPair.a)}</h3><p>Check its direct connection with <b>{labelForId(currentPair.b)}</b>. Earlier room pairs are already finished.</p></div><div className="candidate-review"><p className="eyebrow">{currentExistingEdge ? "CubiGraph suggestion" : "No CubiGraph suggestion"}</p><h3>{labelForId(currentPair.a)} <span>↔</span> {labelForId(currentPair.b)}</h3><p>{currentExistingEdge ? <>CubiGraph suggests <b>{edgePassFor(currentExistingEdge.predicate).short}</b>. Is that correct?</> : <>Inspect the plan. Do these rooms have a direct relationship?</>}</p>{currentExistingEdge && <button className="accept-suggestion" onClick={()=>reviewCurrentPair(currentExistingEdge.predicate)}>✓ Yes, correct</button>}<p className="change-label">{currentExistingEdge ? "Or choose a different outcome:" : "Choose an outcome:"}</p><div className="relation-choices">{EDGE_PASSES.filter(pass=>pass.predicate !== currentExistingEdge?.predicate).map(pass=><button key={pass.predicate} onClick={()=>reviewCurrentPair(pass.predicate)}>{pass.short}</button>)}</div><button className="reject-suggestion" onClick={()=>reviewCurrentPair(null)}>✕ No direct relation</button></div></> : <div className="candidate-review complete"><h3>All room pairs checked</h3><p>Every unordered room pair was reviewed exactly once. You can now confirm the graph.</p></div>}
          <details className="accepted-links"><summary>View accepted links ({plan.edges.length})</summary><div className="edges">{plan.edges.map(edge=><div className="edge-card" key={edgePairKey(edge)}><span>{labelForId(edge.a)} <b>—</b> {labelForId(edge.b)}</span><small>{edgePassFor(edge.predicate).short}</small><button className="locate-edge" onClick={()=>{const key=edgePairKey(edge);setReviewedPairs(current=>{const next=new Set(current);next.delete(key);return next});setEdgesConfirmed(false);setShowAllRelations(false)}}>Recheck</button><button onClick={()=>{const key=edgePairKey(edge);updateEdges(current=>({...current,edges:current.edges.filter(candidate=>edgePairKey(candidate)!==key)}));setReviewedPairs(current=>{const next=new Set(current);next.delete(key);return next})}}>Remove</button></div>)}</div></details>
          <button className="confirm-edges" disabled={!allPairsReviewed} onClick={()=>setEdgesConfirmed(true)}>{allPairsReviewed ? "✓ Confirm graph edges" : `Check ${edgePairs.length-reviewedPairs.size} more pair(s)`}</button>{edgesConfirmed && <p className="ready">Graph confirmed. You can download the review JSON.</p>}<label className="notes">Review notes<textarea value={notes} onChange={e=>setNotes(e.target.value)} placeholder="Why did you change this?"/></label>
        </>}</aside>
    </section></>}
  </main>;
}
createRoot(document.getElementById("root")).render(new URLSearchParams(window.location.search).get("mode") === "review" ? <App/> : <ManualAnnotator/>);
