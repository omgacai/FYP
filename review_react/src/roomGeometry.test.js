import test from 'node:test';
import assert from 'node:assert/strict';
import {bounds,polygonError,contains,withPolygon,exportGraph,SCHEMA} from './roomGeometry.js';
const lShape=[[.1,.1],[.8,.1],[.8,.3],[.3,.3],[.3,.8],[.1,.8]];
test('concave room retains its outline and gets the full rectangular envelope',()=>{
  const n=withPolygon({id:'a',x:.5,y:.5},lShape);
  assert.deepEqual(n.polygon,lShape);assert.deepEqual(n.bbox_xyxy,[.1,.1,.8,.8]);assert.ok(contains(lShape,[n.x,n.y]));assert.equal(contains(lShape,[.5,.5]),false);
});
test('rejects crossing, zero-area, repeated and out-of-image corners',()=>{
  for(const p of [[[0,0],[1,1],[0,1],[1,0]],[[0,0],[.5,.5],[1,1]],[[0,0],[0,0],[1,1]],[[0,0],[2,0],[0,1]]])assert.ok(polygonError(p));
  assert.equal(polygonError(lShape),null);
});
test('outline refinement updates box without changing node identity or an inside anchor',()=>{
  const room=withPolygon({id:'room-1',x:.2,y:.2},lShape);
  const edited=withPolygon(room,lShape.map(([x,y])=>[x===.8?.9:x,y]));
  assert.equal(edited.id,'room-1');assert.equal(edited.x,.2);assert.deepEqual(edited.bbox_xyxy,[.1,.1,.9,.8]);
});
test('JSON round-trip preserves geometry and edges and regenerates stale boxes',()=>{
  const original={nodes:[{id:'a',x:.2,y:.2,polygon:lShape,bbox_xyxy:[0,0,0,0]}],edges:[{a:'a',b:'b',relation:'adjacent_to'}]};
  const saved=exportGraph(original),restored=exportGraph(JSON.parse(JSON.stringify(saved)));
  assert.deepEqual(saved,restored);assert.equal(saved.schema_version,SCHEMA);assert.equal(saved.geometry_complete,true);assert.deepEqual(saved.nodes[0].bbox_xyxy,bounds(lShape));assert.deepEqual(saved.edges,original.edges);
});
test('old point-only rooms remain incomplete without fabricated geometry',()=>{
  const d=exportGraph({schema_version:'floorplan-manual-graph/1',nodes:[{id:'old',x:.1,y:.2}],edges:[]});
  assert.equal(d.geometry_complete,false);assert.equal(d.nodes[0].polygon,null);assert.equal(d.nodes[0].bbox_xyxy,null);
});
