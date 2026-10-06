import test from 'node:test';
import assert from 'node:assert/strict';
import {checkReachability} from './reachability.js';
const nodes=['entry','kitchen','bath','outdoor'].map(id=>({id}));
test('DFS follows doors/open passages in either direction, not adjacency or uncertainty',()=>{
 const edges=[{a:'kitchen',b:'entry',relation:'connected_by_door'},{a:'kitchen',b:'bath',relation:'open_connected'},{a:'bath',b:'outdoor',relation:'adjacent_to'},{a:'entry',b:'outdoor',relation:'uncertain'}];
 assert.deepEqual(checkReachability(nodes,edges,'entry').unreachable,['outdoor']);
 edges.push({a:'bath',b:'outdoor',relation:'connected_by_door'});
 assert.equal(checkReachability(nodes,edges,'outdoor').order.length,4);
});
test('cycles, isolated nodes and missing start',()=>{
 const edges=[{a:'entry',b:'kitchen',relation:'open_connected'},{a:'kitchen',b:'bath',relation:'open_connected'},{a:'bath',b:'entry',relation:'connected_by_door'}];
 assert.equal(new Set(checkReachability(nodes,edges,'entry').order).size,3);
 assert.deepEqual(checkReachability(nodes,edges,'outdoor').unreachable,['entry','kitchen','bath']);
 assert.equal(checkReachability([],[],'entry'),null);
});
test('separate components require explicit approval independent of room type',()=>{
 const rooms=[{id:'entry',type:'Entry'},{id:'garage',type:'Garage'},{id:'storage',type:'Storage'},{id:'balcony',type:'Outdoor'}];
 const edges=[{a:'garage',b:'storage',relation:'connected_by_door'},{a:'entry',b:'balcony',relation:'adjacent_to'}];
 const first=checkReachability(rooms,edges,'entry');
 assert.equal(first.status,'review');assert.equal(first.separate_components.length,2);
 assert.ok(first.separate_components.every(c=>!c.valid_separate_access));
 const review={start_node:'entry',separate_components:first.separate_components.map(c=>({...c,valid_separate_access:true}))};
 assert.equal(checkReachability(rooms,edges,'entry',JSON.parse(JSON.stringify(review))).status,'pass');
 review.separate_components[0].valid_separate_access=false;
 assert.equal(checkReachability(rooms,edges,'entry',review).status,'review');
});
test('changed component membership or main entry does not inherit acceptance',()=>{
 const review={start_node:'entry',separate_components:[{room_ids:['kitchen','bath'],valid_separate_access:true},{room_ids:['outdoor'],valid_separate_access:true}]};
 const joined=[{a:'kitchen',b:'bath',relation:'open_connected'}];
 assert.equal(checkReachability(nodes,joined,'entry',review).status,'pass');
 assert.equal(checkReachability(nodes,[],'entry',review).status,'review');
 assert.equal(checkReachability(nodes,[...joined,{a:'bath',b:'outdoor',relation:'connected_by_door'}],'entry',review).status,'review');
 assert.equal(checkReachability(nodes,joined,'outdoor',review).status,'review');
});
test('separate level is explicitly accepted without inventing a traversable edge',()=>{
 const review={start_node:'entry',separate_components:[{room_ids:['kitchen','bath'],valid_separate_level:true},{room_ids:['outdoor'],valid_separate_access:true}]};
 const edges=[{a:'kitchen',b:'bath',relation:'open_connected'}];
 const result=checkReachability(nodes,edges,'entry',JSON.parse(JSON.stringify(review)));
 assert.equal(result.status,'pass');assert.equal(result.order.length,1);
 assert.equal(result.separate_components[0].valid_separate_level,true);
 assert.equal(result.separate_components[0].valid_separate_access,false);
 assert.equal(checkReachability(nodes,[],'entry',review).status,'review');
 assert.equal(checkReachability(nodes,edges,'entry').status,'review');
});
