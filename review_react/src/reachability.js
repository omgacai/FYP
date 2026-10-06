// Door and open passage are symmetric access relations. Adjacency is not access.
export const componentKey = ids => JSON.stringify([...ids].sort());
export const componentAccepted = component => component.valid_separate_access===true || component.valid_separate_level===true;
export function checkReachability(nodes, edges, start, review = {}) {
  const neighbors=new Map(nodes.map(n=>[n.id,[]]));
  for(const edge of edges){
    if(!['connected_by_door','open_connected'].includes(edge.relation)||!neighbors.has(edge.a)||!neighbors.has(edge.b))continue;
    neighbors.get(edge.a).push(edge.b);neighbors.get(edge.b).push(edge.a);
  }
  if(!neighbors.has(start))return null;
  const visited=new Set();
  const walk=seed=>{const order=[],stack=[seed];while(stack.length){const id=stack.pop();if(visited.has(id))continue;visited.add(id);order.push(id);for(const next of neighbors.get(id))if(!visited.has(next))stack.push(next);}return order;};
  const order=walk(start),separate_components=[];
  // Approval only applies to exactly the same component under the same entry.
  const approvals=review.start_node===start && Array.isArray(review.separate_components)?review.separate_components:[];
  for(const node of nodes)if(!visited.has(node.id)){
    const room_ids=walk(node.id).sort();
    const prior=approvals.find(c=>Array.isArray(c.room_ids)&&componentKey(c.room_ids)===componentKey(room_ids));
    separate_components.push({room_ids,valid_separate_access:prior?.valid_separate_access===true,valid_separate_level:prior?.valid_separate_level===true});
  }
  return {start,order,separate_components,unreachable:nodes.filter(n=>!order.includes(n.id)).map(n=>n.id),total:nodes.length,status:separate_components.some(c=>!componentAccepted(c))?'review':'pass'};
}
