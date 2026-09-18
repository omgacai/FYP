// All geometry is in normalized original-raster coordinates, never screen pixels.
export const SCHEMA = 'floorplan-manual-graph/2';
export function bounds(points) {
  return [Math.min(...points.map(p=>p[0])), Math.min(...points.map(p=>p[1])), Math.max(...points.map(p=>p[0])), Math.max(...points.map(p=>p[1]))];
}
const cross = (a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const onSegment=(a,b,p)=>Math.abs(cross(a,b,p))<1e-10 && p[0]>=Math.min(a[0],b[0])-1e-10 && p[0]<=Math.max(a[0],b[0])+1e-10 && p[1]>=Math.min(a[1],b[1])-1e-10 && p[1]<=Math.max(a[1],b[1])+1e-10;
function intersects(a,b,c,d) {
  return (cross(a,b,c)*cross(a,b,d)<0 && cross(c,d,a)*cross(c,d,b)<0) || onSegment(a,b,c)||onSegment(a,b,d)||onSegment(c,d,a)||onSegment(c,d,b);
}
export function polygonError(points) {
  if(!Array.isArray(points)||points.length<3)return 'An outline needs at least 3 corners.';
  if(points.some(p=>!Array.isArray(p)||p.length!==2||p.some(v=>!Number.isFinite(v)||v<0||v>1)))return 'Outline corners must lie inside the image.';
  let area=0;
  for(let i=0;i<points.length;i++) {
    const a=points[i],b=points[(i+1)%points.length];
    if(Math.hypot(a[0]-b[0],a[1]-b[1])<1e-8)return 'Two neighbouring corners overlap.';
    area+=a[0]*b[1]-b[0]*a[1];
    for(let j=i+1;j<points.length;j++) {
      if(j===i+1||(i===0&&j===points.length-1))continue;
      if(intersects(a,b,points[j],points[(j+1)%points.length]))return 'Outline edges cross or touch. Move or remove a corner.';
    }
  }
  return Math.abs(area)<1e-10?'Outline must enclose an area.':null;
}
export function contains(points,p) {
  let inside=false;
  for(let i=0,j=points.length-1;i<points.length;j=i++) {
    const a=points[j],b=points[i];if(onSegment(a,b,p))return true;
    if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])inside=!inside;
  }
  return inside;
}
// A scanline interior point also works for concave rooms whose box centre is outside.
export function interiorPoint(points) {
  const ys=[...new Set(points.map(p=>p[1]))].sort((a,b)=>a-b);
  let best=null;
  for(let k=1;k<ys.length;k++) {
    const y=(ys[k-1]+ys[k])/2,xs=[];
    for(let i=0;i<points.length;i++) {const a=points[i],b=points[(i+1)%points.length];if((a[1]>y)!==(b[1]>y))xs.push(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]));}
    xs.sort((a,b)=>a-b);
    for(let i=0;i+1<xs.length;i+=2)if(!best||xs[i+1]-xs[i]>best.width)best={x:(xs[i]+xs[i+1])/2,y,width:xs[i+1]-xs[i]};
  }
  return [best.x,best.y];
}
export function withPolygon(node,points) {
  const error=polygonError(points);if(error)throw Error(error);
  const [x,y]=contains(points,[node.x,node.y])?[node.x,node.y]:interiorPoint(points);
  return {...node,x,y,polygon:points,bbox_xyxy:bounds(points)};
}
export function exportGraph(doc) {
  const nodes=doc.nodes.map(n=>n.polygon?withPolygon(n,n.polygon):{...n,polygon:null,bbox_xyxy:null});
  return {...doc,schema_version:SCHEMA,coordinate_system:'normalized_xy_top_left',bbox_definition:'axis_aligned_polygon_envelope',geometry_complete:nodes.length>0&&nodes.every(n=>n.polygon),nodes};
}
