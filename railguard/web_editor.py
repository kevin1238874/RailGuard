"""Self-contained Streamlit v2 calibration canvas with independent zoom and pan.

Point coordinates are image-normalized so zoom, pan, resizing and re-renders
never alter the saved camera calibration. Uploaded bytes are only used as an
image data URL; no external JS dependency is loaded.
"""
from __future__ import annotations

import streamlit as st

CSS = r"""
.rg-editor {width:100%; color:var(--st-text-color);font-family:var(--st-font,sans-serif)}
.rg-toolbar {display:flex;flex-wrap:wrap;align-items:center;gap:.45rem;margin-bottom:.65rem}
.rg-toolbar button {cursor:pointer;color:var(--st-text-color);font-weight:600;
 background:var(--st-secondary-background-color);border:1px solid rgba(140,140,140,.45);
 border-radius:.5rem;padding:.46rem .6rem;white-space:nowrap}
.rg-toolbar button[aria-pressed="true"] {outline:2px solid var(--st-primary-color);outline-offset:1px}
.rg-toolbar .rg-spacer {flex:1 1 auto;min-width:8px}
.rg-toolbar .rg-zoom-label {min-width:3.2rem;text-align:center;font-variant-numeric:tabular-nums}
.rg-viewport {position:relative;width:100%;height:clamp(330px,65vh,690px);overflow:hidden;
 background:#18191b;border:1px solid rgba(140,140,140,.45);border-radius:.7rem;
 touch-action:none;user-select:none;}
.rg-stage {position:absolute;left:0;top:0;transform-origin:top left;will-change:transform;}
.rg-stage img {display:block;width:100%;height:100%;pointer-events:none;user-select:none}
.rg-stage svg {position:absolute;left:0;top:0;width:100%;height:100%;touch-action:none;
 cursor:crosshair;user-select:none}
.rg-viewport[data-pan="true"] .rg-stage svg {cursor:grab}
.rg-viewport[data-dragging="true"] .rg-stage svg {cursor:grabbing}
.rg-editor .rg-note {font-size:.84rem;opacity:.85;margin-top:.55rem}
.rg-editor .rg-help {font-size:.83rem;opacity:.75;margin-left:auto}
"""

HTML = """
<div class="rg-editor">
 <div class="rg-toolbar" aria-label="Edit safety zone">
  <button type="button" data-index="0">L0 · Start</button>
  <button type="button" data-index="1">L1 · End</button>
  <button type="button" data-index="2">E0 · Start</button>
  <button type="button" data-index="3">E1 · End</button>
  <span class="rg-spacer"></span>
  <button type="button" data-action="zoom-out" aria-label="Zoom out image">−</button>
  <span class="rg-zoom-label" aria-live="polite">100%</span>
  <button type="button" data-action="zoom-in" aria-label="Zoom in image">＋</button>
  <button type="button" data-action="fit">Fit</button>
  <button type="button" data-action="pan" aria-pressed="false">✋ Pan</button>
 </div>
 <div class="rg-viewport" data-pan="false" aria-label="Image calibration viewport">
  <div class="rg-stage">
    <img alt="Railway platform image for drawing safety and platform edge lines" draggable="false" />
    <svg xmlns="http://www.w3.org/2000/svg" role="group" aria-label="Four editable safety-zone endpoints"></svg>
  </div>
 </div>
 <div class="rg-note">Orange L = painted safety line; blue E = platform edge. Select a point and click, or drag its marker. Use ＋/− or the mouse wheel to zoom **inside the image**; select ✋ Pan to move the enlarged image. Fit restores the full view. Confirm both lines before inference.</div>
</div>
"""

JS = r"""
export default function(component) {
  const {data, parentElement, setTriggerValue} = component;
  const root = parentElement.querySelector('.rg-editor');
  const viewport = root.querySelector('.rg-viewport');
  const stage = root.querySelector('.rg-stage');
  const img = stage.querySelector('img');
  const svg = stage.querySelector('svg');
  const pointButtons = Array.from(root.querySelectorAll('button[data-index]'));
  const zoomLabel = root.querySelector('.rg-zoom-label');
  const ns = 'http://www.w3.org/2000/svg';
  const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
  const colors = ['#FFB020','#FFB020','#34C4FF','#34C4FF'];
  const points = (data.points || [null,null,null,null]).map(p => p ? [p[0],p[1]] : null);
  let selected = points.findIndex(p => p === null);
  if (selected === -1) selected = 0;
  let panMode = false, active = null;
  let width = 0, height = 0, baseW = 0, baseH = 0;
  let zoom = clamp(Number(data.view?.zoom) || 1,1,12);
  let center = Array.isArray(data.view?.center) ? data.view.center : [.5,.5];
  let tx=0,ty=0;
  img.src = 'data:image/jpeg;base64,' + (data.image_jpeg || '');

  const el = (name, attrs={}) => {
    const node = document.createElementNS(ns,name);
    for (const [k,v] of Object.entries(attrs)) node.setAttribute(k,String(v));
    return node;
  };
  const xy = p => [p[0]*width,p[1]*height];
  const pointerLoc = event => {
    const rect = svg.getBoundingClientRect();
    return [clamp((event.clientX - rect.left) / Math.max(rect.width,1),0,1),
            clamp((event.clientY - rect.top) / Math.max(rect.height,1),0,1)];
  };
  function saveView() {
    const vw = viewport.clientWidth, vh=viewport.clientHeight;
    const cx=(vw*.5-tx)/(Math.max(baseW*zoom,1));
    const cy=(vh*.5-ty)/(Math.max(baseH*zoom,1));
    return {zoom:zoom, center:[clamp(cx,0,1),clamp(cy,0,1)]};
  }
  function constrain() {
    const vw=viewport.clientWidth, vh=viewport.clientHeight;
    const sw=baseW*zoom, sh=baseH*zoom;
    tx=sw<=vw ? (vw-sw)/2 : clamp(tx,vw-sw,0);
    ty=sh<=vh ? (vh-sh)/2 : clamp(ty,vh-sh,0);
  }
  function redrawTransform() {
    constrain();
    stage.style.transform=`translate(${tx}px,${ty}px) scale(${zoom})`;
    zoomLabel.textContent=Math.round(zoom*100)+'%';
  }
  function syncSize() {
    if (!img.naturalWidth || !img.naturalHeight) return;
    const old = width > 0 ? saveView() : {zoom,center};
    width=img.naturalWidth; height=img.naturalHeight;
    const factor=Math.min(viewport.clientWidth/width,viewport.clientHeight/height);
    baseW=Math.max(1,width*factor);baseH=Math.max(1,height*factor);
    stage.style.width=baseW+'px';stage.style.height=baseH+'px';
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    zoom=old.zoom;
    tx=viewport.clientWidth*.5 - old.center[0]*baseW*zoom;
    ty=viewport.clientHeight*.5 - old.center[1]*baseH*zoom;
    redrawTransform();draw();
  }
  function changeZoom(next,clientX=null,clientY=null) {
    if (!width) return;
    next=clamp(next,1,12);
    const rect=viewport.getBoundingClientRect();
    const px=clientX===null ? rect.width/2 : clientX-rect.left;
    const py=clientY===null ? rect.height/2 : clientY-rect.top;
    const qx=(px-tx)/zoom, qy=(py-ty)/zoom;
    zoom=next;
    tx=px-qx*zoom;ty=py-qy*zoom;
    redrawTransform();
  }
  function draw() {
    if(!width || !height) return;
    svg.replaceChildren();
    if (points.every(p => p !== null)) {
      const poly=[0,1,3,2].map(i=>xy(points[i]).join(',')).join(' ');
      svg.appendChild(el('polygon',{points:poly,fill:'#FFC74C','fill-opacity':'.22',
        stroke:'#FFD166','stroke-width':Math.max(1,width*.0015),'stroke-opacity':'.65'}));
    }
    for (const [i,j,col] of [[0,1,colors[0]],[2,3,colors[2]]]) {
      if (!points[i] || !points[j]) continue;
      const a=xy(points[i]),b=xy(points[j]),sw=Math.max(3,width*.003);
      svg.appendChild(el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],
        stroke:'#111','stroke-opacity':'.85','stroke-width':sw+4}));
      svg.appendChild(el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],
        stroke:col,'stroke-width':sw}));
    }
    const radius=Math.max(6,Math.min(width,height)*.016);
    points.forEach((p,i)=>{
      if(!p)return;
      const [x,y]=xy(p);
      svg.appendChild(el('circle',{cx:x,cy:y,r:radius+Math.max(7,radius*.7),
        fill:'transparent','data-index':i,style:'cursor:grab'}));
      svg.appendChild(el('circle',{cx:x,cy:y,r:radius,fill:colors[i],stroke:'#101010',
        'stroke-width':Math.max(2,radius*.25),'pointer-events':'none'}));
      const label=el('text',{x:x+radius*1.4,y:y-radius*.9,fill:'#fff',stroke:'#111',
        'paint-order':'stroke','stroke-width':Math.max(2,radius*.4),
        'font-size':radius*2,'font-weight':'700','pointer-events':'none'});
      label.textContent=['L0','L1','E0','E1'][i];svg.appendChild(label);
    });
    pointButtons.forEach((b,i)=>b.setAttribute('aria-pressed',i===selected&&!panMode?'true':'false'));
    viewport.dataset.pan=panMode?'true':'false';
    root.querySelector('button[data-action="pan"]').setAttribute('aria-pressed',panMode?'true':'false');
  }
  pointButtons.forEach(b => b.addEventListener('click',()=>{
    selected=Number(b.dataset.index);panMode=false;draw();
  }));
  root.querySelector('[data-action="zoom-in"]').onclick=()=>changeZoom(zoom*1.3);
  root.querySelector('[data-action="zoom-out"]').onclick=()=>changeZoom(zoom/1.3);
  root.querySelector('[data-action="fit"]').onclick=()=>{
    zoom=1;tx=(viewport.clientWidth-baseW)/2;ty=(viewport.clientHeight-baseH)/2;
    redrawTransform();
  };
  root.querySelector('[data-action="pan"]').onclick=()=>{panMode=!panMode;draw();};
  viewport.addEventListener('wheel', e=>{
    e.preventDefault();e.stopPropagation();
    changeZoom(zoom*Math.exp(-e.deltaY*.0014),e.clientX,e.clientY);
  },{passive:false});
  svg.addEventListener('pointerdown',e=>{
    if (e.button!==0 && e.button!==1 && e.pointerType!=='touch') return;
    e.preventDefault();
    const target=e.target;
    const index=target.hasAttribute('data-index')?Number(target.getAttribute('data-index')):-1;
    const movingImage=panMode||e.button===1;
    if (movingImage) {
      active={kind:'pan',id:e.pointerId,x:e.clientX,y:e.clientY};
      viewport.dataset.dragging='true';
    } else {
      if(index>=0 && index<4) selected=index;
      active={kind:'point',id:e.pointerId,index:selected};
      points[selected]=pointerLoc(e);draw();
    }
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove',e=>{
    if(!active || active.id!==e.pointerId)return;
    if(active.kind==='pan'){
      tx+=e.clientX-active.x;ty+=e.clientY-active.y;
      active.x=e.clientX;active.y=e.clientY;
      redrawTransform();
    }else{
      points[active.index]=pointerLoc(e);draw();
    }
  });
  svg.addEventListener('pointerup',e=>{
    if(!active || active.id!==e.pointerId)return;
    if(active.kind==='point'){
      points[active.index]=pointerLoc(e);
      setTriggerValue('edit',{generation:data.generation,points,view:saveView()});
    }
    active=null;viewport.dataset.dragging='false';
    if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId);
    draw();
  });
  svg.addEventListener('pointercancel',e=>{
    active=null;viewport.dataset.dragging='false';
    if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId);
  });
  img.onload=syncSize;
  if(img.complete&&img.naturalWidth)syncSize();
  if(typeof ResizeObserver!=='undefined'){
    const observer=new ResizeObserver(()=>syncSize());
    observer.observe(viewport);
  }
}
"""

# Register once per app run; no external component packages are required.
zone_canvas = st.components.v2.component("railguard_zone_canvas", html=HTML, css=CSS, js=JS)


def render_zone_canvas(image_b64: str, points: list, generation: int,
                       view: dict | None = None):
    """Emit result.edit on point placement/drag, with normalized points and view."""
    return zone_canvas(
        key="railguard_zone_editor",
        data={"image_jpeg": image_b64, "points": points, "generation": generation,
              "view": view or {"zoom": 1, "center": [0.5, 0.5]}},
        on_edit_change=lambda: None,
    )
