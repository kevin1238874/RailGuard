"""A self-contained draggable/clickable calibration canvas for Streamlit >= 1.56.

Uses Streamlit's first-party components v2, not external third-party
click-coordinate services. All content is built with DOM APIs (no innerHTML
from uploaded data); image bytes are passed via a data URL.
"""
from __future__ import annotations

import streamlit as st

CSS = r"""
.rg-editor { font-family: var(--st-font, sans-serif); color: var(--st-text-color); }
.rg-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem; margin:0 0 .6rem; }
.rg-toolbar button { cursor:pointer; color:var(--st-text-color); font-weight:650;
    background:var(--st-secondary-background-color); border:1px solid rgba(140,140,140,.4);
    border-radius:.55rem; padding:.45rem .7rem; }
.rg-toolbar button[aria-pressed="true"] { outline:2px solid var(--st-primary-color); outline-offset:1px; }
.rg-toolbar .rg-chip { font-size:.83rem; opacity:.8; margin-left:auto; }
.rg-stage { position:relative; width:100%; overflow:hidden; border-radius:.7rem;
    border:1px solid rgba(150,150,150,.35); background:#151515; }
.rg-stage img { width:100%; height:auto; display:block; pointer-events:none; user-select:none; }
.rg-stage svg { position:absolute; inset:0; width:100%; height:100%; touch-action:none;
    cursor:crosshair; user-select:none; }
.rg-editor .rg-note { font-size:.84rem; opacity:.85; margin-top:.6rem; }
"""
HTML = """
<div class="rg-editor">
  <div class="rg-toolbar" role="group" aria-label="Select point to place or drag">
    <button type="button" data-index="0">L0 · Safety start</button>
    <button type="button" data-index="1">L1 · Safety end</button>
    <button type="button" data-index="2">E0 · Edge start</button>
    <button type="button" data-index="3">E1 · Edge end</button>
    <span class="rg-chip" aria-live="polite"></span>
  </div>
  <div class="rg-stage">
    <img alt="Uploaded railway platform image. Use the selection buttons and click or drag markers to calibrate." draggable="false" />
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" preserveAspectRatio="none"
      role="group" aria-label="Edit four safety-zone endpoints"></svg>
  </div>
  <div class="rg-note">Select L0/L1/E0/E1 then click the image, or drag an existing point. Orange = safety line L; blue = platform edge E. Yellow overlay = proposed risk zone. All points are provisional until confirmed.</div>
</div>
"""
JS = r"""
export default function(component) {
  const {data, parentElement, setTriggerValue} = component;
  const root = parentElement.querySelector('.rg-editor');
  const stage = root.querySelector('.rg-stage');
  const img = stage.querySelector('img');
  const svg = stage.querySelector('svg');
  const buttons = Array.from(root.querySelectorAll('button[data-index]'));
  const chip = root.querySelector('.rg-chip');
  const ns = 'http://www.w3.org/2000/svg';
  img.src = 'data:image/jpeg;base64,' + (data.image_jpeg || '');

  // Parent data is authoritative after a Python rerun (auto detect / YAML load).
  let points = (data.points || [null, null, null, null]).map(p => p ? [p[0], p[1]] : null);
  let selected = points.findIndex(p => p === null);
  if (selected === -1) selected = 0;
  let dragging = null;
  const colors = ['#FFB020', '#FFB020', '#33B9FF', '#33B9FF'];
  const clamp = v => Math.max(0, Math.min(1, v));
  const el = (type, attrs={}) => {
    const node = document.createElementNS(ns, type);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    return node;
  };
  const xy = p => [p[0] * 1000, p[1] * 1000];
  const loc = event => {
    const rect = svg.getBoundingClientRect();
    return [clamp((event.clientX - rect.left) / rect.width),
            clamp((event.clientY - rect.top) / rect.height)];
  };
  function render() {
    svg.replaceChildren();
    if (points.every(p => p !== null)) {
      const polygon = [0, 1, 3, 2].map(i => xy(points[i]).join(',')).join(' ');
      svg.appendChild(el('polygon', {points: polygon, fill: '#FFCA4F', 'fill-opacity': '.20',
            stroke: '#FFD065', 'stroke-width': 2, 'stroke-opacity': '.35'}));
    }
    for (const [i,j,color] of [[0,1,colors[0]], [2,3,colors[2]]]) {
      if (points[i] && points[j]) {
        const a = xy(points[i]), b = xy(points[j]);
        svg.appendChild(el('line', {x1:a[0],y1:a[1],x2:b[0],y2:b[1],
            stroke:'#111', 'stroke-width':9, 'stroke-opacity':'.85'}));
        svg.appendChild(el('line', {x1:a[0],y1:a[1],x2:b[0],y2:b[1],
            stroke:color, 'stroke-width':5}));
      }
    }
    points.forEach((point, i) => {
      if (!point) return;
      const [x,y] = xy(point);
      svg.appendChild(el('circle', {cx:x,cy:y,r:17,fill:colors[i],stroke:'#111',
        'stroke-width':5,'data-index':i, style:'cursor:grab'}));
      const label = el('text', {x:x+23, y:y-20, fill:'#fff',stroke:'#111',
        'stroke-width':4, 'paint-order':'stroke', 'font-size':33,
        'font-weight':'700', 'pointer-events':'none'});
      label.textContent = ['L0','L1','E0','E1'][i];
      svg.appendChild(label);
    });
    buttons.forEach((b,idx) => b.setAttribute('aria-pressed',idx === selected ? 'true' : 'false'));
    const done = points.filter(p => p !== null).length;
    chip.textContent = done + '/4 points placed';
  }
  buttons.forEach(button => {
    button.onclick = () => { selected = Number(button.dataset.index); render(); };
  });
  svg.onpointerdown = event => {
    if (event.button !== 0 && event.pointerType !== 'touch') return;
    event.preventDefault();
    const index = Number(event.target.getAttribute('data-index'));
    if (Number.isInteger(index) && index >= 0 && index < 4 &&
        event.target.hasAttribute('data-index')) selected = index;
    dragging = selected;
    svg.setPointerCapture(event.pointerId);
    points[selected] = loc(event);
    render();
  };
  svg.onpointermove = event => {
    if (dragging === null) return;
    points[dragging] = loc(event);
    render();
  };
  svg.onpointerup = event => {
    if (dragging === null) return;
    points[dragging] = loc(event);
    dragging = null;
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    setTriggerValue('edit', {generation:data.generation, points:points});
  };
  svg.onpointercancel = event => {
    dragging = null;
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
  };
  render();
}
"""

# Components must be registered once per app run, not inside a loop.
zone_canvas = st.components.v2.component("railguard_zone_canvas", html=HTML, css=CSS, js=JS)


def render_zone_canvas(image_b64: str, points: list, generation: int):
    """Return a one-shot result.edit with four normalized coordinates after a user edit."""
    return zone_canvas(
        key="railguard_zone_editor",
        data={"image_jpeg": image_b64, "points": points, "generation": generation},
        on_edit_change=lambda: None,
    )
