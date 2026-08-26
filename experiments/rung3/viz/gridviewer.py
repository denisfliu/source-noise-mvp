"""Synced grid of compact 3D trajectory viewers (Denis, 2026-08-12).

Differences from cloudviewer.py: ALL panels share ONE legend — a checkbox toggles that arm in
every panel simultaneously; panels are compact (2-up grid) so most scenes are visible at once;
middle-drag (or shift-drag) pans, left-drag orbits, wheel zooms. One script block drives all
panels (cloudviewer's per-panel scripts bound their legend handlers to every page legend by
index — a latent cross-talk bug this design removes).

grid_html(panels, arms) ->  (legend_html, grid_html, script_html)
  panels = [{"scene": str, "title": str, "id": str, "groups": {armkey: [Nx3, ...]}}]
  arms   = [{"key": str, "label": str, "color": [r,g,b]}]   # order = legend order
"""
import base64
import json
import os

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
HEIGHT = 300
MAX_PTS = 9000


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def grid_html(panels, arms):
    payload = {"arms": [a["key"] for a in arms], "panels": []}
    for p in panels:
        z = np.load(f"{SP}/scene_cloud_{p['scene']}.npz")
        pts, rgb = z["pts"].astype(np.float32), z["rgb"]
        if len(pts) > MAX_PTS:
            k = np.random.default_rng(0).permutation(len(pts))[:MAX_PTS]
            pts, rgb = pts[k], rgb[k]
        centre = pts.mean(0)
        colors = {a["key"]: a["color"] for a in arms}
        payload["panels"].append({
            "id": p["id"], "n": int(len(pts)),
            "pts": _b64((pts - centre).astype(np.float32)), "rgb": _b64(rgb.astype(np.uint8)),
            "groups": {k: {"color": colors[k],
                           "trajs": [_b64((np.asarray(t, np.float32)[:, :3] - centre).astype(np.float32))
                                     for t in ts]}
                       for k, ts in p["groups"].items() if ts},
        })
    legend = "".join(
        f'<label class="lg"><input type="checkbox" checked data-key="{a["key"]}">'
        f'<span class="sw" style="background:rgb({a["color"][0]},{a["color"][1]},{a["color"][2]})"></span>'
        f'{a["label"]}</label>' for a in arms)
    legend += '<label class="lg"><input type="checkbox" checked data-key="__cloud"><span class="sw" style="background:#888"></span>scene cloud</label>'
    grid = "".join(
        f'<div class="gp"><div class="gt">{p["title"]}</div><canvas id="{p["id"]}" height="{HEIGHT}"></canvas></div>'
        for p in panels)
    js = """
<script>
(function(){
const D = __PAYLOAD__;
function dec(b64){const s=atob(b64);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
const VS=`attribute vec3 p; attribute vec3 c; uniform mat4 M; uniform float ps;
 varying vec3 vc; void main(){ gl_Position=M*vec4(p,1.0); gl_PointSize=ps; vc=c; }`;
const FS=`precision mediump float; varying vec3 vc; uniform float alpha;
 void main(){ gl_FragColor=vec4(vc,alpha); }`;
const on={}; D.arms.forEach(k=>on[k]=true); on.__cloud=true;
const views=[];
D.panels.forEach(P=>{
  const cv=document.getElementById(P.id);
  const gl=cv.getContext("webgl",{antialias:true,alpha:false});
  function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);return x;}
  const pr=gl.createProgram(); gl.attachShader(pr,sh(gl.VERTEX_SHADER,VS)); gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,FS));
  gl.linkProgram(pr); gl.useProgram(pr);
  const aP=gl.getAttribLocation(pr,"p"), aC=gl.getAttribLocation(pr,"c");
  const uM=gl.getUniformLocation(pr,"M"), uPS=gl.getUniformLocation(pr,"ps"), uA=gl.getUniformLocation(pr,"alpha");
  function buf(a){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);return b;}
  const pts=new Float32Array(dec(P.pts).buffer), rgb=dec(P.rgb);
  const rf=new Float32Array(rgb.length); for(let i=0;i<rgb.length;i++)rf[i]=rgb[i]/255;
  const bP=buf(pts), bC=buf(rf);
  const gs={};
  Object.entries(P.groups).forEach(([k,g])=>{
    gs[k]=g.trajs.map(t=>{const a=new Float32Array(dec(t).buffer);
      const c=new Float32Array(a.length);
      for(let i=0;i<a.length;i+=3){c[i]=g.color[0]/255;c[i+1]=g.color[1]/255;c[i+2]=g.color[2]/255;}
      return {n:a.length/3,bp:buf(a),bc:buf(c)};});
  });
  const st={yaw:-0.6,pitch:0.45,dist:9,panx:0,pany:0};
  function mat(){
    const cy=Math.cos(st.yaw),sy=Math.sin(st.yaw),cp=Math.cos(st.pitch),sp=Math.sin(st.pitch);
    const ex=st.dist*cp*sy, ey=-st.dist*cp*cy, ez=st.dist*sp;
    const f=[-ex,-ey,-ez]; const fl=Math.hypot(...f); f.forEach((v,i)=>f[i]=v/fl);
    let s=[f[1],-f[0],0]; const sl=Math.hypot(...s)||1; s=s.map(v=>v/sl);
    const u=[s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0]];
    const V=[s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0,
     -(s[0]*ex+s[1]*ey+s[2]*ez)+st.panx, -(u[0]*ex+u[1]*ey+u[2]*ez)+st.pany, (f[0]*ex+f[1]*ey+f[2]*ez), 1];
    const asp=cv.width/cv.height, n=0.05, fa=200, t=1/Math.tan(0.5);
    const Pm=[t/asp,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];
    const M=new Float32Array(16);
    for(let i=0;i<4;i++)for(let j=0;j<4;j++){let v=0;for(let k=0;k<4;k++)v+=Pm[k*4+j]*V[i*4+k];M[i*4+j]=v;}
    return M;
  }
  function draw(){
    const dpr=Math.min(window.devicePixelRatio||1,2);
    cv.width=cv.clientWidth*dpr; cv.height=__H__*dpr;
    gl.viewport(0,0,cv.width,cv.height);
    gl.clearColor(0.07,0.08,0.10,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    gl.uniformMatrix4fv(uM,false,mat());
    if(on.__cloud){
      gl.uniform1f(uPS,1.3*dpr); gl.uniform1f(uA,0.55);
      gl.bindBuffer(gl.ARRAY_BUFFER,bP); gl.enableVertexAttribArray(aP); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
      gl.bindBuffer(gl.ARRAY_BUFFER,bC); gl.enableVertexAttribArray(aC); gl.vertexAttribPointer(aC,3,gl.FLOAT,false,0,0);
      gl.drawArrays(gl.POINTS,0,P.n);
    }
    gl.uniform1f(uA,0.95); gl.uniform1f(uPS,2.4*dpr);
    Object.entries(gs).forEach(([k,ts])=>{ if(!on[k]) return;
      ts.forEach(t=>{
        gl.bindBuffer(gl.ARRAY_BUFFER,t.bp); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
        gl.bindBuffer(gl.ARRAY_BUFFER,t.bc); gl.vertexAttribPointer(aC,3,gl.FLOAT,false,0,0);
        gl.drawArrays(gl.LINE_STRIP,0,t.n); gl.drawArrays(gl.POINTS,0,t.n);
      });
    });
  }
  let drag=null;
  cv.addEventListener("mousedown",e=>{ if(e.button===1) e.preventDefault();
    drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===1}; });
  cv.addEventListener("auxclick",e=>e.preventDefault());
  window.addEventListener("mouseup",()=>drag=null);
  window.addEventListener("mousemove",e=>{ if(!drag) return;
    const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
    if(drag.pan){ st.panx+=dx*0.004; st.pany-=dy*0.004; }
    else { st.yaw+=dx*0.006; st.pitch=Math.max(-1.4,Math.min(1.4,st.pitch+dy*0.006)); }
    draw();
  });
  cv.addEventListener("wheel",e=>{e.preventDefault(); st.dist*=Math.exp(e.deltaY*0.0012);
    st.dist=Math.max(1.5,Math.min(60,st.dist)); draw();},{passive:false});
  views.push(draw);
});
function redraw(){views.forEach(d=>d());}
document.querySelectorAll('.glegend input[data-key]').forEach(cb=>cb.addEventListener("change",()=>{
  on[cb.dataset.key]=cb.checked; redraw();}));
window.addEventListener("resize",redraw); redraw();
})();
</script>"""
    js = js.replace("__PAYLOAD__", json.dumps(payload)).replace("__H__", str(HEIGHT))
    return legend, grid, js


STYLE = """
.glegend{position:sticky;top:0;z-index:5;display:flex;gap:18px;flex-wrap:wrap;align-items:center;
 background:var(--card);border:1px solid var(--line);border-radius:10px;padding:11px 16px;
 font:600 .8rem ui-monospace,Menlo,monospace;color:var(--mut)}
.glegend .lg{display:flex;gap:6px;align-items:center;cursor:pointer}
.glegend .sw{width:20px;height:3px;border-radius:2px;display:inline-block}
.ggrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px;margin-top:12px}
.gp{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px 10px}
.gt{font:600 .8rem ui-monospace,Menlo,monospace;color:var(--mut);margin:0 0 6px}
.gp canvas{width:100%;display:block;border-radius:6px;background:#12141a;cursor:grab}
.gp canvas:active{cursor:grabbing}
"""
