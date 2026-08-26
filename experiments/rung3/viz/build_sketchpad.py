"""Sketchpad: draw a corrective sketch on the scene point cloud, get SNMVP_PIN_PROMPT json.

Workflow: pick scene -> set the placement height (z slider) -> sketch mode on -> click
waypoints in order (first point = activation trigger; last = handback) -> copy the JSON from
the panel into experiments/rung3/sketch_<name>.json -> serve with SNMVP_PIN_PROMPT pointing
at it (scripts/run_sketch_cmpl.sh is the template). Yaw is auto-derived from the path
tangent at export.

  python3 build_sketchpad.py   (writes sketchpad.html next to this file)
"""
import base64
import glob
import json
import os

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
RUN = "/home/dfliu/ctxrun"
GOAL_C, GOAL_H = [1.525, -0.615, 1.0], [0.3, 0.3, 0.5]
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}
SCENES = {"left_and_center": (["left", "center"], "traj_gmsig3_cmpl_*.npy",
                              "go through the center gate from the left and hover over the stuffed animal"),
          "right_and_center": (["right", "center"], "traj_gmsig3_cmpr_*.npy",
                               "go through the center gate from the right and hover over the stuffed animal")}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def box_edges(c, h):
    c, h = np.asarray(c), np.asarray(h)
    k = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    corners = c + k * h
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [corners[[a, b]].astype(np.float32) for a, b in idx]


payload = {}
for scene, (aps, pat, prompt) in SCENES.items():
    z = np.load(f"{SP}/scene_cloud_{scene}.npz")
    pts, rgb = z["pts"].astype(np.float32), z["rgb"]
    if len(pts) > 40000:
        k = np.random.default_rng(0).permutation(len(pts))[:40000]
        pts, rgb = pts[k], rgb[k]
    refs = [np.load(f)[:, :3].astype(np.float32) for f in sorted(glob.glob(f"{RUN}/{pat}"))]
    marks = [np.array(APERTURE[a] + [APERTURE[a][0]], np.float32) for a in aps] + box_edges(GOAL_C, GOAL_H)
    payload[scene] = {"n": int(len(pts)), "pts": b64(pts), "rgb": b64(rgb.astype(np.uint8)),
                      "refs": [b64(t) for t in refs], "marks": [b64(m) for m in marks],
                      "prompt": prompt}
J = json.dumps(payload)

page = """<title>Sketchpad</title>
<style>
:root{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0;--warn:#ffab42}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;padding:20px 16px 60px}
main{max-width:1160px;margin:0 auto}
h1{font-size:21px;margin:0 0 2px}
.sub{color:var(--mut);margin:0 0 14px;max-width:100ch}
.row{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
.pad{flex:2 1 640px;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px}
canvas{width:100%;display:block;border-radius:6px;background:#12141a;cursor:grab}
canvas.sketching{cursor:crosshair}
.side{flex:1 1 300px;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:12px;
 display:flex;flex-direction:column;gap:10px;font-size:13px}
.ui{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace;color:var(--mut)}
.ui label{display:inline-flex;gap:5px;align-items:center;cursor:pointer}
button{background:#1d2530;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 11px;
 font:600 12px system-ui;cursor:pointer}
button:hover{border-color:var(--acc)}
button.primary{background:var(--acc);color:#0c1116;border-color:var(--acc)}
textarea{width:100%;min-height:210px;background:#0d1116;color:#cfe3ef;border:1px solid var(--line);
 border-radius:6px;font:11.5px ui-monospace,Menlo,monospace;padding:8px;box-sizing:border-box}
input[type=text],input[type=number]{background:#0d1116;color:var(--ink);border:1px solid var(--line);
 border-radius:5px;padding:4px 7px;font:12px ui-monospace,Menlo,monospace}
input[type=range]{width:150px}
.f{display:flex;gap:7px;align-items:center;justify-content:space-between}
.f span{color:var(--mut)}
.zval{color:var(--warn);font:600 12px ui-monospace,Menlo,monospace}
.npts{color:var(--acc);font-weight:600}
.hint{color:var(--mut);font-size:12px}
</style>
<main>
<h1>Sketchpad</h1>
<p class="sub">Draw the corrective segment: <b>sketch mode on &rarr; click waypoints in order</b> on the
height plane set by the z slider (shown as a faint grid). First point is the activation trigger
(the drone must fly within <i>enter_radius</i> of it), last point is where the head takes back over
under <i>prompt_after</i>. <b>Use "top view" when placing points</b> — clicking from an oblique
camera lands the point where the ray meets the height plane, which can sit tens of cm behind
what you are visually aiming at (this parallax cost the first hand-drawn CMPR sketch the
right gate by 25 cm). Grey = the unguided flights (where it goes wrong), blue/yellow = judge
geometry. Copy the JSON into <code>experiments/rung3/sketch_&lt;name&gt;.json</code> and serve with
<code>SNMVP_PIN_PROMPT</code>.</p>
<div class="row">
 <div class="pad">
  <canvas id="cv" height="560"></canvas>
  <div class="ui">
   <label><input type="radio" name="sc" value="left_and_center" checked> left_and_center</label>
   <label><input type="radio" name="sc" value="right_and_center"> right_and_center</label>
   <label><input type="checkbox" id="sketchmode"> <b style="color:var(--warn)">sketch mode</b></label>
   <label>z <input type="range" id="zsl" min="0.2" max="2.0" step="0.05" value="1.5">
    <span class="zval" id="zv">1.50</span> m</label>
   <label><input type="checkbox" id="showrefs" checked> unguided flights</label>
   <button id="topview" title="parallax-free drawing view">top view</button>
   <span style="margin-left:auto">drag orbit &middot; wheel zoom &middot; shift-drag pan</span>
  </div>
 </div>
 <div class="side">
  <div class="f"><span>waypoints</span><span class="npts" id="np">0</span></div>
  <div class="f"><button id="undo">undo point</button><button id="clear">clear</button></div>
  <div class="f"><span>prompt_after</span></div>
  <input type="text" id="prompt" value="">
  <div class="f"><span>enter_radius</span><input type="number" id="er" value="0.45" step="0.05" style="width:70px"></div>
  <div class="f"><span>sigma_serve</span><input type="number" id="ss" value="0.0" step="0.05" style="width:70px"></div>
  <div class="f"><span>step_m</span><input type="number" id="sm" value="0.025" step="0.005" style="width:70px"></div>
  <textarea id="json" readonly></textarea>
  <button class="primary" id="copy">copy JSON</button>
  <span class="hint" id="msg">yaw is auto-derived from the path tangent at export.</span>
 </div>
</div>
</main>
<script>
(function(){
const D = __PAYLOAD__;
const cv = document.getElementById("cv");
const gl = cv.getContext("webgl", {antialias:true, alpha:false});
function dec(b){const s=atob(b);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
const vs=`attribute vec3 p; attribute vec3 c; uniform mat4 M; uniform float ps;
 varying vec3 vc; void main(){ gl_Position=M*vec4(p,1.0); gl_PointSize=ps; vc=c; }`;
const fs=`precision mediump float; varying vec3 vc; uniform float alpha;
 void main(){ gl_FragColor=vec4(vc,alpha); }`;
function sh(t,s2){const s=gl.createShader(t);gl.shaderSource(s,s2);gl.compileShader(s);return s;}
const prog=gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,vs));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,fs));
gl.linkProgram(prog);gl.useProgram(prog);
const aP=gl.getAttribLocation(prog,"p"),aC=gl.getAttribLocation(prog,"c");
const uM=gl.getUniformLocation(prog,"M"),uPS=gl.getUniformLocation(prog,"ps"),uA=gl.getUniformLocation(prog,"alpha");
function buf(a){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);return b;}
function colorize(a,c){const o=new Float32Array(a.length);for(let i=0;i<a.length;i+=3){o[i]=c[0];o[i+1]=c[1];o[i+2]=c[2];}return o;}
// prebuild per-scene buffers (world coords, no centering; camera targets scene mean)
const S={};
for(const k in D){
  const p=new Float32Array(dec(D[k].pts).buffer), r8=dec(D[k].rgb);
  const rc=new Float32Array(r8.length);for(let i=0;i<r8.length;i++)rc[i]=r8[i]/255;
  let cx=0,cy=0,cz=0;for(let i=0;i<p.length;i+=3){cx+=p[i];cy+=p[i+1];cz+=p[i+2];}
  const n=p.length/3;
  S[k]={n:n,bp:buf(p),bc:buf(rc),centre:[cx/n,cy/n,cz/n],
    refs:D[k].refs.map(t=>{const a=new Float32Array(dec(t).buffer);
      return {n:a.length/3,bp:buf(a),bc:buf(colorize(a,[0.55,0.5,0.55]))};}),
    marks:D[k].marks.map(t=>{const a=new Float32Array(dec(t).buffer);
      return {n:a.length/3,bp:buf(a),bc:buf(colorize(a,[0.5,0.82,0.94]))};}),
    prompt:D[k].prompt};
}
let scene="left_and_center";
let yaw=-0.6,pitch=0.45,dist=9,panx=0,pany=0;
let zsel=1.5, sketching=false, showRefs=true;
let W=[];   // waypoints, world coords [x,y,z]
function cam(){
  const C=S[scene].centre;
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const ex=dist*cp*sy,ey=-dist*cp*cy,ez=dist*sp;
  const f=[-ex,-ey,-ez];let fl=Math.hypot(f[0],f[1],f[2]);f[0]/=fl;f[1]/=fl;f[2]/=fl;
  let s=[f[1],-f[0],0];const sl=Math.hypot(s[0],s[1],s[2])||1;s=[s[0]/sl,s[1]/sl,s[2]/sl];
  const u=[s[1]*f[2]-s[2]*f[1],s[2]*f[0]-s[0]*f[2],s[0]*f[1]-s[1]*f[0]];
  return {eye:[C[0]+ex,C[1]+ey,C[2]+ez],s:s,u:u,f:f,C:C};
}
function mat(){
  const c=cam(), e=c.eye, s=c.s, u=c.u, f=c.f;
  const V=[s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0,
   -(s[0]*e[0]+s[1]*e[1]+s[2]*e[2])+panx, -(u[0]*e[0]+u[1]*e[1]+u[2]*e[2])+pany,
   (f[0]*e[0]+f[1]*e[1]+f[2]*e[2]),1];
  const asp=cv.width/cv.height,n=0.05,fa=200,t=1/Math.tan(0.5);
  const P=[t/asp,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];
  const M=new Float32Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){let v=0;for(let k2=0;k2<4;k2++)v+=P[k2*4+j]*V[i*4+k2];M[i*4+j]=v;}
  return M;
}
function unproject(mx,my){
  // pixel -> world ray -> intersect plane z=zsel. pan offsets the eye in screen axes.
  const rect=cv.getBoundingClientRect();
  const nx=((mx-rect.left)/rect.width)*2-1, ny=1-((my-rect.top)/rect.height)*2;
  const c=cam(), t=1/Math.tan(0.5), asp=cv.width/cv.height;
  const e=[c.eye[0]-panx*c.s[0]-pany*c.u[0], c.eye[1]-panx*c.s[1]-pany*c.u[1],
           c.eye[2]-panx*c.s[2]-pany*c.u[2]];
  const vx=nx*asp/t, vy=ny/t;
  let d=[vx*c.s[0]+vy*c.u[0]+c.f[0], vx*c.s[1]+vy*c.u[1]+c.f[1], vx*c.s[2]+vy*c.u[2]+c.f[2]];
  if(Math.abs(d[2])<1e-6) return null;
  const tt=(zsel-e[2])/d[2];
  if(tt<=0) return null;
  return [e[0]+tt*d[0], e[1]+tt*d[1], zsel];
}
function gridBufs(){
  // faint grid on the placement plane, spanning the scene footprint
  const C=S[scene].centre, L=[];
  for(let i=-5;i<=5;i++){
    L.push(C[0]-5,C[1]+i,zsel, C[0]+5,C[1]+i,zsel);
    L.push(C[0]+i,C[1]-5,zsel, C[0]+i,C[1]+5,zsel);
  }
  const a=new Float32Array(L);
  return {n:a.length/3,bp:buf(a),bc:buf(colorize(a,[0.25,0.29,0.36]))};
}
function wpBufs(){
  if(!W.length) return null;
  const a=new Float32Array(W.length*3);
  W.forEach((p,i)=>{a[i*3]=p[0];a[i*3+1]=p[1];a[i*3+2]=p[2];});
  return {n:W.length,bp:buf(a),bc:buf(colorize(a,[1.0,0.67,0.26]))};
}
function bindDraw(o,mode,ps){
  gl.bindBuffer(gl.ARRAY_BUFFER,o.bp);gl.enableVertexAttribArray(aP);gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,o.bc);gl.enableVertexAttribArray(aC);gl.vertexAttribPointer(aC,3,gl.FLOAT,false,0,0);
  if(ps)gl.uniform1f(uPS,ps);
  gl.drawArrays(mode,0,o.n);
}
function draw(){
  const dpr=Math.min(window.devicePixelRatio||1,2);
  cv.width=cv.clientWidth*dpr;cv.height=560*dpr;
  gl.viewport(0,0,cv.width,cv.height);
  gl.clearColor(0.07,0.08,0.10,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  gl.uniformMatrix4fv(uM,false,mat());
  const sc=S[scene];
  gl.uniform1f(uA,0.55);bindDraw({n:sc.n,bp:sc.bp,bc:sc.bc},gl.POINTS,1.6*dpr);
  if(sketching){gl.uniform1f(uA,0.5);bindDraw(gridBufs(),gl.LINES,1);}
  gl.uniform1f(uA,0.95);
  if(showRefs)sc.refs.forEach(t=>{bindDraw(t,gl.LINE_STRIP,2.5*dpr);});
  sc.marks.forEach(t=>{bindDraw(t,gl.LINE_STRIP,2.5*dpr);});
  const wb=wpBufs();
  if(wb){gl.uniform1f(uA,1.0);bindDraw(wb,gl.LINE_STRIP,9*dpr);gl.drawArrays(gl.POINTS,0,wb.n);}
}
function exportJson(){
  const pts=W.map((p,i)=>{
    let yw=0;
    if(W.length>1){
      const p0=(i<W.length-1)?W[i]:W[i-1], p1=(i<W.length-1)?W[i+1]:W[i];
      yw=Math.atan2(p1[1]-p0[1],p1[0]-p0[0]);
    }
    return [+p[0].toFixed(3),+p[1].toFixed(3),+p[2].toFixed(3),+yw.toFixed(3)];
  });
  const o={points:pts,
    prompt_after:document.getElementById("prompt").value,
    enter_radius:+document.getElementById("er").value,
    step_m:+document.getElementById("sm").value,
    sigma_serve:+document.getElementById("ss").value,
    end_margin_m:0.1};
  document.getElementById("json").value=JSON.stringify(o,null,1);
  document.getElementById("np").textContent=W.length;
}
let drag=null,moved=false;
cv.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,sh:e.shiftKey};moved=false;});
window.addEventListener("mouseup",e=>{
  if(drag&&!moved&&sketching&&e.target===cv){
    const p=unproject(e.clientX,e.clientY);
    if(p){W.push(p);exportJson();draw();}
  }
  drag=null;
});
window.addEventListener("mousemove",e=>{
  if(!drag)return;
  const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
  if(Math.abs(dx)+Math.abs(dy)>3)moved=true;
  drag.x=e.clientX;drag.y=e.clientY;
  if(drag.sh){panx+=dx*0.004;pany-=dy*0.004;}
  else{yaw+=dx*0.006;pitch=Math.max(-1.4,Math.min(1.4,pitch+dy*0.006));}
  draw();
});
cv.addEventListener("wheel",e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.0012);
  dist=Math.max(1.5,Math.min(60,dist));draw();},{passive:false});
document.querySelectorAll('input[name=sc]').forEach(r=>r.addEventListener("change",()=>{
  scene=r.value;W=[];document.getElementById("prompt").value=S[scene].prompt;exportJson();draw();}));
document.getElementById("sketchmode").addEventListener("change",e=>{
  sketching=e.target.checked;cv.classList.toggle("sketching",sketching);draw();});
document.getElementById("zsl").addEventListener("input",e=>{
  zsel=+e.target.value;document.getElementById("zv").textContent=zsel.toFixed(2);draw();});
document.getElementById("showrefs").addEventListener("change",e=>{showRefs=e.target.checked;draw();});
document.getElementById("topview").addEventListener("click",()=>{
  pitch=1.52;dist=7.5;panx=0;pany=0;draw();});
document.getElementById("undo").addEventListener("click",()=>{W.pop();exportJson();draw();});
document.getElementById("clear").addEventListener("click",()=>{W=[];exportJson();draw();});
["prompt","er","ss","sm"].forEach(id=>document.getElementById(id).addEventListener("input",exportJson));
document.getElementById("copy").addEventListener("click",()=>{
  const ta=document.getElementById("json");ta.select();
  try{navigator.clipboard.writeText(ta.value);}catch(_){document.execCommand("copy");}
  document.getElementById("msg").textContent="copied.";});
document.getElementById("prompt").value=S[scene].prompt;
window.addEventListener("resize",draw);
exportJson();draw();
})();
</script>
"""
out = f"{SP}/sketchpad.html"
open(out, "w").write(page.replace("__PAYLOAD__", J))
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
