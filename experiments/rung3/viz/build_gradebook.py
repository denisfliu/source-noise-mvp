"""Gradebook: hand-evaluation tool for every paper-relevant rollout (2026-09-02).

Point cloud + one trajectory at a time + scrubber + the automatic judge/clearance verdicts
verbatim, with pass / fail / unsure buttons and notes. Grades persist in the artifact's
shared db (collection `grades`, doc id = trajectory file stem) and in localStorage; the
page can hand back a grades.json. Read grades back from the session with
`Artifact read_db collection=grades`.

  python3 build_gradebook.py          (writes gradebook.html next to this file)

Scope: the arms and rows the paper draft cites (atomics x 8 arms, unguided compounds,
every sketch row incl. the xswap re-attribution, pin applications, the scratch-sketch
control). Retired arms (gmm/gmmmh/ctl/dsplit/gmsig/gmsig2/gmsig4) and the moved-gate
suite are deliberately excluded.
"""
import base64
import glob
import json
import os
import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__))
RD = os.path.dirname(SP)
RUN = "/home/dfliu/ctxrun"
FALSIFY = os.path.expanduser("~/code/falsify-pi")
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
SCENES = ["left", "right", "center", "left_and_center", "right_and_center"]
SAFETY = {"left": "left_gate", "right": "right_gate", "center": "center_gate",
          "left_and_center": "left_and_center", "right_and_center": "right_and_center"}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


from catalogue import AUTO, CELLS, trial_files  # noqa: E402  (shared with build_sketchreview)

cells_out, n_traj = [], 0
for c in CELLS:
    fs = trial_files(c["prefix"])
    if not fs:
        continue
    trials = []
    for f in fs:
        stem = os.path.basename(f)[:-4]
        a = np.load(f)[:, :3].astype(np.float32)
        au = AUTO.get(stem, {})
        trials.append({"name": stem, "n": int(len(a)), "xyz": b64(a),
                       "judge": au.get("judge"), "clear": au.get("clear"),
                       "minclr": au.get("minclr"), "clrstep": au.get("clrstep")})
        n_traj += 1
    sk = None
    if c["sketch"] and os.path.exists(c["sketch"]):
        sk = np.asarray(json.load(open(c["sketch"]))["points"], np.float32)[:, :3].tolist()
    cells_out.append({"id": c["id"], "label": c["label"], "arm": c["arm"], "campaign": c["campaign"],
                      "scene": c["scene"], "sketch": sk, "trials": trials})

scenes_out = {}
for s in SCENES:
    z = np.load(f"{SP}/scene_cloud_{s}.npz")
    pts, rgb = z["pts"].astype(np.float32), z["rgb"].astype(np.uint8)
    safety = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{SAFETY[s]}.yaml"))
    if "ordered_miss_gate" in safety:
        gates = [g["corners"] for g in safety["ordered_miss_gate"]["gates"]]
    else:
        gates = [safety["miss_gate"]["corners"]]
    scenes_out[s] = {"n": int(len(pts)), "pts": b64(pts), "rgb": b64(rgb), "gates": gates,
                     "goal": {"c": GOAL_C.tolist(), "h": GOAL_H.tolist()}}

DATA = json.dumps({"scenes": scenes_out, "cells": cells_out, "body_r": 0.18,
                   "built": "2026-09-02"})

HTML = r"""<title>Gate Flight Gradebook</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#0e1116;--panel:#151a21;--panel2:#1b212a;--line:#2a313b;--ink:#e6ebf2;--mut:#8d97a6;
--acc:#7cd0f0;--pass:#5fd38a;--fail:#f0705c;--unsure:#e8b84a;--sketch:#ffab42;--goal:#f8d25a;
--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 var(--sans);overflow:hidden}
button{font:inherit;color:inherit;background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--acc)}
button:focus-visible,input:focus-visible,textarea:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
button.on{border-color:var(--acc);color:var(--acc)}
#app{display:grid;grid-template-rows:48px 1fr;height:100vh}
header{display:flex;align-items:center;gap:18px;padding:0 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;font-weight:600;margin:0;letter-spacing:.01em}
header .prog{font-family:var(--mono);font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}
header .prog b{color:var(--ink);font-weight:500}
header .bar{width:160px;height:6px;background:var(--panel2);border:1px solid var(--line);border-radius:3px;overflow:hidden}
header .bar i{display:block;height:100%;background:var(--acc);width:0}
header .sp{flex:1}
#main{display:grid;grid-template-columns:270px 1fr 360px;min-height:0}
#rail{border-right:1px solid var(--line);overflow-y:auto;background:var(--panel);padding:8px 0}
#rail h2{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:12px 12px 4px;font-weight:600}
.cell{display:grid;grid-template-columns:1fr auto;gap:2px 8px;padding:6px 12px;cursor:pointer;border-left:3px solid transparent}
.cell:hover{background:var(--panel2)}
.cell.on{border-left-color:var(--acc);background:var(--panel2)}
.cell .l{font-size:13px}
.cell .a{font-size:11px;color:var(--mut);grid-column:1/-1}
.cell .t{font-family:var(--mono);font-size:11px;color:var(--mut);text-align:right;font-variant-numeric:tabular-nums}
.cell .t b{font-weight:500}
.cell .t .p{color:var(--pass)} .cell .t .f{color:var(--fail)} .cell .t .u{color:var(--unsure)}
#stage{display:grid;grid-template-rows:1fr auto;min-width:0;min-height:0}
#cv{width:100%;height:100%;display:block;cursor:grab}
#cvwrap{position:relative;min-height:0}
#ctl{border-top:1px solid var(--line);background:var(--panel);padding:8px 12px;display:grid;gap:8px}
#ctl .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
#scrub{flex:1;min-width:200px;accent-color:var(--acc)}
.stepbox{font-family:var(--mono);font-size:12px;color:var(--mut);min-width:120px;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:14px;flex-wrap:wrap;font:11px var(--mono);color:var(--mut)}
.legend label{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
.legend .sw{width:16px;height:3px;border-radius:2px;display:inline-block}
#hud{position:absolute;left:10px;top:10px;font:12px var(--mono);color:var(--mut);background:rgba(14,17,22,.75);padding:6px 9px;border-radius:6px;border:1px solid var(--line);pointer-events:none;max-width:70%}
#hud b{color:var(--ink);font-weight:500}
#insp{border-left:1px solid var(--line);background:var(--panel);overflow-y:auto;display:flex;flex-direction:column}
#insp section{padding:12px 14px;border-bottom:1px solid var(--line)}
#insp h3{margin:0 0 6px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);font-weight:600}
.trials{display:flex;flex-direction:column;gap:2px}
.trial{display:grid;grid-template-columns:34px 1fr auto;gap:8px;align-items:center;padding:5px 8px;border-radius:6px;cursor:pointer;font-family:var(--mono);font-size:12px}
.trial:hover{background:var(--panel2)}
.trial.on{background:var(--panel2);outline:1px solid var(--line)}
.trial .id{color:var(--mut)}
.chips{display:flex;gap:4px;flex-wrap:wrap}
.chip{font:11px var(--mono);padding:1px 6px;border-radius:4px;border:1px solid var(--line);color:var(--mut);white-space:nowrap}
.chip.ok{color:var(--pass);border-color:rgba(95,211,138,.4)} .chip.bad{color:var(--fail);border-color:rgba(240,112,92,.4)}
.chip.warn{color:var(--unsure);border-color:rgba(232,184,74,.4)}
.mark{width:10px;height:10px;border-radius:50%;border:1px solid var(--line);display:inline-block}
.mark.pass{background:var(--pass);border-color:var(--pass)} .mark.fail{background:var(--fail);border-color:var(--fail)}
.mark.unsure{background:var(--unsure);border-color:var(--unsure)}
.raw{font:11px/1.5 var(--mono);color:var(--mut);white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;margin:6px 0 0}
.grade{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.grade button{padding:10px 6px;font-weight:600;border-width:2px}
.grade button kbd{display:block;font:11px var(--mono);color:var(--mut);font-weight:400;margin-top:2px}
#bpass.on{border-color:var(--pass);color:var(--pass)} #bfail.on{border-color:var(--fail);color:var(--fail)} #bunsure.on{border-color:var(--unsure);color:var(--unsure)}
textarea{width:100%;min-height:64px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:8px;font:13px var(--sans);resize:vertical}
.small{font-size:12px;color:var(--mut)}
.keys{font:11px var(--mono);color:var(--mut);display:grid;grid-template-columns:auto 1fr;gap:2px 10px}
.keys kbd{color:var(--ink)}
#status{font:11px var(--mono);color:var(--mut)}
#status.ok{color:var(--pass)} #status.bad{color:var(--fail)}
.export{display:flex;gap:6px;flex-wrap:wrap}
#exportbox{display:none;margin-top:6px}
#exportbox textarea{min-height:120px;font:11px var(--mono)}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
<div id="app">
<header>
  <h1>Gate Flight Gradebook</h1>
  <span class="prog"><b id="pdone">0</b> / <span id="ptotal">0</span> graded</span>
  <span class="bar"><i id="pbar"></i></span>
  <span class="prog" id="ptally"></span>
  <span class="sp"></span>
  <button id="bungraded" title="Show only cells with ungraded trials">ungraded only</button>
  <button id="bnext" title="Jump to the next ungraded trial (N)">next ungraded</button>
  <span id="status">local only</span>
</header>
<div id="main">
  <nav id="rail" aria-label="cells"></nav>
  <div id="stage">
    <div id="cvwrap"><canvas id="cv"></canvas><div id="hud"></div></div>
    <div id="ctl">
      <div class="row">
        <button id="bplay" title="space">play</button>
        <input id="scrub" type="range" min="0" max="0" value="0" aria-label="step">
        <span class="stepbox" id="stepbox">step 0 / 0</span>
        <select id="speed" aria-label="playback speed"><option value="1">1x</option><option value="2" selected>2x</option><option value="4">4x</option><option value="8">8x</option></select>
      </div>
      <div class="row">
        <span class="legend">
          <label><input type="checkbox" id="tcloud" checked><span class="sw" style="background:#8d97a6"></span>scene cloud</label>
          <label><input type="checkbox" id="tothers" checked><span class="sw" style="background:#4b5563"></span>other trials in cell</label>
          <label><input type="checkbox" id="tgeom" checked><span class="sw" style="background:var(--acc)"></span>judge apertures + goal box</label>
          <label><input type="checkbox" id="tsketch" checked><span class="sw" style="background:var(--sketch)"></span>sketch command</label>
          <label><input type="checkbox" id="tring" checked><span class="sw" style="background:#fff"></span>0.18 m body ring</label>
        </span>
        <span class="sp" style="flex:1"></span>
        <button data-view="iso">iso</button><button data-view="top">top</button><button data-view="front">front</button><button data-view="side">side</button><button data-view="follow" id="bfollow" title="camera tracks the drone">follow</button>
      </div>
    </div>
  </div>
  <aside id="insp">
    <section>
      <h3 id="celltitle">cell</h3>
      <div class="small" id="cellsub"></div>
      <div class="trials" id="trials"></div>
    </section>
    <section>
      <h3>automatic verdicts</h3>
      <div class="chips" id="autochips"></div>
      <div class="raw" id="autoraw">select a trial</div>
    </section>
    <section>
      <h3>your verdict</h3>
      <div class="grade">
        <button id="bpass">pass<kbd>P</kbd></button>
        <button id="bfail">fail<kbd>F</kbd></button>
        <button id="bunsure">unsure<kbd>U</kbd></button>
      </div>
      <div class="row" style="display:flex;gap:6px;margin-top:6px;align-items:center">
        <button id="bclear" class="small">clear (X)</button>
        <label class="small" style="margin-left:auto"><input type="checkbox" id="autoadv" checked> advance after grading</label>
      </div>
      <textarea id="note" placeholder="note (what you saw: graze at west post, wrong-direction pass, descends before goal, ...)"></textarea>
      <div class="small" id="gradedat"></div>
    </section>
    <section>
      <h3>export</h3>
      <div class="export">
        <button id="bsave">save grades.json</button>
        <button id="bcopy">show JSON</button>
      </div>
      <div id="exportbox"><textarea id="exporttxt" readonly></textarea></div>
    </section>
    <section>
      <h3>keys</h3>
      <div class="keys">
        <kbd>P F U X</kbd><span>pass / fail / unsure / clear</span>
        <kbd>J K</kbd><span>next / previous trial</span>
        <kbd>N</kbd><span>next ungraded trial</span>
        <kbd>← →</kbd><span>scrub one step (shift: 10)</span>
        <kbd>space</kbd><span>play / pause</span>
        <kbd>Home End</kbd><span>first / last step</span>
        <kbd>drag</kbd><span>orbit · wheel zoom · shift-drag pan</span>
      </div>
    </section>
  </aside>
</div>
</div>
<script>
const D = __DATA__;
// ------------------------------------------------------------- data decode
function dec(b){const s=atob(b);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
for(const s of Object.values(D.scenes)){s.P=new Float32Array(dec(s.pts).buffer);s.C=dec(s.rgb);
  let cx=0,cy=0,cz=0;for(let i=0;i<s.n;i++){cx+=s.P[3*i];cy+=s.P[3*i+1];cz+=s.P[3*i+2];}s.centre=[cx/s.n,cy/s.n,cz/s.n];}
const CELLS=D.cells; const TRIALS=[]; // flat order
CELLS.forEach((c,ci)=>c.trials.forEach((t,ti)=>{t.ci=ci;t.ti=ti;t.P=new Float32Array(dec(t.xyz).buffer);delete t.xyz;TRIALS.push(t);}));
const byName=Object.fromEntries(TRIALS.map(t=>[t.name,t]));
// ------------------------------------------------------------- grades (local + db)
const LS='gradebook.v1'; let grades={}; try{grades=JSON.parse(localStorage.getItem(LS)||'{}');}catch(e){grades={};}
let db=null, dbUnsub=null; const statusEl=document.getElementById('status');
function setStatus(t,c){statusEl.textContent=t;statusEl.className=c||'';}
function persistLocal(){try{localStorage.setItem(LS,JSON.stringify(grades));}catch(e){}}
async function setGrade(name,patch){
  const t=byName[name]; const cur=grades[name]||{};
  const g={...cur,...patch,cell:CELLS[t.ci].id,ts:new Date().toISOString()};
  if(!g.v && !(g.note||'').trim()){delete grades[name];}else grades[name]=g;
  persistLocal(); renderAll();
  if(db){try{ if(grades[name]) await db.collection('grades').doc(name).set(grades[name]); else await db.collection('grades').doc(name).delete(); setStatus('saved to shared db','ok'); }
    catch(e){ setStatus('db write failed: '+(e&&e.code||e),'bad'); } }
}
(async()=>{
  if(!window.claude||!window.claude.use) return;
  try{ db=await claude.use('db'); }catch(e){ db=null; }
  if(!db){ setStatus('local only (no shared db in this view)'); return; }
  setStatus('shared db connected','ok');
  dbUnsub=db.collection('grades').onSnapshot(snap=>{
    let changed=false;
    snap.docChanges().forEach(ch=>{ const d=ch.doc.data(); const n=ch.doc.id;
      if(ch.type==='removed'){ if(grades[n]){delete grades[n];changed=true;} return; }
      if(!d) return; const cur=grades[n];
      if(!cur || (d.ts||'')>=(cur.ts||'')){ grades[n]={v:d.v,note:d.note||'',cell:d.cell,ts:d.ts}; changed=true; }
    });
    if(changed){persistLocal();renderAll();}
  }, e=>setStatus('db subscription ended: '+e.code,'bad'));
  // push local grades the db does not have yet (first connection from this browser)
  try{ const have=await db.collection('grades').get(); const ids=new Set(have.docs.map(d=>d.id));
    for(const [n,g] of Object.entries(grades)) if(!ids.has(n)) await db.collection('grades').doc(n).set(g); }catch(e){}
})();
// ------------------------------------------------------------- WebGL
const cv=document.getElementById('cv'); const gl=cv.getContext('webgl',{antialias:true,alpha:false});
const vs=`attribute vec3 p;attribute vec4 c;uniform mat4 M;uniform float ps;varying vec4 vc;void main(){gl_Position=M*vec4(p,1.0);gl_PointSize=ps;vc=c;}`;
const fs=`precision mediump float;varying vec4 vc;uniform float alpha;void main(){gl_FragColor=vec4(vc.rgb,vc.a*alpha);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o;}
const prog=gl.createProgram();gl.attachShader(prog,sh(gl.VERTEX_SHADER,vs));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(prog);gl.useProgram(prog);
const aP=gl.getAttribLocation(prog,'p'),aC=gl.getAttribLocation(prog,'c'),uM=gl.getUniformLocation(prog,'M'),uPS=gl.getUniformLocation(prog,'ps'),uA=gl.getUniformLocation(prog,'alpha');
gl.enableVertexAttribArray(aP);gl.enableVertexAttribArray(aC);
function buf(a){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);return b;}
function mk(pos,rgba){return {n:pos.length/3,bp:buf(pos),bc:buf(rgba)};}
function solid(pos,col){const c=new Float32Array(pos.length/3*4);for(let i=0;i<pos.length/3;i++){c.set(col,4*i);}return mk(pos,c);}
function centred(P,centre){const o=new Float32Array(P.length);for(let i=0;i<P.length;i+=3){o[i]=P[i]-centre[0];o[i+1]=P[i+1]-centre[1];o[i+2]=P[i+2]-centre[2];}return o;}
const sceneGL={};
function sceneBuffers(name){ if(sceneGL[name]) return sceneGL[name]; const s=D.scenes[name];
  const P=centred(s.P,s.centre); const C=new Float32Array(s.n*4); for(let i=0;i<s.n;i++){C[4*i]=s.C[3*i]/255;C[4*i+1]=s.C[3*i+1]/255;C[4*i+2]=s.C[3*i+2]/255;C[4*i+3]=1;}
  const cloud=mk(P,C);
  const geom=[]; const cyan=[0.49,0.82,0.94,1], gold=[0.97,0.82,0.35,1];
  s.gates.forEach(g=>{const pts=g.concat([g[0]]).flat(); geom.push({...solid(centred(new Float32Array(pts),s.centre),cyan),mode:'strip'});});
  const c=s.goal.c,h=s.goal.h; const corners=[];for(const sx of[-1,1])for(const sy of[-1,1])for(const sz of[-1,1])corners.push([c[0]+sx*h[0],c[1]+sy*h[1],c[2]+sz*h[2]]);
  const E=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]]; const seg=[];E.forEach(([a,b])=>seg.push(...corners[a],...corners[b]));
  geom.push({...solid(centred(new Float32Array(seg),s.centre),gold),mode:'lines'});
  return sceneGL[name]={cloud,geom};
}
// ------------------------------------------------------------- state
let ci=0, ti=0, step=0, playing=false, follow=false, filterUngraded=false;
let cam={yaw:-0.6,pitch:0.45,dist:5.5,panx:0,pany:0};
let cellGL=null; // {trials:[{line,grad}], sketch}
function cell(){return CELLS[ci];} function trial(){return cell().trials[ti];}
function loadCell(){
  const c=cell(); const s=D.scenes[c.scene];
  cellGL={trials:c.trials.map(t=>{const P=centred(t.P,s.centre); const n=t.n;
      const C=new Float32Array(n*4); for(let i=0;i<n;i++){const f=i/Math.max(1,n-1); // time gradient: cyan -> magenta
        C[4*i]=0.35+0.6*f;C[4*i+1]=0.85-0.55*f;C[4*i+2]=0.95;C[4*i+3]=1;}
      return {line:mk(P,C),faint:solid(P,[0.42,0.47,0.55,1])};}),
    sketch:c.sketch?solid(centred(new Float32Array(c.sketch.flat()),s.centre),[1,0.67,0.26,1]):null};
  step=0; document.getElementById('scrub').max=Math.max(0,trial().n-1);
}
// ring around the drone at radius body_r (horizontal), + vertical ring
const RING=(()=>{const N=48,o=[];for(let i=0;i<=N;i++){const a=2*Math.PI*i/N;o.push(Math.cos(a),Math.sin(a),0);}
  for(let i=0;i<=N;i++){const a=2*Math.PI*i/N;o.push(Math.cos(a),0,Math.sin(a));}return new Float32Array(o);})();
const ringBuf=solid(RING,[1,1,1,0.9]); const dyn=gl.createBuffer(); const dynC=gl.createBuffer();
function mat(target){
  const {yaw,pitch,dist,panx,pany}=cam; const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const ex=target[0]+dist*cp*sy, ey=target[1]-dist*cp*cy, ez=target[2]+dist*sp;
  const f=[target[0]-ex,target[1]-ey,target[2]-ez]; const fl=Math.hypot(...f)||1; f.forEach((v,i)=>f[i]=v/fl);
  let up=[0,0,1]; if(Math.abs(f[2])>0.999) up=[0,1,0];
  let s=[f[1]*up[2]-f[2]*up[1],f[2]*up[0]-f[0]*up[2],f[0]*up[1]-f[1]*up[0]]; const sl=Math.hypot(...s)||1; s=s.map(v=>v/sl);
  const u=[s[1]*f[2]-s[2]*f[1],s[2]*f[0]-s[0]*f[2],s[0]*f[1]-s[1]*f[0]];
  const V=[s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0, -(s[0]*ex+s[1]*ey+s[2]*ez)+panx, -(u[0]*ex+u[1]*ey+u[2]*ez)+pany, (f[0]*ex+f[1]*ey+f[2]*ez),1];
  const asp=cv.width/cv.height,n=0.05,fa=200,t=1/Math.tan(0.5*0.9);
  const P=[t/asp,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0]; const M=new Float32Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){let v=0;for(let k=0;k<4;k++)v+=P[k*4+j]*V[i*4+k];M[i*4+j]=v;} return M;
}
function bind(o){gl.bindBuffer(gl.ARRAY_BUFFER,o.bp);gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,o.bc);gl.vertexAttribPointer(aC,4,gl.FLOAT,false,0,0);}
function pos(t,i){const s=D.scenes[cell().scene].centre;return [t.P[3*i]-s[0],t.P[3*i+1]-s[1],t.P[3*i+2]-s[2]];}
function draw(){
  const dpr=Math.min(devicePixelRatio||1,2); const w=cv.clientWidth,h=cv.clientHeight; if(cv.width!==w*dpr||cv.height!==h*dpr){cv.width=w*dpr;cv.height=h*dpr;}
  gl.viewport(0,0,cv.width,cv.height); gl.clearColor(0.055,0.067,0.086,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  const t=trial(); const sb=sceneBuffers(cell().scene); const here=pos(t,Math.min(step,t.n-1));
  gl.uniformMatrix4fv(uM,false,mat(follow?here:[0,0,0]));
  if(T.cloud.checked){gl.uniform1f(uPS,1.7*dpr);gl.uniform1f(uA,0.6);bind(sb.cloud);gl.drawArrays(gl.POINTS,0,sb.cloud.n);}
  gl.uniform1f(uA,1);
  if(T.geom.checked){gl.uniform1f(uPS,2*dpr);sb.geom.forEach(g=>{bind(g);gl.drawArrays(g.mode==='lines'?gl.LINES:gl.LINE_STRIP,0,g.n);});}
  if(T.sketch.checked&&cellGL.sketch){bind(cellGL.sketch);gl.uniform1f(uPS,7*dpr);gl.drawArrays(gl.LINE_STRIP,0,cellGL.sketch.n);gl.drawArrays(gl.POINTS,0,cellGL.sketch.n);}
  if(T.others.checked){gl.uniform1f(uA,0.55);cellGL.trials.forEach((o,i)=>{if(i===ti)return;bind(o.faint);gl.drawArrays(gl.LINE_STRIP,0,o.n||o.faint.n);});gl.uniform1f(uA,1);}
  const cur=cellGL.trials[ti]; bind(cur.line); gl.uniform1f(uPS,3.2*dpr); gl.drawArrays(gl.LINE_STRIP,0,cur.line.n); gl.drawArrays(gl.POINTS,0,cur.line.n);
  // drone marker + ring
  const r=D.body_r; const ring=new Float32Array(RING.length); for(let i=0;i<RING.length;i+=3){ring[i]=here[0]+r*RING[i];ring[i+1]=here[1]+r*RING[i+1];ring[i+2]=here[2]+r*RING[i+2];}
  const past=step>=t.n-1;
  if(T.ring.checked){gl.bindBuffer(gl.ARRAY_BUFFER,dyn);gl.bufferData(gl.ARRAY_BUFFER,ring,gl.DYNAMIC_DRAW);gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,ringBuf.bc);gl.vertexAttribPointer(aC,4,gl.FLOAT,false,0,0);gl.uniform1f(uPS,2*dpr);gl.drawArrays(gl.LINE_STRIP,0,49);gl.drawArrays(gl.LINE_STRIP,49,49);}
  // marker point (white) and min-clearance point (red) if known
  const mk2=[...here]; const col=[1,1,1,1]; if(t.clrstep!=null&&t.clrstep<t.n){mk2.push(...pos(t,t.clrstep));col.push(0.94,0.44,0.36,1);}
  gl.bindBuffer(gl.ARRAY_BUFFER,dyn);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(mk2),gl.DYNAMIC_DRAW);gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,dynC);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(col),gl.DYNAMIC_DRAW);gl.vertexAttribPointer(aC,4,gl.FLOAT,false,0,0);
  gl.uniform1f(uPS,11*dpr);gl.drawArrays(gl.POINTS,0,mk2.length/3);
  // HUD
  const raw=t.P; const i=Math.min(step,t.n-1); const clr=(t.minclr!=null)?`min clearance <b>${t.minclr.toFixed(3)} m</b> @${t.clrstep}`:'no clearance score';
  document.getElementById('hud').innerHTML=`<b>${t.name}</b> · step <b>${i}</b>/${t.n-1} · xyz <b>${raw[3*i].toFixed(2)}, ${raw[3*i+1].toFixed(2)}, ${raw[3*i+2].toFixed(2)}</b><br>${clr}${past?' · end':''}`;
  document.getElementById('stepbox').textContent=`step ${i} / ${t.n-1}`;
}
const T={cloud:tcloud,others:tothers,geom:tgeom,sketch:tsketch,ring:tring}; Object.values(T).forEach(x=>x.addEventListener('change',draw));
// camera interaction
let drag=null;
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,sh:e.shiftKey};cv.style.cursor='grabbing';});
addEventListener('mouseup',()=>{drag=null;cv.style.cursor='grab';});
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
  if(drag.sh){cam.panx+=dx*0.004;cam.pany-=dy*0.004;}else{cam.yaw+=dx*0.006;cam.pitch=Math.max(-1.5,Math.min(1.5,cam.pitch+dy*0.006));}draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();cam.dist*=Math.exp(e.deltaY*0.0012);cam.dist=Math.max(0.6,Math.min(60,cam.dist));draw();},{passive:false});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{const v=b.dataset.view;
  if(v==='follow'){follow=!follow;b.classList.toggle('on',follow);if(follow){cam.dist=Math.min(cam.dist,4);}draw();return;}
  cam.panx=0;cam.pany=0; if(v==='iso'){cam.yaw=-0.6;cam.pitch=0.45;cam.dist=5.5;} if(v==='top'){cam.yaw=0;cam.pitch=1.5;cam.dist=6;}
  if(v==='front'){cam.yaw=0;cam.pitch=0.05;cam.dist=6;} if(v==='side'){cam.yaw=Math.PI/2;cam.pitch=0.05;cam.dist=6;} draw();}));
// playback
const scrub=document.getElementById('scrub'); scrub.addEventListener('input',()=>{step=+scrub.value;draw();});
let last=0; function tick(ts){ if(!playing) return; const sp=+document.getElementById('speed').value; if(ts-last>1000/(30*sp)){last=ts; step=Math.min(trial().n-1,step+1); scrub.value=step; draw(); if(step>=trial().n-1){playing=false;bplay.textContent='play';}} requestAnimationFrame(tick);}
function togglePlay(){playing=!playing;bplay.textContent=playing?'pause':'play';if(playing){if(step>=trial().n-1)step=0;requestAnimationFrame(tick);}}
bplay.addEventListener('click',togglePlay);
// ------------------------------------------------------------- UI render
function tally(c){let p=0,f=0,u=0;c.trials.forEach(t=>{const g=grades[t.name];if(!g||!g.v)return;if(g.v==='pass')p++;else if(g.v==='fail')f++;else u++;});return {p,f,u,g:p+f+u,n:c.trials.length};}
function autoTally(c){let r=0,cl=0,nr=0,ncl=0;c.trials.forEach(t=>{if(t.judge){nr++;if(/SUCCESS=True/.test(t.judge)&&!/wrong_dir=[1-9]/.test(t.judge))r++;}if(t.clear){ncl++;if(/CLEAN=True/.test(t.clear))cl++;}});return {r,nr,cl,ncl};}
function renderRail(){
  const rail=document.getElementById('rail'); rail.innerHTML='';
  let camp=null;
  CELLS.forEach((c,i)=>{const tl=tally(c); if(filterUngraded&&tl.g>=tl.n) return;
    if(c.campaign!==camp){camp=c.campaign;const h=document.createElement('h2');h.textContent=camp;rail.appendChild(h);}
    const a=autoTally(c); const el=document.createElement('div'); el.className='cell'+(i===ci?' on':''); el.tabIndex=0;
    el.innerHTML=`<span class="l">${c.label}</span><span class="t"><b>${tl.g}</b>/${tl.n}${tl.g?` <span class="p">${tl.p}</span>·<span class="f">${tl.f}</span>·<span class="u">${tl.u}</span>`:''}</span>
      <span class="a">${c.arm} · auto: route ${a.nr?a.r+'/'+a.nr:'–'} · clear ${a.ncl?a.cl+'/'+a.ncl:'–'}</span>`;
    el.addEventListener('click',()=>{selectTrial(i,0);}); el.addEventListener('keydown',e=>{if(e.key==='Enter')selectTrial(i,0);});
    rail.appendChild(el);});
  const on=rail.querySelector('.cell.on'); if(on) on.scrollIntoView({block:'nearest'});
}
function chipsFor(t){const out=[]; if(!t.judge&&!t.clear){out.push(['no automatic score','']);return out;}
  if(t.judge){const m=t.judge.match(/transit=(\w+)@(\w+)/); if(m) out.push([`transit ${m[1]==='True'?'@'+m[2]:'no'}`,m[1]==='True'?'ok':'bad']);
    const g=t.judge.match(/gates=(\d)\/(\d)/); if(g) out.push([`gates ${g[1]}/${g[2]}`,g[1]===g[2]?'ok':'bad']);
    const w=t.judge.match(/wrong_dir=(\d+)/); if(w) out.push([`wrong-dir ${w[1]}`,w[1]==='0'?'ok':'bad']);
    const go=t.judge.match(/goal=(\w+)/); if(go) out.push([`goal ${go[1]==='True'?'yes':'no'}`,go[1]==='True'?'ok':'bad']);
    const dw=t.judge.match(/dwell=(\S+)/); if(dw) out.push([`dwell ${dw[1]}`,dw[1]==='None'?'bad':'ok']);
    const s=/SUCCESS=True/.test(t.judge); out.push([`judge ${s?'SUCCESS':'FAIL'}`,s?'ok':'bad']);}
  if(t.clear){const c=/CLEAN=True/.test(t.clear); out.push([`clearance ${t.minclr!=null?t.minclr.toFixed(3)+' m':''} ${c?'clean':'< 0.18'}`,c?'ok':'bad']);}
  return out;}
function renderInspector(){
  const c=cell(),t=trial(); document.getElementById('celltitle').textContent=`${c.campaign} · ${c.label}`;
  document.getElementById('cellsub').textContent=`${c.arm} · scene ${c.scene} · ${c.trials.length} trials`;
  const tr=document.getElementById('trials'); tr.innerHTML='';
  c.trials.forEach((x,i)=>{const g=grades[x.name]; const el=document.createElement('div'); el.className='trial'+(i===ti?' on':''); el.tabIndex=0;
    const ch=chipsFor(x).filter(([l])=>/^judge|^clearance|^no auto/.test(l)).map(([l,k])=>`<span class="chip ${k}">${l.replace('judge ','').replace('clearance ','clr ')}</span>`).join('');
    el.innerHTML=`<span class="id">#${x.name.split('_').pop()}</span><span class="chips">${ch}</span><span class="mark ${g&&g.v||''}" title="${g&&g.v||'ungraded'}"></span>`;
    el.addEventListener('click',()=>selectTrial(ci,i)); el.addEventListener('keydown',e=>{if(e.key==='Enter')selectTrial(ci,i);}); tr.appendChild(el);});
  document.getElementById('autochips').innerHTML=chipsFor(t).map(([l,k])=>`<span class="chip ${k}">${l}</span>`).join('');
  document.getElementById('autoraw').textContent=[t.judge?'judge:  '+t.judge:null,t.clear?'clear:  '+t.clear:null].filter(Boolean).join('\n')||'no automatic verdict recorded for this trajectory';
  const g=grades[t.name]||{}; ['pass','fail','unsure'].forEach(v=>document.getElementById('b'+v).classList.toggle('on',g.v===v));
  const note=document.getElementById('note'); if(note.value!==(g.note||'')) note.value=g.note||'';
  document.getElementById('gradedat').textContent=g.ts?`graded ${new Date(g.ts).toLocaleString()}`:'ungraded';
}
function renderHeader(){const total=TRIALS.length; let done=0,p=0,f=0,u=0; TRIALS.forEach(t=>{const g=grades[t.name];if(g&&g.v){done++;if(g.v==='pass')p++;else if(g.v==='fail')f++;else u++;}});
  pdone.textContent=done; ptotal.textContent=total; pbar.style.width=(100*done/total)+'%'; ptally.innerHTML=`<span style="color:var(--pass)">${p} pass</span> · <span style="color:var(--fail)">${f} fail</span> · <span style="color:var(--unsure)">${u} unsure</span>`;}
function renderAll(){renderHeader();renderRail();renderInspector();}
function selectTrial(c,t){const changed=c!==ci; ci=c; ti=Math.max(0,Math.min(t,cell().trials.length-1)); playing=false; bplay.textContent='play'; if(changed||!cellGL) loadCell(); step=0; scrub.max=trial().n-1; scrub.value=0; renderAll(); draw();}
function nextTrial(d){let c=ci,t=ti+d; if(t>=cell().trials.length){c=(ci+1)%CELLS.length;t=0;} if(t<0){c=(ci-1+CELLS.length)%CELLS.length;t=CELLS[c].trials.length-1;} selectTrial(c,t);}
function nextUngraded(){const start=TRIALS.indexOf(trial()); for(let k=1;k<=TRIALS.length;k++){const x=TRIALS[(start+k)%TRIALS.length]; const g=grades[x.name]; if(!g||!g.v){selectTrial(x.ci,x.ti);return;}} setStatus('everything is graded','ok');}
// grading controls
['pass','fail','unsure'].forEach(v=>document.getElementById('b'+v).addEventListener('click',()=>grade(v)));
function grade(v){const t=trial(); const cur=grades[t.name]||{}; setGrade(t.name,{v:cur.v===v?null:v,note:document.getElementById('note').value}); if(cur.v!==v&&document.getElementById('autoadv').checked) setTimeout(()=>nextUngradedFrom(),120);}
function nextUngradedFrom(){ // prefer next trial in this cell, then next ungraded anywhere
  const c=cell(); for(let i=ti+1;i<c.trials.length;i++){const g=grades[c.trials[i].name]; if(!g||!g.v){selectTrial(ci,i);return;}} nextUngraded(); }
bclear.addEventListener('click',()=>setGrade(trial().name,{v:null,note:''}));
let noteTimer=null; document.getElementById('note').addEventListener('input',e=>{clearTimeout(noteTimer); noteTimer=setTimeout(()=>{const t=trial(); const cur=grades[t.name]||{}; setGrade(t.name,{v:cur.v||null,note:e.target.value});},600);});
bungraded.addEventListener('click',()=>{filterUngraded=!filterUngraded;bungraded.classList.toggle('on',filterUngraded);renderRail();});
bnext.addEventListener('click',nextUngraded);
// export
function exportJSON(){const rows=Object.entries(grades).map(([name,g])=>({name,cell:g.cell,verdict:g.v||null,note:g.note||'',ts:g.ts})).sort((a,b)=>a.name<b.name?-1:1);
  return JSON.stringify({built:D.built,exported:new Date().toISOString(),n_graded:rows.filter(r=>r.verdict).length,n_total:TRIALS.length,grades:rows},null,1);}
bsave.addEventListener('click',async()=>{const dl=window.claude&&window.claude.use?await claude.use('downloads'):null; const txt=exportJSON();
  if(!dl){exportbox.style.display='block';exporttxt.value=txt;setStatus('downloads unavailable here; JSON shown below');return;}
  try{await dl.save({filename:'grades.json',data:txt});setStatus('grades.json saved','ok');}catch(e){setStatus('save '+(e&&e.code||'failed'),'bad');}});
bcopy.addEventListener('click',()=>{exportbox.style.display=exportbox.style.display==='block'?'none':'block';exporttxt.value=exportJSON();});
// keys
addEventListener('keydown',e=>{ if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'&&e.target.type!=='range') return; const k=e.key.toLowerCase();
  if(k==='p')grade('pass'); else if(k==='f')grade('fail'); else if(k==='u')grade('unsure'); else if(k==='x')setGrade(trial().name,{v:null,note:''});
  else if(k==='j'||k==='arrowdown'){e.preventDefault();nextTrial(1);} else if(k==='k'||k==='arrowup'){e.preventDefault();nextTrial(-1);}
  else if(k==='n')nextUngraded(); else if(k===' '){e.preventDefault();togglePlay();}
  else if(k==='arrowright'){e.preventDefault();step=Math.min(trial().n-1,step+(e.shiftKey?10:1));scrub.value=step;draw();}
  else if(k==='arrowleft'){e.preventDefault();step=Math.max(0,step-(e.shiftKey?10:1));scrub.value=step;draw();}
  else if(k==='home'){step=0;scrub.value=0;draw();} else if(k==='end'){step=trial().n-1;scrub.value=step;draw();}
});
addEventListener('resize',draw);
// boot: first ungraded trial, else first trial
(function(){let x=TRIALS.find(t=>!(grades[t.name]&&grades[t.name].v))||TRIALS[0]; ci=x.ci; ti=x.ti; loadCell(); renderAll(); draw();})();
</script>
"""

out = f"{SP}/gradebook.html"
open(out, "w").write(HTML.replace("__DATA__", DATA))
sz = os.path.getsize(out) / 1e6
print(f"wrote {out}: {len(cells_out)} cells, {n_traj} trajectories, {sz:.1f} MB")
