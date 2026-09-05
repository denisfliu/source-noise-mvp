"""Sketch review: every drawn command (deduplicated by waypoints) over its scene cloud, with
the drawing's OWN clearance to the gate cloud (the same geometry gate_clearance.py scores
flights against), where the line pierces each judge aperture and by what margin, the
flights it produced as faint context, and good / bad / redraw grading with notes.

Grades persist in the artifact db (collection `sketches`, doc id = drawing id) and in
localStorage; read back with `Artifact read_db collection=sketches`.

  /home/dfliu/code/tv/bin/python build_sketchreview.py   (needs torch: gate cloud)
"""
import base64
import glob
import hashlib
import json
import os
import sys

import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__))
RD = os.path.dirname(SP)
sys.path.insert(0, SP)
sys.path.insert(0, RD)
from catalogue import AUTO, CELLS, trial_files  # noqa: E402

from sketch_geom import BODY_R, STEP, evaluate, gates_for  # noqa: E402

GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
SCENES = ["left", "right", "center", "left_and_center", "right_and_center"]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


# ------------------------------------------------------------------ drawings
files = sorted(f for f in glob.glob(f"{RD}/sketch_*.json") if "_mg_" not in f)
groups = {}
for f in files:
    d = json.load(open(f)); pts = np.asarray(d["points"], float)[:, :3]
    h = hashlib.md5(json.dumps(np.round(pts, 4).tolist()).encode()).hexdigest()[:8]
    g = groups.setdefault(h, {"pts": pts, "files": []})
    g["files"].append({"file": os.path.basename(f), "sigma": d.get("sigma_serve", 0.0), "step_m": d.get("step_m"),
                       "carrot": d.get("carrot", 0), "enter_radius": d.get("enter_radius"), "prompt_after": d.get("prompt_after")})

LABELS = {"sketch_cmpl.json": "Corrective machine sketch, L->C",
          "sketch_cmpl_denis.json": "Hand-drawn, L->C",
          "sketch_cmpl_min4.json": "4-click minimal, L->C",
          "sketch_cmpl_min4v2.json": "4-click minimal v2, L->C (point 3 moved)",
          "sketch_cmpr_denis_r2.json": "Hand-drawn r2, R->C (points 2, 4, 5 moved)",
          "sketch_cmpr_denis.json": "Hand-drawn round 1, R->C",
          "sketch_cmpr_denis_r1.json": "Hand-drawn r1, R->C",
          "sketch_cmpr_min4.json": "4-click minimal, R->C",
          "sketch_cmpr_min5.json": "5-click, R->C",
          "sketch_cmpr_min5f.json": "5-click corrected waypoint, R->C",
          "sketch_fig8.json": "Figure-eight",
          "sketch_orbit.json": "Orbit, 1.5 loops",
          "sketch_realdemo_right.json": "Real demo replayed as sketch, right gate",
          "sketch_tempo06.json": "Tempo route (0.6x / 1.0x / 1.5x)"}
ORDER = list(LABELS)



def scene_of(fname):
    return "left_and_center" if "cmpl" in fname else "right_and_center" if "cmpr" in fname else "right"


cell_by_sketch = {}
for c in CELLS:
    if c["sketch"]:
        cell_by_sketch.setdefault(os.path.basename(c["sketch"]), []).append(c)

drawings = []
for h, g in groups.items():
    primary = min(g["files"], key=lambda x: ORDER.index(x["file"]) if x["file"] in ORDER else 99)["file"]
    scene = scene_of(primary)
    P = g["pts"]; ev = evaluate(P, scene); dense, dist, seg = ev["dense"], ev["dist"], ev["seg"]
    i = int(dist.argmin())
    seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
    # flights that used any variant of this drawing
    flights = []
    for v in g["files"]:
        for c in cell_by_sketch.get(v["file"], []):
            fs = trial_files(c["id"])
            if not fs:
                continue
            trials, route, clean, nr, nc = [], 0, 0, 0, 0
            for f in fs:
                stem = os.path.basename(f)[:-4]; au = AUTO.get(stem, {})
                a = np.load(f)[:, :3].astype(np.float32); trials.append({"n": int(len(a)), "xyz": b64(a)})
                if au.get("judge"):
                    nr += 1; route += ("SUCCESS=True" in au["judge"] and "wrong_dir=0" in au["judge"]) or ("SUCCESS=True" in au["judge"] and "wrong_dir" not in au["judge"])
                if au.get("clear"):
                    nc += 1; clean += "CLEAN=True" in au["clear"]
            flights.append({"cell": c["id"], "label": f"{c['label']} · {c['arm']}", "variant": v["file"],
                            "arm": c["arm"], "best": c["arm"].startswith("xswap"),
                            "route": f"{route}/{nr}" if nr else "–", "clean": f"{clean}/{nc}" if nc else "–",
                            "nroute": [int(route), nr], "nclean": [int(clean), nc], "trials": trials})
    flights.sort(key=lambda f: (not f["best"], f["arm"], f["variant"]))
    best = [f for f in flights if f["best"]]
    best_tally = {"route": [sum(f["nroute"][0] for f in best), sum(f["nroute"][1] for f in best)],
                  "clean": [sum(f["nclean"][0] for f in best), sum(f["nclean"][1] for f in best)],
                  "cells": [{"label": f["label"], "route": f["route"], "clean": f["clean"]} for f in best]}
    drawings.append({
        "id": primary[:-5], "label": LABELS.get(primary, primary), "scene": scene, "files": g["files"],
        "pts": P.round(4).tolist(), "seglen": seglen.round(3).tolist(), "length": round(float(seglen.sum()), 3),
        "dense": b64(dense.astype(np.float32)), "ndense": int(len(dense)), "dist": b64(dist.astype(np.float32)),
        "minclr": round(float(dist[i]), 3), "minclr_xyz": dense[i].round(3).tolist(), "minclr_seg": int(seg[i]),
        "under": int((dist < BODY_R).sum()) * STEP,
        "crossings": ev["crossings"], "flights": flights, "best": best_tally})
# Only the drawings the paper's results rest on (Denis, 2026-09-02), tagged with the element they support.
PAPER = {
    "sketch_cmpl_min4": "Table 3 row 1: 4-Click Route Sketch, Compound (Left -> Center) — sigma 0 and sigma 0.5 variants",
    "sketch_cmpl_min4v2": "Table 3 row 1 REPLACEMENT: point 3 moved so segment 2->3 pierces the center aperture at its midpoint",
    "sketch_cmpr_denis_r1": "Table 3 row 2: Hand-Drawn Sketch, Compound (Right -> Center)",
    "sketch_cmpr_denis_r2": "Table 3 row 2 REPLACEMENT: point 2 at the right-aperture midpoint; points 4, 5 forward on the center midline",
    "sketch_cmpl_denis": "Table 2 compound column (pinned arms 10/10) and ablation item 1 (same sketch through unpinned pi0: 0/5)",
    "sketch_orbit": "Table 3: OOD Motion, Orbit; ablation item 1 (orbit through unpinned pi0, 6.2x tracking error)",
    "sketch_fig8": "Table 3: OOD Motion, Figure-8",
    "sketch_tempo06": "Section 5.4 bullet: Tempo Modulation 0.6x / 1.0x / 1.5x",
}
drawings = [d for d in drawings if d["id"] in PAPER]
for d in drawings:
    d["paper"] = PAPER[d["id"]]
drawings.sort(key=lambda d: list(PAPER).index(d["id"]))

scenes_out = {}
for s in SCENES:
    z = np.load(f"{SP}/scene_cloud_{s}.npz")
    pts, rgb = z["pts"].astype(np.float32), z["rgb"].astype(np.uint8)
    scenes_out[s] = {"n": int(len(pts)), "pts": b64(pts), "rgb": b64(rgb),
                     "gates": [g.tolist() for _, g in gates_for(s)], "goal": {"c": GOAL_C.tolist(), "h": GOAL_H.tolist()}}
DATA = json.dumps({"scenes": scenes_out, "drawings": drawings, "body_r": BODY_R, "step": STEP, "built": "2026-09-02"})

HTML = r"""<title>Sketch Review</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#0e1116;--panel:#151a21;--panel2:#1b212a;--line:#2a313b;--ink:#e6ebf2;--mut:#8d97a6;
--acc:#7cd0f0;--good:#5fd38a;--bad:#f0705c;--redraw:#e8b84a;--sketch:#ffab42;--goal:#f8d25a;
--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box} html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 var(--sans);overflow:hidden}
button{font:inherit;color:inherit;background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--acc)} button.on{border-color:var(--acc);color:var(--acc)}
button:focus-visible,input:focus-visible,textarea:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
#app{display:grid;grid-template-rows:48px 1fr;height:100vh}
header{display:flex;align-items:center;gap:18px;padding:0 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;font-weight:600;margin:0} .sp{flex:1}
.prog{font-family:var(--mono);font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums} .prog b{color:var(--ink);font-weight:500}
#main{display:grid;grid-template-columns:290px 1fr 400px;min-height:0}
#rail{border-right:1px solid var(--line);overflow-y:auto;background:var(--panel);padding:8px 0}
#rail h2{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:12px 12px 4px;font-weight:600}
.dr{display:grid;grid-template-columns:1fr auto;gap:2px 8px;padding:7px 12px;cursor:pointer;border-left:3px solid transparent}
.dr:hover{background:var(--panel2)} .dr.on{border-left-color:var(--acc);background:var(--panel2)}
.dr .l{font-size:13px} .dr .a{font-size:11px;color:var(--mut);grid-column:1/-1;font-family:var(--mono)}
.dr .a .bad{color:var(--bad)} .dr .a .ok{color:var(--good)}
.mark{width:10px;height:10px;border-radius:50%;border:1px solid var(--line);display:inline-block;align-self:center}
.mark.good{background:var(--good);border-color:var(--good)} .mark.bad{background:var(--bad);border-color:var(--bad)} .mark.redraw{background:var(--redraw);border-color:var(--redraw)}
#stage{display:grid;grid-template-rows:1fr auto;min-width:0;min-height:0}
#cvwrap{position:relative;min-height:0;overflow:hidden} #cv{width:100%;height:100%;display:block;cursor:grab}
.wp{position:absolute;transform:translate(-50%,-140%);font:11px var(--mono);color:#0e1116;background:var(--sketch);border-radius:9px;padding:0 5px;pointer-events:none;font-weight:500}
.xp{position:absolute;transform:translate(-50%,40%);font:10px var(--mono);color:var(--acc);pointer-events:none;white-space:nowrap}
#hud{position:absolute;left:10px;top:10px;font:12px var(--mono);color:var(--mut);background:rgba(14,17,22,.8);padding:6px 9px;border-radius:6px;border:1px solid var(--line);pointer-events:none;max-width:70%}
#hud b{color:var(--ink);font-weight:500}
#ctl{border-top:1px solid var(--line);background:var(--panel);padding:8px 12px;display:grid;gap:8px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.legend{display:flex;gap:14px;flex-wrap:wrap;font:11px var(--mono);color:var(--mut)}
.legend label{display:inline-flex;gap:6px;align-items:center;cursor:pointer} .legend .sw{width:16px;height:3px;border-radius:2px;display:inline-block}
#insp{border-left:1px solid var(--line);background:var(--panel);overflow-y:auto}
#insp section{padding:12px 14px;border-bottom:1px solid var(--line)}
#insp h3{margin:0 0 6px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);font-weight:600}
.small{font-size:12px;color:var(--mut)}
table{border-collapse:collapse;font:11px var(--mono);width:100%;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;white-space:nowrap} th{color:var(--mut);font-weight:500} td:first-child,th:first-child{text-align:left}
.tw{overflow-x:auto} .ok{color:var(--good)} .badc{color:var(--bad)} .warn{color:var(--redraw)} .dr .a .warn{color:var(--redraw)}
.kpi{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:4px 0 8px}
.kpi div{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.kpi b{display:block;font:500 15px var(--mono)} .kpi span{font-size:10px;color:var(--mut);letter-spacing:.04em;text-transform:uppercase}
.grade{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.grade button{padding:10px 6px;font-weight:600;border-width:2px} .grade kbd{display:block;font:11px var(--mono);color:var(--mut);font-weight:400}
#bgood.on{border-color:var(--good);color:var(--good)} #bbad.on{border-color:var(--bad);color:var(--bad)} #bredraw.on{border-color:var(--redraw);color:var(--redraw)}
textarea{width:100%;min-height:70px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:8px;font:13px var(--sans);resize:vertical;margin-top:6px}
#status{font:11px var(--mono);color:var(--mut)} #status.ok{color:var(--good)} #status.bad{color:var(--bad)}
#exportbox{display:none;margin-top:6px} #exportbox textarea{min-height:120px;font:11px var(--mono)}
.keys{font:11px var(--mono);color:var(--mut);display:grid;grid-template-columns:auto 1fr;gap:2px 10px} .keys kbd{color:var(--ink)}
.fl{display:flex;flex-direction:column;gap:3px;font:11px var(--mono)} .fl label{display:flex;gap:6px;align-items:center;cursor:pointer}
.fl .sw{width:14px;height:3px;border-radius:2px;display:inline-block;flex:none} .fl .t{margin-left:auto;color:var(--mut)}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
<div id="app">
<header><h1>Sketch Review</h1><span class="prog">the six drawings behind the paper's sketch results</span><span class="prog"><b id="pdone">0</b> / <span id="ptotal">0</span> reviewed</span><span class="prog" id="ptally"></span><span class="sp"></span><span id="status">local only</span></header>
<div id="main">
  <nav id="rail" aria-label="drawings"></nav>
  <div id="stage">
    <div id="cvwrap"><canvas id="cv"></canvas><div id="hud"></div><div id="labels"></div></div>
    <div id="ctl">
      <div class="row"><span class="legend">
        <label><input type="checkbox" id="tcloud" checked><span class="sw" style="background:#8d97a6"></span>scene cloud</label>
        <label><input type="checkbox" id="tgeom" checked><span class="sw" style="background:var(--acc)"></span>judge apertures + goal box</label>
        <label><input type="checkbox" id="twp" checked><span class="sw" style="background:var(--sketch)"></span>waypoint numbers</label>
        <label><input type="checkbox" id="tclr" checked><span class="sw" style="background:linear-gradient(90deg,#5fd38a,#f0705c)"></span>colour line by its clearance (red &lt; 0.18 m)</label>
        <label><input type="checkbox" id="tring" checked><span class="sw" style="background:#fff"></span>0.18 m ring at closest approach</label>
        <label><input type="checkbox" id="tflights" checked><span class="sw" style="background:#4b5563"></span>flights</label></span>
        <span class="sp"></span>
        <button data-view="iso">iso</button><button data-view="top">top</button><button data-view="front">front</button><button data-view="side">side</button><button data-view="gate">look through gate</button></div>
    </div>
  </div>
  <aside id="insp">
    <section><h3 id="title">drawing</h3><div class="small" id="sub"></div>
      <div class="kpi"><div><b id="kroute">–</b><span>xswap route-clean</span></div><div><b id="kclean">–</b><span>xswap collision-free</span></div><div><b id="kcells">–</b><span>xswap cells (seeds)</span></div></div>
      <div class="small" id="bestcells" style="margin:-4px 0 8px"></div>
      <div class="kpi"><div><b id="kclr">–</b><span>min clearance to gate cloud</span></div><div><b id="kunder">–</b><span>line length under 0.18 m</span></div><div><b id="klen">–</b><span>route length</span></div></div>
      <div class="small">Clearance is measured on the drawn line itself, against the same gate cloud the flight scorer uses. A line that already passes inside 0.18 m of a post asks the drone to graze.</div>
    </section>
    <section><h3>aperture crossings</h3><div class="tw" id="xtab"></div></section>
    <section><h3>waypoints</h3><div class="tw" id="wtab"></div></section>
    <section><h3>variants and the flights they produced</h3><div class="fl" id="flights"></div></section>
    <section><h3>your verdict on the drawing</h3>
      <div class="grade"><button id="bgood">good<kbd>G</kbd></button><button id="bbad">bad<kbd>B</kbd></button><button id="bredraw">redraw<kbd>R</kbd></button></div>
      <div class="row" style="margin-top:6px"><button id="bclear">clear (X)</button></div>
      <textarea id="note" placeholder="what is wrong with the line (pierces west third of center aperture, too tight around post 2, descends early, ...)"></textarea>
      <div class="small" id="gradedat"></div></section>
    <section><h3>export</h3><div class="row"><button id="bsave">save sketches.json</button><button id="bcopy">show JSON</button></div><div id="exportbox"><textarea id="exporttxt" readonly></textarea></div></section>
    <section><h3>keys</h3><div class="keys"><kbd>G B R X</kbd><span>good / bad / redraw / clear</span><kbd>J K</kbd><span>next / previous drawing</span><kbd>drag</kbd><span>orbit · wheel zoom · shift-drag pan</span></div></section>
  </aside>
</div></div>
<script>
const D=__DATA__;
function dec(b){const s=atob(b);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
for(const s of Object.values(D.scenes)){s.P=new Float32Array(dec(s.pts).buffer);s.C=dec(s.rgb);let cx=0,cy=0,cz=0;for(let i=0;i<s.n;i++){cx+=s.P[3*i];cy+=s.P[3*i+1];cz+=s.P[3*i+2];}s.centre=[cx/s.n,cy/s.n,cz/s.n];}
const DR=D.drawings; DR.forEach(d=>{d.dense=new Float32Array(dec(d.dense).buffer);d.dist=new Float32Array(dec(d.dist).buffer);d.flights.forEach(f=>f.trials.forEach(t=>{t.P=new Float32Array(dec(t.xyz).buffer);delete t.xyz;}));});
// grades
const LS='sketchreview.v1'; let grades={}; try{grades=JSON.parse(localStorage.getItem(LS)||'{}');}catch(e){}
let db=null; const statusEl=document.getElementById('status'); function setStatus(t,c){statusEl.textContent=t;statusEl.className=c||'';}
function persist(){try{localStorage.setItem(LS,JSON.stringify(grades));}catch(e){}}
async function setGrade(id,patch){const cur=grades[id]||{}; const g={...cur,...patch,ts:new Date().toISOString()}; if(!g.v&&!(g.note||'').trim()) delete grades[id]; else grades[id]=g; persist(); renderAll();
  if(db){try{ if(grades[id]) await db.collection('sketches').doc(id).set(grades[id]); else await db.collection('sketches').doc(id).delete(); setStatus('saved to shared db','ok'); }catch(e){setStatus('db write failed: '+(e&&e.code||e),'bad');}}}
(async()=>{ if(!window.claude||!window.claude.use) return; try{db=await claude.use('db');}catch(e){db=null;} if(!db){setStatus('local only (no shared db in this view)');return;} setStatus('shared db connected','ok');
  db.collection('sketches').onSnapshot(snap=>{let ch=false; snap.docChanges().forEach(c=>{const d=c.doc.data(),n=c.doc.id; if(c.type==='removed'){if(grades[n]){delete grades[n];ch=true;}return;} if(!d)return; const cur=grades[n]; if(!cur||(d.ts||'')>=(cur.ts||'')){grades[n]={v:d.v,note:d.note||'',ts:d.ts};ch=true;}}); if(ch){persist();renderAll();}},e=>setStatus('db subscription ended: '+e.code,'bad'));
  try{const have=await db.collection('sketches').get(); const ids=new Set(have.docs.map(d=>d.id)); for(const [n,g] of Object.entries(grades)) if(!ids.has(n)) await db.collection('sketches').doc(n).set(g);}catch(e){} })();
// GL
const cv=document.getElementById('cv'),gl=cv.getContext('webgl',{antialias:true,alpha:false});
const vs=`attribute vec3 p;attribute vec4 c;uniform mat4 M;uniform float ps;varying vec4 vc;void main(){gl_Position=M*vec4(p,1.0);gl_PointSize=ps;vc=c;}`,fs=`precision mediump float;varying vec4 vc;uniform float alpha;void main(){gl_FragColor=vec4(vc.rgb,vc.a*alpha);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o;} const prog=gl.createProgram();gl.attachShader(prog,sh(gl.VERTEX_SHADER,vs));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(prog);gl.useProgram(prog);
const aP=gl.getAttribLocation(prog,'p'),aC=gl.getAttribLocation(prog,'c'),uM=gl.getUniformLocation(prog,'M'),uPS=gl.getUniformLocation(prog,'ps'),uA=gl.getUniformLocation(prog,'alpha'); gl.enableVertexAttribArray(aP);gl.enableVertexAttribArray(aC);
function buf(a){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);return b;} function mk(pos,col){return {n:pos.length/3,bp:buf(pos),bc:buf(col)};}
function solid(pos,col){const c=new Float32Array(pos.length/3*4);for(let i=0;i<pos.length/3;i++)c.set(col,4*i);return mk(pos,c);}
function centred(P,c){const o=new Float32Array(P.length);for(let i=0;i<P.length;i+=3){o[i]=P[i]-c[0];o[i+1]=P[i+1]-c[1];o[i+2]=P[i+2]-c[2];}return o;}
function bind(o){gl.bindBuffer(gl.ARRAY_BUFFER,o.bp);gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,o.bc);gl.vertexAttribPointer(aC,4,gl.FLOAT,false,0,0);}
const sceneGL={}; function sceneBuffers(name){if(sceneGL[name])return sceneGL[name];const s=D.scenes[name];const C=new Float32Array(s.n*4);for(let i=0;i<s.n;i++){C[4*i]=s.C[3*i]/255;C[4*i+1]=s.C[3*i+1]/255;C[4*i+2]=s.C[3*i+2]/255;C[4*i+3]=1;}
  const geom=[];s.gates.forEach(g=>geom.push({...solid(centred(new Float32Array(g.concat([g[0]]).flat()),s.centre),[0.49,0.82,0.94,1]),mode:'strip'}));
  const c=s.goal.c,h=s.goal.h,co=[];for(const sx of[-1,1])for(const sy of[-1,1])for(const sz of[-1,1])co.push([c[0]+sx*h[0],c[1]+sy*h[1],c[2]+sz*h[2]]);const E=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]],seg=[];E.forEach(([a,b])=>seg.push(...co[a],...co[b]));
  geom.push({...solid(centred(new Float32Array(seg),s.centre),[0.97,0.82,0.35,1]),mode:'lines'}); return sceneGL[name]={cloud:mk(centred(s.P,s.centre),C),geom};}
const PAL=[[0.55,0.60,0.70],[0.45,0.70,0.60],[0.70,0.55,0.65],[0.60,0.65,0.45],[0.50,0.58,0.78],[0.75,0.60,0.45],[0.45,0.72,0.72],[0.68,0.50,0.50]];
let di=0, cam={yaw:-0.6,pitch:0.45,dist:5.5,panx:0,pany:0}, G=null, flightOn=[];
function dr(){return DR[di];}
function load(){const d=dr(),s=D.scenes[d.scene]; const dense=centred(d.dense,s.centre); const n=d.ndense;
  const cc=new Float32Array(n*4),co=new Float32Array(n*4); for(let i=0;i<n;i++){const x=d.dist[i]; const f=Math.max(0,Math.min(1,(x-0.10)/(0.35-0.10))); // red at <=0.10, green at >=0.35
    cc[4*i]=0.94-0.57*f;cc[4*i+1]=0.44+0.39*f;cc[4*i+2]=0.36+0.18*f;cc[4*i+3]=1; co[4*i]=1;co[4*i+1]=0.67;co[4*i+2]=0.26;co[4*i+3]=1;}
  const wp=centred(new Float32Array(d.pts.flat()),s.centre);
  G={line:mk(dense,cc),lineo:mk(dense,co),wp:solid(wp,[1,0.67,0.26,1]),
     cross:solid(centred(new Float32Array(d.crossings.map(c=>c.xyz).flat()),s.centre),[0.49,0.82,0.94,1]),
     minpt:solid(centred(new Float32Array(d.minclr_xyz),s.centre),[1,1,1,1]),
     flights:d.flights.map((f,k)=>({col:PAL[k%PAL.length],trials:f.trials.map(t=>solid(centred(t.P,s.centre),[...PAL[k%PAL.length],1]))}))};
  flightOn=d.flights.map(f=>f.best||!d.flights.some(x=>x.best));}
const RING=(()=>{const N=48,o=[];for(let i=0;i<=N;i++){const a=2*Math.PI*i/N;o.push(Math.cos(a),Math.sin(a),0);}for(let i=0;i<=N;i++){const a=2*Math.PI*i/N;o.push(Math.cos(a),0,Math.sin(a));}return new Float32Array(o);})();
const ringC=solid(RING,[1,1,1,0.9]),dyn=gl.createBuffer();
function mat(){const {yaw,pitch,dist,panx,pany}=cam,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const ex=dist*cp*sy,ey=-dist*cp*cy,ez=dist*sp;const f=[-ex,-ey,-ez];const fl=Math.hypot(...f)||1;f.forEach((v,i)=>f[i]=v/fl);let up=[0,0,1];if(Math.abs(f[2])>0.999)up=[0,1,0];
  let s=[f[1]*up[2]-f[2]*up[1],f[2]*up[0]-f[0]*up[2],f[0]*up[1]-f[1]*up[0]];const sl=Math.hypot(...s)||1;s=s.map(v=>v/sl);const u=[s[1]*f[2]-s[2]*f[1],s[2]*f[0]-s[0]*f[2],s[0]*f[1]-s[1]*f[0]];
  const V=[s[0],u[0],-f[0],0,s[1],u[1],-f[1],0,s[2],u[2],-f[2],0,-(s[0]*ex+s[1]*ey+s[2]*ez)+panx,-(u[0]*ex+u[1]*ey+u[2]*ez)+pany,(f[0]*ex+f[1]*ey+f[2]*ez),1];
  const asp=cv.width/cv.height,n=0.05,fa=200,t=1/Math.tan(0.45);const P=[t/asp,0,0,0,0,t,0,0,0,0,(fa+n)/(n-fa),-1,0,0,2*fa*n/(n-fa),0];const M=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let v=0;for(let k=0;k<4;k++)v+=P[k*4+j]*V[i*4+k];M[i*4+j]=v;}return M;}
function project(M,p){const x=M[0]*p[0]+M[4]*p[1]+M[8]*p[2]+M[12],y=M[1]*p[0]+M[5]*p[1]+M[9]*p[2]+M[13],w=M[3]*p[0]+M[7]*p[1]+M[11]*p[2]+M[15];if(w<=0)return null;return [(x/w*0.5+0.5)*cv.clientWidth,(0.5-y/w*0.5)*cv.clientHeight];}
const T={cloud:tcloud,geom:tgeom,wp:twp,clr:tclr,ring:tring,flights:tflights}; Object.values(T).forEach(x=>x.addEventListener('change',draw));
function draw(){const d=dr(),s=D.scenes[d.scene],dpr=Math.min(devicePixelRatio||1,2),w=cv.clientWidth,h=cv.clientHeight;if(cv.width!==w*dpr||cv.height!==h*dpr){cv.width=w*dpr;cv.height=h*dpr;}
  gl.viewport(0,0,cv.width,cv.height);gl.clearColor(0.055,0.067,0.086,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  const M=mat();gl.uniformMatrix4fv(uM,false,M);const sb=sceneBuffers(d.scene);
  if(T.cloud.checked){gl.uniform1f(uPS,1.7*dpr);gl.uniform1f(uA,0.6);bind(sb.cloud);gl.drawArrays(gl.POINTS,0,sb.cloud.n);} gl.uniform1f(uA,1);
  if(T.geom.checked){gl.uniform1f(uPS,2*dpr);sb.geom.forEach(g=>{bind(g);gl.drawArrays(g.mode==='lines'?gl.LINES:gl.LINE_STRIP,0,g.n);});}
  if(T.flights.checked){gl.uniform1f(uA,0.5);G.flights.forEach((f,k)=>{if(!flightOn[k])return;f.trials.forEach(t=>{bind(t);gl.drawArrays(gl.LINE_STRIP,0,t.n);});});gl.uniform1f(uA,1);}
  const L=T.clr.checked?G.line:G.lineo;bind(L);gl.uniform1f(uPS,4*dpr);gl.drawArrays(gl.LINE_STRIP,0,L.n);gl.drawArrays(gl.POINTS,0,L.n);
  bind(G.wp);gl.uniform1f(uPS,12*dpr);gl.drawArrays(gl.POINTS,0,G.wp.n);
  if(G.cross.n){bind(G.cross);gl.uniform1f(uPS,9*dpr);gl.drawArrays(gl.POINTS,0,G.cross.n);}
  const m=d.minclr_xyz,c=s.centre,here=[m[0]-c[0],m[1]-c[1],m[2]-c[2]];
  if(T.ring.checked){const r=D.body_r,ring=new Float32Array(RING.length);for(let i=0;i<RING.length;i+=3){ring[i]=here[0]+r*RING[i];ring[i+1]=here[1]+r*RING[i+1];ring[i+2]=here[2]+r*RING[i+2];}
    gl.bindBuffer(gl.ARRAY_BUFFER,dyn);gl.bufferData(gl.ARRAY_BUFFER,ring,gl.DYNAMIC_DRAW);gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,ringC.bc);gl.vertexAttribPointer(aC,4,gl.FLOAT,false,0,0);gl.uniform1f(uPS,2*dpr);gl.drawArrays(gl.LINE_STRIP,0,49);gl.drawArrays(gl.LINE_STRIP,49,49);
    bind(G.minpt);gl.uniform1f(uPS,10*dpr);gl.drawArrays(gl.POINTS,0,1);}
  // labels
  const lab=document.getElementById('labels');lab.innerHTML='';
  if(T.wp.checked){d.pts.forEach((p,i)=>{const q=project(M,[p[0]-c[0],p[1]-c[1],p[2]-c[2]]);if(!q)return;const e=document.createElement('span');e.className='wp';e.style.left=q[0]+'px';e.style.top=q[1]+'px';e.textContent=i+1;lab.appendChild(e);});
    d.crossings.forEach((x,i)=>{const q=project(M,[x.xyz[0]-c[0],x.xyz[1]-c[1],x.xyz[2]-c[2]]);if(!q)return;const e=document.createElement('span');e.className='xp';e.style.left=q[0]+'px';e.style.top=q[1]+'px';e.textContent=`${x.gate} ${x.inside?'in':'OUT'} ${x.edge_margin.toFixed(2)}m`;lab.appendChild(e);});}
  document.getElementById('hud').innerHTML=`<b>${d.id}</b> · ${d.scene} · min clearance <b>${d.minclr.toFixed(3)} m</b> on segment ${d.minclr_seg+1}→${d.minclr_seg+2} at (${d.minclr_xyz.map(v=>v.toFixed(2)).join(', ')})`;}
let drag=null;cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,sh:e.shiftKey};});addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.sh){cam.panx+=dx*0.004;cam.pany-=dy*0.004;}else{cam.yaw+=dx*0.006;cam.pitch=Math.max(-1.5,Math.min(1.5,cam.pitch+dy*0.006));}draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();cam.dist*=Math.exp(e.deltaY*0.0012);cam.dist=Math.max(0.6,Math.min(60,cam.dist));draw();},{passive:false});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{const v=b.dataset.view;cam.panx=0;cam.pany=0;
  if(v==='iso'){cam.yaw=-0.6;cam.pitch=0.45;cam.dist=5.5;} if(v==='top'){cam.yaw=0;cam.pitch=1.5;cam.dist=6;} if(v==='front'){cam.yaw=0;cam.pitch=0.05;cam.dist=6;} if(v==='side'){cam.yaw=Math.PI/2;cam.pitch=0.05;cam.dist=6;}
  if(v==='gate'){const g=D.scenes[dr().scene].gates; const last=g[g.length-1]; const u=[last[1][0]-last[0][0],last[1][1]-last[0][1]]; cam.yaw=Math.atan2(u[1],u[0])+Math.PI/2; cam.pitch=0.02; cam.dist=4;} draw();}));
// UI
function tally(){let g=0,b=0,r=0;DR.forEach(d=>{const x=grades[d.id];if(!x||!x.v)return;if(x.v==='good')g++;else if(x.v==='bad')b++;else r++;});return {g,b,r};}
function renderRail(){const rail=document.getElementById('rail');rail.innerHTML='<h2>drawings</h2>';DR.forEach((d,i)=>{const gr=grades[d.id];const el=document.createElement('div');el.className='dr'+(i===di?' on':'');el.tabIndex=0;
  const bad=d.minclr<D.body_r, out=d.crossings.some(x=>!x.inside);
  const B=d.best, bt=B.route[1]?`xswap route <span class="${B.route[0]===B.route[1]?'ok':'bad'}">${B.route[0]}/${B.route[1]}</span> · clear <span class="${B.clean[0]===B.clean[1]?'ok':'bad'}">${B.clean[0]}/${B.clean[1]}</span>`:'<span class="warn">not flown on xswap yet</span>';
  el.innerHTML=`<span class="l">${d.label}</span><span class="mark ${gr&&gr.v||''}"></span><span class="a" style="font-family:var(--sans);color:var(--acc)">${d.paper.split(':')[0]}</span><span class="a">${bt}</span><span class="a">sketch clr <span class="${bad?'bad':'ok'}">${d.minclr.toFixed(2)} m</span> · ${d.pts.length} pts${out?' · <span class="bad">passes beside an aperture</span>':''}</span>`;
  el.addEventListener('click',()=>select(i));el.addEventListener('keydown',e=>{if(e.key==='Enter')select(i);});rail.appendChild(el);});}
function renderInsp(){const d=dr();document.getElementById('title').textContent=d.label;
  document.getElementById('sub').innerHTML=`<span style="color:var(--acc)">${d.paper}</span><br>${d.files.map(f=>f.file).join(', ')} · scene ${d.scene} · prompt after: "${d.files[0].prompt_after}"`;
  const B=d.best; kroute.innerHTML=B.route[1]?`<span class="${B.route[0]===B.route[1]?'ok':'badc'}">${B.route[0]}/${B.route[1]}</span>`:'<span class="warn">–</span>';
  kclean.innerHTML=B.clean[1]?`<span class="${B.clean[0]===B.clean[1]?'ok':'badc'}">${B.clean[0]}/${B.clean[1]}</span>`:'<span class="warn">–</span>'; kcells.textContent=B.cells.length||'0';
  bestcells.innerHTML=B.cells.length?B.cells.map(c=>`${c.label}: route ${c.route}, clear ${c.clean}`).join('<br>'):'<span class="warn">This drawing has not been flown on the flagship (xswap) checkpoint. Cells are queued.</span>';
  kclr.innerHTML=`<span class="${d.minclr<D.body_r?'badc':'ok'}">${d.minclr.toFixed(3)} m</span>`;kunder.innerHTML=`<span class="${d.under>0?'badc':'ok'}">${d.under.toFixed(2)} m</span>`;klen.textContent=d.length.toFixed(2)+' m';
  xtab.innerHTML=d.crossings.length?`<table><tr><th>gate</th><th>where (x, y, z)</th><th>across</th><th>height</th><th>inside</th><th>edge margin</th><th>cloud clr</th><th>dir</th></tr>`+d.crossings.map(x=>`<tr><td>${x.gate}</td><td>${x.xyz.map(v=>v.toFixed(2)).join(', ')}</td><td>${x.u.toFixed(2)} / ${x.W.toFixed(2)}</td><td>${x.v.toFixed(2)} / ${x.H.toFixed(2)}</td><td class="${x.inside?'ok':'badc'}">${x.inside?'yes':'NO'}</td><td class="${x.edge_margin<0.3?'warn':'ok'}">${x.edge_margin.toFixed(2)}</td><td class="${x.cloud_clearance<D.body_r?'badc':'ok'}">${x.cloud_clearance.toFixed(2)}</td><td>${x.direction}</td></tr>`).join('')+'</table>':'<span class="small">the line never crosses a judge aperture plane</span>';
  wtab.innerHTML=`<table><tr><th>#</th><th>x</th><th>y</th><th>z</th><th>seg length</th></tr>`+d.pts.map((p,i)=>`<tr><td>${i+1}</td><td>${p[0].toFixed(2)}</td><td>${p[1].toFixed(2)}</td><td>${p[2].toFixed(2)}</td><td>${i<d.seglen.length?d.seglen[i].toFixed(2):''}</td></tr>`).join('')+'</table>';
  const fl=document.getElementById('flights');fl.innerHTML='';d.files.forEach(v=>{const h=document.createElement('div');h.className='small';h.textContent=`${v.file}: sigma ${v.sigma}, step ${v.step_m} m, carrot ${v.carrot}`;fl.appendChild(h);
    d.flights.forEach((f,k)=>{if(f.variant!==v.file)return;const c=PAL[k%PAL.length].map(x=>Math.round(x*255));const l=document.createElement('label');l.innerHTML=`<input type="checkbox" ${flightOn[k]?'checked':''}><span class="sw" style="background:rgb(${c})"></span>${f.label}<span class="t">route ${f.route} · clear ${f.clean}</span>`;l.querySelector('input').addEventListener('change',e=>{flightOn[k]=e.target.checked;draw();});fl.appendChild(l);});});
  if(!d.flights.length){fl.innerHTML+='<span class="small">no flights on record for this drawing</span>';}
  const g=grades[d.id]||{};['good','bad','redraw'].forEach(v=>document.getElementById('b'+v).classList.toggle('on',g.v===v));const note=document.getElementById('note');if(note.value!==(g.note||''))note.value=g.note||'';gradedat.textContent=g.ts?`reviewed ${new Date(g.ts).toLocaleString()}`:'not reviewed';}
function renderAll(){const t=tally();pdone.textContent=t.g+t.b+t.r;ptotal.textContent=DR.length;ptally.innerHTML=`<span style="color:var(--good)">${t.g} good</span> · <span style="color:var(--bad)">${t.b} bad</span> · <span style="color:var(--redraw)">${t.r} redraw</span>`;renderRail();renderInsp();}
function select(i){di=(i+DR.length)%DR.length;load();renderAll();draw();}
['good','bad','redraw'].forEach(v=>document.getElementById('b'+v).addEventListener('click',()=>grade(v)));
function grade(v){const d=dr(),cur=grades[d.id]||{};setGrade(d.id,{v:cur.v===v?null:v,note:document.getElementById('note').value});}
bclear.addEventListener('click',()=>setGrade(dr().id,{v:null,note:''}));
let nt=null;document.getElementById('note').addEventListener('input',e=>{clearTimeout(nt);nt=setTimeout(()=>{const cur=grades[dr().id]||{};setGrade(dr().id,{v:cur.v||null,note:e.target.value});},600);});
function exportJSON(){return JSON.stringify({built:D.built,exported:new Date().toISOString(),reviews:DR.map(d=>({id:d.id,label:d.label,files:d.files.map(f=>f.file),min_clearance_m:d.minclr,crossings:d.crossings,verdict:(grades[d.id]||{}).v||null,note:(grades[d.id]||{}).note||''}))},null,1);}
bsave.addEventListener('click',async()=>{const dl=window.claude&&window.claude.use?await claude.use('downloads'):null;const txt=exportJSON();if(!dl){exportbox.style.display='block';exporttxt.value=txt;setStatus('downloads unavailable here; JSON shown below');return;}try{await dl.save({filename:'sketches.json',data:txt});setStatus('sketches.json saved','ok');}catch(e){setStatus('save '+(e&&e.code||'failed'),'bad');}});
bcopy.addEventListener('click',()=>{exportbox.style.display=exportbox.style.display==='block'?'none':'block';exporttxt.value=exportJSON();});
addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||(e.target.tagName==='INPUT'&&e.target.type!=='checkbox'))return;const k=e.key.toLowerCase();if(k==='g')grade('good');else if(k==='b')grade('bad');else if(k==='r')grade('redraw');else if(k==='x')setGrade(dr().id,{v:null,note:''});else if(k==='j'||k==='arrowdown'){e.preventDefault();select(di+1);}else if(k==='k'||k==='arrowup'){e.preventDefault();select(di-1);}});
addEventListener('resize',draw); select(0);
</script>
"""
out = f"{SP}/sketchreview.html"
open(out, "w").write(HTML.replace("__DATA__", DATA))
print(f"wrote {out}: {len(drawings)} drawings, {sum(len(f['trials']) for d in drawings for f in d['flights'])} flights, {os.path.getsize(out)/1e6:.1f} MB")
for d in drawings:
    xs = "; ".join(f"{x['gate']} {'in' if x['inside'] else 'OUT'} margin {x['edge_margin']:.2f} clr {x['cloud_clearance']:.2f}" for x in d["crossings"])
    print(f"  {d['id']:28s} minclr {d['minclr']:.3f} under {d['under']:.2f} m  | {xs}")
