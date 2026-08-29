"""Local live-intent cockpit (2026-08-29): NOT an artifact — artifacts cannot reach
localhost (CSP), so this page is opened from disk and WebSockets to the serve bridge.

  python3 build_live_intent.py --scene right   -> live_intent_right.html
"""
import argparse
import base64
import os

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="right")
ap.add_argument("--port", type=int, default=8765)
a = ap.parse_args()
z = np.load(f"{SP}/scene_cloud_{a.scene}.npz")
pts, rgb = z["pts"].astype(np.float32), z["rgb"]
if len(pts) > 40000:
    k = np.random.default_rng(0).permutation(len(pts))[:40000]
    pts, rgb = pts[k], rgb[k]
b64 = lambda x: base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()

page = """<!doctype html><html><head><meta charset="utf-8"><title>Live Intent</title>
<style>
body{margin:0;background:#0f1216;color:#e4e9f1;font:14px/1.5 system-ui;display:flex;
flex-direction:column;height:100vh}
#top{padding:10px 14px;background:#151a21;border-bottom:1px solid #28303c;display:flex;
gap:14px;align-items:center;flex-wrap:wrap}
#sent{font:600 17px ui-monospace,Menlo,monospace;color:#eb6ed2;flex:1 1 400px}
#meta{font:12px ui-monospace,Menlo,monospace;color:#8b94a5}
button{background:#1d2530;color:#e4e9f1;border:1px solid #28303c;border-radius:6px;
padding:7px 14px;font:600 13px system-ui;cursor:pointer}
button.on{background:#7cd0f0;color:#0c1116}
button.go{background:#60eba0;color:#0c1116}
canvas{flex:1;width:100%;display:block;cursor:grab}
#gatebar{display:none;padding:8px 14px;background:#2a1c10;border-bottom:1px solid #4a3018;
gap:10px;align-items:center}
#gatebar.show{display:flex}
#log{position:fixed;right:10px;bottom:10px;width:330px;max-height:36vh;overflow-y:auto;
background:#151a21cc;border:1px solid #28303c;border-radius:8px;padding:8px;
font:11px ui-monospace,Menlo,monospace;color:#9aa4b6}
</style></head><body>
<div id="top">
 <span id="sent">waiting for server…</span>
 <span id="meta"></span>
 <button id="mode">mode: AUTO</button>
 <label>&sigma; <input type="range" id="sig" min="-1" max="150" value="-1" style="width:110px">
 <span id="sigv">map</span></label>
 <span id="conn" style="color:#f06e6e">&#9679; disconnected</span>
</div>
<div id="gatebar">
 <b style="color:#ffab42">PROPOSAL PENDING</b>
 <button class="go" id="approve">APPROVE</button>
 <button id="rotL">&#8634; rotate L15</button>
 <button id="rotR">&#8635; rotate R15</button>
</div>
<canvas id="cv"></canvas>
<div id="log"></div>
<script>
const N=__N__, PORT=__PORT__;
function dec(b){const s=atob(b);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
const pts=new Float32Array(dec("__PTS__").buffer), rgb8=dec("__RGB__");
const rgbf=new Float32Array(rgb8.length); for(let i=0;i<rgb8.length;i++)rgbf[i]=rgb8[i]/255;
const cv=document.getElementById("cv"), gl=cv.getContext("webgl",{antialias:true});
const vs=`attribute vec3 p;attribute vec3 c;uniform mat4 M;uniform float ps;varying vec3 vc;
void main(){gl_Position=M*vec4(p,1.0);gl_PointSize=ps;vc=c;}`;
const fs=`precision mediump float;varying vec3 vc;uniform float alpha;
void main(){gl_FragColor=vec4(vc,alpha);}`;
function sh(t,s2){const s=gl.createShader(t);gl.shaderSource(s,s2);gl.compileShader(s);return s;}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));
gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
const aP=gl.getAttribLocation(pr,"p"),aC=gl.getAttribLocation(pr,"c");
const uM=gl.getUniformLocation(pr,"M"),uPS=gl.getUniformLocation(pr,"ps"),uA=gl.getUniformLocation(pr,"alpha");
function buf(a){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,a,gl.DYNAMIC_DRAW);return b;}
const bP=buf(pts),bC=buf(rgbf);
let trail=[],intent=[],chunk=[];
function lineBufs(arr,col){const a=new Float32Array(arr.length*3);
 arr.forEach((p,i)=>{a[i*3]=p[0];a[i*3+1]=p[1];a[i*3+2]=p[2];});
 const c=new Float32Array(arr.length*3);
 for(let i=0;i<arr.length;i++){c[i*3]=col[0];c[i*3+1]=col[1];c[i*3+2]=col[2];}
 return {n:arr.length,bp:buf(a),bc:buf(c)};}
let yaw=-0.6,pitch=0.5,dist=8,panx=0,pany=0;
const C=[1.2,-0.4,1.0];
function mat(){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const e=[C[0]+dist*cp*sy,C[1]-dist*cp*cy,C[2]+dist*sp];
 const f=[C[0]-e[0],C[1]-e[1],C[2]-e[2]];let fl=Math.hypot(f[0],f[1],f[2]);f.forEach((v,i)=>f[i]=v/fl);
 let s=[f[1],-f[0],0];const sl=Math.hypot(s[0],s[1],s[2])||1;s=s.map(v=>v/sl);
 const u=[s[1]*f[2]-s[2]*f[1],s[2]*f[0]-s[0]*f[2],s[0]*f[1]-s[1]*f[0]];
 const V=[s[0],u[0],-f[0],0,s[1],u[1],-f[1],0,s[2],u[2],-f[2],0,
  -(s[0]*e[0]+s[1]*e[1]+s[2]*e[2])+panx,-(u[0]*e[0]+u[1]*e[1]+u[2]*e[2])+pany,
  (f[0]*e[0]+f[1]*e[1]+f[2]*e[2]),1];
 const asp=cv.width/cv.height,n=0.05,fa=200,t=1/Math.tan(0.5);
 const P=[t/asp,0,0,0,0,t,0,0,0,0,(fa+n)/(n-fa),-1,0,0,2*fa*n/(n-fa),0];
 const M=new Float32Array(16);
 for(let i=0;i<4;i++)for(let j=0;j<4;j++){let v=0;for(let k=0;k<4;k++)v+=P[k*4+j]*V[i*4+k];M[i*4+j]=v;}
 return M;}
function bindDraw(o,mode,ps){gl.bindBuffer(gl.ARRAY_BUFFER,o.bp);
 gl.enableVertexAttribArray(aP);gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
 gl.bindBuffer(gl.ARRAY_BUFFER,o.bc);gl.enableVertexAttribArray(aC);
 gl.vertexAttribPointer(aC,3,gl.FLOAT,false,0,0);
 if(ps)gl.uniform1f(uPS,ps);gl.drawArrays(mode,0,o.n);}
function draw(){const dpr=Math.min(devicePixelRatio||1,2);
 cv.width=cv.clientWidth*dpr;cv.height=cv.clientHeight*dpr;
 gl.viewport(0,0,cv.width,cv.height);gl.clearColor(0.07,0.08,0.10,1);
 gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
 gl.uniformMatrix4fv(uM,false,mat());
 gl.uniform1f(uA,0.5);bindDraw({n:N,bp:bP,bc:bC},gl.POINTS,1.6*dpr);
 gl.uniform1f(uA,1.0);
 if(trail.length>1)bindDraw(lineBufs(trail,[0.38,0.92,0.63]),gl.LINE_STRIP,3*dpr);
 if(chunk.length>1)bindDraw(lineBufs(chunk,[0.49,0.66,1.0]),gl.LINE_STRIP,3*dpr);
 if(intent.length>1){const o=lineBufs(intent,[0.92,0.43,0.82]);
  bindDraw(o,gl.LINE_STRIP,5*dpr);gl.drawArrays(gl.POINTS,0,o.n);}
}
let drag=null;
cv.onmousedown=e=>drag={x:e.clientX,y:e.clientY,sh:e.shiftKey};
window.onmouseup=()=>drag=null;
window.onmousemove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 drag.x=e.clientX;drag.y=e.clientY;
 if(drag.sh){panx+=dx*0.004;pany-=dy*0.004;}else{yaw+=dx*0.006;
 pitch=Math.max(-1.4,Math.min(1.4,pitch+dy*0.006));}draw();};
cv.onwheel=e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.0012);
 dist=Math.max(1.5,Math.min(40,dist));draw();};
const logEl=document.getElementById("log");
function log(t){const d=document.createElement("div");
 d.textContent=new Date().toLocaleTimeString()+" "+t;
 logEl.prepend(d);while(logEl.childElementCount>60)logEl.lastChild.remove();}
let ws,mode="auto";
function connect(){ws=new WebSocket("ws://127.0.0.1:"+PORT);
 ws.onopen=()=>{document.getElementById("conn").innerHTML="&#9679; live";
  document.getElementById("conn").style.color="#60eba0";};
 ws.onclose=()=>{document.getElementById("conn").innerHTML="&#9679; disconnected";
  document.getElementById("conn").style.color="#f06e6e";setTimeout(connect,1500);};
 ws.onmessage=ev=>{const m=JSON.parse(ev.data);
  if(m.type==="proposal"){
   document.getElementById("sent").textContent=m.text;
   document.getElementById("meta").textContent=
    "sigma*="+m.sigma_star.toFixed(1)+" serve="+m.sigma_serve.toFixed(2)+" phase="+m.phase;
   intent=m.intent;trail.push(m.pos);
   document.getElementById("gatebar").className=m.gate?"show":"";
   log((m.gate?"GATE ":"")+m.text);draw();}
  if(m.type==="executed"){chunk=m.chunk;draw();}
  if(m.type==="still_waiting"){log("…server still waiting on your decision");}};
}
connect();
document.getElementById("mode").onclick=function(){
 mode=mode==="auto"?"gate":"auto";
 this.textContent="mode: "+mode.toUpperCase();this.className=mode==="gate"?"on":"";
 ws.send(JSON.stringify({type:"mode",mode}));};
document.getElementById("approve").onclick=()=>ws.send(JSON.stringify({type:"decision",action:"approve"}));
document.getElementById("rotL").onclick=()=>ws.send(JSON.stringify({type:"decision",action:"rotate",deg:15}));
document.getElementById("rotR").onclick=()=>ws.send(JSON.stringify({type:"decision",action:"rotate",deg:-15}));
document.getElementById("sig").oninput=function(){
 const v=+this.value;document.getElementById("sigv").textContent=v<0?"map":(v/100).toFixed(2);
 ws.send(JSON.stringify({type:"sigma",value:v<0?null:v/100}));};
draw();
</script></body></html>"""
page = page.replace("__N__", str(len(pts))).replace("__PORT__", str(a.port))
page = page.replace("__PTS__", b64(pts)).replace("__RGB__", b64(rgb.astype(np.uint8)))
out = f"{SP}/live_intent_{a.scene}.html"
open(out, "w").write(page)
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB) — open in a browser, ws port {a.port}")
