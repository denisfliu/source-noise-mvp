"""Reusable interactive 3D trajectory viewer: scene point cloud + one polyline per run.

`viewer_html(scene, groups, title_note)` returns a self-contained HTML block (inline WebGL,
no libraries, no external requests) that can be dropped into any artifact page.

groups = [{"label": str, "color": [r,g,b], "trajs": [Nx3 arrays]}, ...]
"""
import base64
import json
import os

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def viewer_html(scene, groups, note="", height=560, elem_id="v3d", max_pts=None):
    z = np.load(f"{SP}/scene_cloud_{scene}.npz")
    pts, rgb = z["pts"].astype(np.float32), z["rgb"]
    if max_pts and len(pts) > max_pts:   # keep the viewer payload small
        k = np.random.default_rng(0).permutation(len(pts))[:max_pts]
        pts, rgb = pts[k], rgb[k]
    centre = pts.mean(0)
    payload = {
        "n": int(len(pts)),
        "pts": _b64((pts - centre).astype(np.float32)),
        "rgb": _b64(rgb.astype(np.uint8)),
        "centre": centre.tolist(),
        "groups": [{"label": g["label"], "color": g["color"],
                    "trajs": [_b64((np.asarray(t, np.float32)[:, :3] - centre).astype(np.float32))
                              for t in g["trajs"]]}
                   for g in groups],
    }
    j = json.dumps(payload)
    legend = "".join(
        f'<label class="lg"><input type="checkbox" checked data-g="{i}">'
        f'<span class="sw" style="background:rgb({g["color"][0]},{g["color"][1]},{g["color"][2]})"></span>'
        f'{g["label"]} <span class="ct">({len(g["trajs"])})</span></label>'
        for i, g in enumerate(groups))
    return f"""
<div class="v3dwrap">
 <canvas id="{elem_id}" height="{height}"></canvas>
 <div class="v3dui">{legend}
  <label class="lg"><input type="checkbox" checked id="{elem_id}_cloud"> scene cloud</label>
  <span class="hint">drag to orbit · wheel to zoom · shift-drag to pan</span></div>
 {f'<p class="v3dnote">{note}</p>' if note else ''}
</div>
<script>
(function(){{
const D = {j};
const cv = document.getElementById("{elem_id}");
const gl = cv.getContext("webgl", {{antialias:true, alpha:false}});
function dec(b64){{const s=atob(b64);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}}
const pts = new Float32Array(dec(D.pts).buffer), rgb = dec(D.rgb);
const rgbf = new Float32Array(rgb.length); for(let i=0;i<rgb.length;i++) rgbf[i]=rgb[i]/255;
const vs = `attribute vec3 p; attribute vec3 c; uniform mat4 M; uniform float ps;
 varying vec3 vc; void main(){{ gl_Position = M*vec4(p,1.0); gl_PointSize = ps; vc = c; }}`;
const fs = `precision mediump float; varying vec3 vc; uniform float alpha;
 void main(){{ gl_FragColor = vec4(vc, alpha); }}`;
function sh(t,src){{const s=gl.createShader(t);gl.shaderSource(s,src);gl.compileShader(s);
 if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(s));return s;}}
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER,vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER,fs));
gl.linkProgram(prog); gl.useProgram(prog);
const aP=gl.getAttribLocation(prog,"p"), aC=gl.getAttribLocation(prog,"c");
const uM=gl.getUniformLocation(prog,"M"), uPS=gl.getUniformLocation(prog,"ps"), uA=gl.getUniformLocation(prog,"alpha");
function buf(arr){{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
 gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);return b;}}
const bP=buf(pts), bC=buf(rgbf);
const groups = D.groups.map(g=>({{label:g.label,color:g.color.map(v=>v/255),
 trajs:g.trajs.map(t=>{{const a=new Float32Array(dec(t).buffer);
   const col=new Float32Array(a.length); for(let i=0;i<a.length;i+=3){{col[i]=g.color[0]/255;col[i+1]=g.color[1]/255;col[i+2]=g.color[2]/255;}}
   return {{n:a.length/3, bp:buf(a), bc:buf(col)}};}}), on:true}}));
let yaw=-0.6, pitch=0.45, dist=9, panx=0, pany=0, cloudOn=true;
function mat(){{
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const ex=dist*cp*sy, ey=-dist*cp*cy, ez=dist*sp;
  const f=[-ex,-ey,-ez]; let fl=Math.hypot(...f); f.forEach((v,i)=>f[i]=v/fl);
  const up=[0,0,1];
  let s=[f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0]];
  const sl=Math.hypot(...s); s=s.map(v=>v/sl);
  const u=[s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0]];
  const V=[s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0,
   -(s[0]*ex+s[1]*ey+s[2]*ez)+panx, -(u[0]*ex+u[1]*ey+u[2]*ez)+pany, (f[0]*ex+f[1]*ey+f[2]*ez), 1];
  const asp=cv.width/cv.height, n=0.05, fa=200, t=1/Math.tan(0.5*1.0);
  const P=[t/asp,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];
  const M=new Float32Array(16);
  for(let i=0;i<4;i++)for(let jj=0;jj<4;jj++){{let v=0;for(let k=0;k<4;k++)v+=P[k*4+jj]*V[i*4+k];M[i*4+jj]=v;}}
  return M;
}}
function draw(){{
  const dpr=Math.min(window.devicePixelRatio||1,2);
  cv.width=cv.clientWidth*dpr; cv.height={height}*dpr;
  gl.viewport(0,0,cv.width,cv.height);
  gl.clearColor(0.07,0.08,0.10,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  const M=mat(); gl.uniformMatrix4fv(uM,false,M);
  if(cloudOn){{
    gl.uniform1f(uPS,1.6*dpr); gl.uniform1f(uA,0.55);
    gl.bindBuffer(gl.ARRAY_BUFFER,bP); gl.enableVertexAttribArray(aP); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,bC); gl.enableVertexAttribArray(aC); gl.vertexAttribPointer(aC,3,gl.FLOAT,false,0,0);
    gl.drawArrays(gl.POINTS,0,D.n);
  }}
  gl.uniform1f(uA,0.95); gl.uniform1f(uPS,3.0*dpr);
  groups.forEach(g=>{{ if(!g.on) return;
    g.trajs.forEach(t=>{{
      gl.bindBuffer(gl.ARRAY_BUFFER,t.bp); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
      gl.bindBuffer(gl.ARRAY_BUFFER,t.bc); gl.vertexAttribPointer(aC,3,gl.FLOAT,false,0,0);
      gl.drawArrays(gl.LINE_STRIP,0,t.n);
      gl.drawArrays(gl.POINTS,0,t.n);
    }});
  }});
}}
let drag=null;
cv.addEventListener("mousedown",e=>drag={{x:e.clientX,y:e.clientY,sh:e.shiftKey}});
window.addEventListener("mouseup",()=>drag=null);
window.addEventListener("mousemove",e=>{{ if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if(drag.sh){{ panx+=dx*0.004; pany-=dy*0.004; }}
  else {{ yaw+=dx*0.006; pitch=Math.max(-1.4,Math.min(1.4,pitch+dy*0.006)); }}
  draw();
}});
cv.addEventListener("wheel",e=>{{e.preventDefault(); dist*=Math.exp(e.deltaY*0.0012); dist=Math.max(1.5,Math.min(60,dist)); draw();}},{{passive:false}});
document.querySelectorAll('.v3dui input[data-g]').forEach(cb=>cb.addEventListener("change",()=>{{
  groups[+cb.dataset.g].on=cb.checked; draw();}}));
document.getElementById("{elem_id}_cloud").addEventListener("change",e=>{{cloudOn=e.target.checked;draw();}});
window.addEventListener("resize",draw); draw();
}})();
</script>"""


STYLE = """
.v3dwrap{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:14px 0}
.v3dwrap canvas{width:100%;display:block;border-radius:7px;background:#12141a;cursor:grab}
.v3dui{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-top:9px;
 font:600 .78rem ui-monospace,Menlo,monospace;color:var(--mut)}
.v3dui .lg{display:flex;gap:6px;align-items:center;cursor:pointer}
.v3dui .sw{width:22px;height:3px;border-radius:2px;display:inline-block}
.v3dui .ct{opacity:.6;font-weight:400}
.v3dui .hint{margin-left:auto;font-weight:400;opacity:.75}
.v3dnote{color:var(--mut);font-size:.84rem;margin:8px 0 0;max-width:80ch}
"""
