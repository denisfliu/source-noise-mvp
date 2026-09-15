"""Show a sketch JSON on a scene cloud with numbered waypoints, a self-crossing count, and the scene extents.
  python build_sketch_check.py <sketch.json> <scene> <out.html> [title]"""
import json, os, sys, numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(SP))
import cloudviewer
from build_traj_page import marks, axes
sk, scene, out = sys.argv[1], sys.argv[2], sys.argv[3]; title = sys.argv[4] if len(sys.argv) > 4 else "Sketch Check"
sks = sk.split(",")   # several sketches: comma-separated; the first is checked in detail, all are drawn
P = np.asarray(json.load(open(sks[0]))["points"], np.float32)[:, :3]
extra = [{"label": os.path.basename(f), "color": c, "trajs": [np.asarray(json.load(open(f))["points"], np.float32)[:, :3]]}
         for f, c in zip(sks[1:], ([120, 200, 255], [200, 120, 255], [120, 255, 200]))]
Q = np.asarray(json.load(open(f"{os.path.dirname(SP)}/sketch_fig8.json"))["points"], np.float32)[:, :3]
cloud = np.load(f"{SP}/scene_cloud_{scene}.npz")["pts"]; lo, hi = np.percentile(cloud, 1, axis=0), np.percentile(cloud, 99, axis=0)
L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
groups = [{"label": f"the sketch ({len(P)} points, {L:.1f} m)", "color": [255, 200, 90], "trajs": [P]},
          {"label": "waypoints", "color": [255, 120, 60], "trajs": [np.stack([p, p + np.array([0, 0, 0.08], np.float32)]) for p in P]},
          {"label": "the flown figure-eight sketch (right scene), for scale", "color": [140, 140, 160], "trajs": [Q]}] + extra + marks(scene) + axes()
def ccw(p, q, r): return (r[1]-p[1])*(q[0]-p[0]) > (q[1]-p[1])*(r[0]-p[0])
def seg_cross(a, b, c, d): return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
X = [(i+1, j+1) for i in range(len(P)-1) for j in range(i+2, len(P)-1) if seg_cross(P[i,:2], P[i+1,:2], P[j,:2], P[j+1,:2])]
outside = [i+1 for i, p in enumerate(P) if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1])]
note = (f"Self-crossings in the xy-plane: {len(X)}" + (f" (segment pairs {X})" if X else "") + ". "
        f"Scene cloud footprint (1st-99th percentile): x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f}. "
        + (f"Waypoints outside that footprint: {outside}." if outside else "All waypoints inside the reconstructed footprint."))
rows = "".join(f"<tr><td>{i+1}</td><td>{p[0]:+.2f}</td><td>{p[1]:+.2f}</td><td>{p[2]:.2f}</td><td>{np.linalg.norm(P[i]-P[i-1]) if i else 0:.2f}</td></tr>" for i, p in enumerate(P))
page = f"""<title>{title}</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:28px 18px 70px}} main{{max-width:1100px;margin:0 auto}}
h1{{font-size:23px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 18px;max-width:92ch}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}} .v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace}} .lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
table{{border-collapse:collapse;font:12px ui-monospace,Menlo,monospace;margin:10px 0 0;font-variant-numeric:tabular-nums}} td,th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}} th{{color:var(--acc);font-weight:500}}
</style>
<main><h1>{title}</h1><p class="sub">{os.path.basename(sk)} on the {scene} scene: the sketch in yellow with waypoints ticked, the previously flown figure-eight in grey for scale, judge boxes, origin axes (x red, y green, z blue).</p>
<div class="vc">{cloudviewer.viewer_html(scene, groups, elem_id="v", max_pts=40000, note=note)}
<table><tr><th>#</th><th>x</th><th>y</th><th>z</th><th>leg (m)</th></tr>{rows}</table></div></main>"""
open(os.path.join(SP, out), "w").write(page); print(note)
