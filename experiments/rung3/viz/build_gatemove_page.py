"""Arbitrary-gate-pose sweep page (2026-08-28): per pose, the scene cloud with the gate's
own points transformed by the same SE(2), the auto-sketch, and the five flights.

  python3 build_gatemove_page.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
RD = os.path.dirname(SP)
GA = np.array([0.195, -1.348]); GB = np.array([0.924, -0.952]); CEN = (GA + GB) / 2
POSES = [("mgm45_0_0", -45, 0, 0, "2/5"), ("mgm25_0_0", -25, 0, 0, "5/5"),
         ("mg25_0_0", 25, 0, 0, "5/5"), ("mg45_0_0", 45, 0, 0, "5/5"),
         ("mg90_0_0", 90, 0, 0, "5/5"), ("mg0_0_5_m0_3", 0, 0.5, -0.3, "0/5"),
         ("mg30_m0_4_0_4", 30, -0.4, 0.4, "5/5")]

Z = np.load(f"{SP}/scene_cloud_right.npz")
pts0, rgb0 = Z["pts"].astype(np.float32), Z["rgb"]
# gate subset: near the original aperture line, z in gate range
rel = pts0[:, :2] - GA
tv0 = (GB - GA) / np.linalg.norm(GB - GA)
nv0 = np.array([tv0[1], -tv0[0]])
gate_m = (np.abs(rel @ nv0) < 0.25) & ((rel @ tv0) > -0.35) & ((rel @ tv0) < 1.15) & (pts0[:, 2] > 0.1)

import tempfile
SECS = []
for tag, dyaw, dx, dy, score in POSES:
    th = math.radians(dyaw)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = CEN - R @ CEN + np.array([dx, dy])
    pts = pts0.copy()
    pts[gate_m, :2] = (R @ pts0[gate_m, :2].T).T + t
    tmp = f"{SP}/scene_cloud_mgtmp.npz"
    np.savez(tmp, pts=pts, rgb=rgb0)
    sk = np.asarray(json.load(open(f"{RD}/sketch_mg_{tag}.json"))["points"], np.float32)[:, :3]
    a2, b2 = R @ GA + t, R @ GB + t
    ap = np.array([[a2[0], a2[1], 0.2], [b2[0], b2[1], 0.2], [b2[0], b2[1], 1.95],
                   [a2[0], a2[1], 1.95], [a2[0], a2[1], 0.2]], np.float32)
    groups = [
        {"label": "auto sketch (4 points, carrot=20)", "color": [255, 171, 66], "trajs": [sk]},
        {"label": f"flights ({score} route-clean; all 5 crossed)", "color": [96, 235, 160],
         "trajs": [np.load(f)[:, :3].astype(np.float32)
                   for f in sorted(glob.glob(f"{RUN}/traj_{tag}_*.npy"))]},
        {"label": "moved aperture (judge)", "color": [124, 208, 240], "trajs": [ap]},
    ]
    SECS.append((f"yaw {dyaw:+.0f} deg, dxy ({dx},{dy}) — {score}",
                 cloudviewer.viewer_html("mgtmp", groups, elem_id=f"v_{tag}", max_pts=40000)))
os.remove(tmp)
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Moved Gates</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;
padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-size:23px;margin:0 0 4px}} h2{{font-size:16px;margin:30px 0 8px;color:var(--acc)}}
.sub{{color:var(--mut);margin:0 0 18px;max-width:92ch}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;
font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}}
.sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}}
.v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
</style>
<main>
<h1>Moved Gates</h1>
<p class="sub">The right gate rigidly moved INSIDE the splat (the cloud shown per viewer has
the gate at its moved pose — what the drone saw), an auto-generated 4-point sketch through
each new aperture, xswap checkpoint, sigma=0, carrot on. Headline: 35/35 flights crossed
their moved gate and reached the goal — including the 90-degree rotation — on a model that
never trained on any of these poses. The 8 route-clean failures are the auto-sketch's exit
leg re-crossing the moved plane on its way to the goal (the -45 and translated poses put
the gate across the return path) — sketch routing, not model capability. Crossing
centering varies (post distances 0.01-0.41): longer straight run-ins would center them.</p>
{body}
</main>
"""
open(f"{SP}/moved_gates.html", "w").write(page)
print("wrote moved_gates.html")
