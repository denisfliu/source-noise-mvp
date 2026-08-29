"""Carrot/kick divergence test page (2026-08-28).

  python3 build_carrot_page.py
"""
import glob
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
RD = os.path.dirname(SP)
pts = np.asarray(json.load(open(f"{RD}/sketch_cmpl_denis.json"))["points"], np.float32)[:, :3]
groups = [
    {"label": "the sketch line", "color": [255, 171, 66], "trajs": [pts]},
    {"label": "no carrot + kick (5/5 route, rejoin 0.11 m, one 0.32 offset)",
     "color": [240, 110, 110],
     "trajs": [np.load(f)[:, :3].astype(np.float32) for f in sorted(glob.glob(f"{RUN}/traj_ck0_*.npy"))]},
    {"label": "carrot=20 + kick (5/5 route, rejoin 0.06 m — tightest)",
     "color": [96, 235, 160],
     "trajs": [np.load(f)[:, :3].astype(np.float32) for f in sorted(glob.glob(f"{RUN}/traj_ck1_*.npy"))]},
    {"label": "carrot=20, no kick (5/5 + 5/5 clean — zero regression)",
     "color": [124, 168, 255],
     "trajs": [np.load(f)[:, :3].astype(np.float32) for f in sorted(glob.glob(f"{RUN}/traj_ck2_*.npy"))]},
]
note = ("A 0.4 m southward kick at step 100 (mid-corridor, before the center crossing). "
        "Both kicked groups complete the route — the re-anchor plus vision already refuse "
        "the shifted-copy failure — but the carrot (green) halves the post-kick "
        "cross-track error and eliminates the 0.32 m offset flight (red group's outlier). "
        "The wrinkle: the carrot's direct rejoin cuts nearer the center-gate west post "
        "(clearance 0/5 vs 2/5 with the kick this close to the gate) — the rejoin length "
        "needs to scale with proximity to structure (gentler carrot near gates).")
html = cloudviewer.viewer_html("left_and_center", groups, note=note, elem_id="v", max_pts=40000)
page = f"""<title>Carrot Kick Test</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;
padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-size:23px;margin:0 0 4px}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;
font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}}
.sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}}
.v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
</style>
<main><h1>Carrot Kick Test</h1><div class="vc">{html}</div></main>
"""
open(f"{SP}/carrot_test.html", "w").write(page)
print("wrote carrot_test.html")
