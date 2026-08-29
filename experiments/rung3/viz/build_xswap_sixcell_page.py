"""xswap six-cell evaluation in the cloud (2026-08-28): every judged flight of the
40/40 + 40/40-clearance sweep, with gmsig3 reference and judge geometry.

  python3 build_xswap_sixcell_page.py
"""
import glob
import os
import sys

import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
FALSIFY = os.path.expanduser("~/code/falsify-pi")
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}


def box_edges(c, h):
    k = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    corners = c + k * h
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [corners[[a, b]] for a, b in idx]


def marks(aps, compound_scene=None):
    if compound_scene:
        safety = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{compound_scene}.yaml"))
        gates = [np.asarray(g["corners"], np.float32) for g in safety["ordered_miss_gate"]["gates"]]
        ap = [np.concatenate([g, g[:1]]) for g in gates]
    else:
        ap = [np.array(APERTURE[a] + [APERTURE[a][0]], np.float32) for a in aps]
    return [{"label": "gate apertures (judge)", "color": [124, 208, 240], "trajs": ap},
            {"label": "goal box (judge)", "color": [248, 210, 90], "trajs": box_edges(GOAL_C, GOAL_H)}]


def load(pat):
    return [np.load(f)[:, :3].astype(np.float32) for f in sorted(glob.glob(pat))]


CELLS = [
    ("LEFT (10/10, 10/10 clean)", "left", ["left"], None,
     f"{RUN}/traj_armxswap_left_*.npy", f"{RUN}/traj_armgmsig3_left_*.npy"),
    ("RIGHT (10/10, 10/10 clean)", "right", ["right"], None,
     f"{RUN}/traj_armxswap_right_*.npy", f"{RUN}/traj_armgmsig3_right_*.npy"),
    ("CFL (10/10, 10/10 clean)", "center", ["center"], None,
     f"{RUN}/traj_xswap_cfl_*.npy", f"{RUN}/traj_gmsig3_cfl_*.npy"),
    ("CFR (10/10, 10/10 clean — the descent graze is GONE)", "center", ["center"], None,
     f"{RUN}/traj_xswap_cfr_*.npy", f"{RUN}/traj_gmsig3_cfr_*.npy"),
    ("CMPL (0/5 unguided, 5/5 clean)", "left_and_center", None, "left_and_center",
     f"{RUN}/traj_xswap_cmpl_*.npy", f"{RUN}/traj_gmsig3_cmpl_*.npy"),
    ("CMPR (0/5 unguided, 5/5 clean)", "right_and_center", None, "right_and_center",
     f"{RUN}/traj_xswap_cmpr_*.npy", f"{RUN}/traj_gmsig3_cmpr_*.npy"),
]
SECS = []
for title, scene, aps, comp, xs, gs in CELLS:
    groups = [
        {"label": "gmsig3 reference", "color": [150, 150, 158], "trajs": load(gs)},
        {"label": "xswap", "color": [96, 235, 160], "trajs": load(xs)},
    ] + marks(aps, comp)
    SECS.append((title, cloudviewer.viewer_html(scene, groups,
                                                elem_id=f"v_{len(SECS)}", max_pts=40000)))
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Xswap Six Cells</title>
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
<h1>Xswap Six Cells</h1>
<p class="sub">Every judged flight of the xswap sweep (green) over its scene cloud, with the
gmsig3 flights (grey) at the same cells for comparison, and the judge geometry (blue
apertures, yellow goal box). Toggle grey off to see the sweep alone; the CFR viewer is the
one to inspect closely — gmsig3's near-post descent tightness is visibly relaxed.</p>
{body}
</main>
"""
open(f"{SP}/xswap_sixcells.html", "w").write(page)
print(f"wrote xswap_sixcells.html")
