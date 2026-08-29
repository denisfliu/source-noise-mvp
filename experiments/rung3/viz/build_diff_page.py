"""Where gmsig3 and xswap differentiate (2026-08-28): CFR clearance tail, real-right
command quality, real-left command style.

  python3 build_diff_page.py
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
APC = np.array([[3.156, -0.328, 0.125], [2.356, -0.327, 0.125],
                [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]], np.float32)
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
k = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
corners = GOAL_C + k * GOAL_H
idx = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
MARKS = [{"label": "center aperture (judge)", "color": [124, 208, 240],
          "trajs": [np.concatenate([APC, APC[:1]])]},
         {"label": "goal box (judge)", "color": [248, 210, 90],
          "trajs": [corners[[a, b]] for a, b in idx]}]

def load(pat, exclude=()):
    return [np.load(f)[:, :3].astype(np.float32) for f in sorted(glob.glob(pat))
            if os.path.basename(f) not in exclude]

WORST = ("traj_gmsig3_cfr_1.npy", "traj_gmsig3_cfr_6.npy", "traj_gmsig3_cfr_3.npy")
SECS = []
g = [{"label": "gmsig3 CFR, other 7 flights (0.26-0.40 clearance)", "color": [150, 150, 158],
      "trajs": load(f"{RUN}/traj_gmsig3_cfr_*.npy", exclude=WORST)},
     {"label": "gmsig3 CFR, LOW-TAIL 3 (0.138 / 0.218 / 0.228)", "color": [240, 110, 110],
      "trajs": [np.load(f"{RUN}/{f}")[:, :3].astype(np.float32) for f in WORST]},
     {"label": "xswap CFR, all 10 (0.250-0.334 — no tail)", "color": [96, 235, 160],
      "trajs": load(f"{RUN}/traj_xswap_cfr_*.npy")}] + MARKS
SECS.append(("DIFF 1 — CFR clearance tail (sim)", cloudviewer.viewer_html(
    "center", g, elem_id="v_cfr", max_pts=40000, note=
    "The medians are similar (0.31 vs 0.29); the difference is VARIANCE. gmsig3's red "
    "flights are its low tail — the 0.138 one is the strict miss — all shaving the west "
    "post on the goal descent. xswap's ten descents cluster: worst flight 0.250. "
    "Cross-supervision bought consistency, not boldness.")))

PATHS = np.load(f"{RUN}/real_pred_chunks.npz")
for side, title, note in [
    ("right", "DIFF 2 — real-frame command QUALITY (right)",
     "Own-head fans on real anchors: xswap (green) is longer (0.79 vs 0.59 m), straighter "
     "(22.6 vs 26.8 deg), and reaches the gate almost twice as often (15 vs 8 crossings, "
     "centered at s=0.40 vs 0.52). This is the trained-in twin where the head was weak."),
    ("left", "DIFF 3 — real-frame command STYLE (left)",
     "Functional parity (crossings unchanged) but the styles split: gmsig3 (amber) imitates "
     "the pilot's line (7.9 deg from it); xswap (green) flies the planner's (32 deg from "
     "the pilot, task-valid). Which style you want on left is a deployment choice, not a "
     "performance fact.")]:
    paths = [PATHS[kk] for kk in PATHS.files if kk.startswith(f"{side}_path_")]
    groups = [{"label": f"real flights ({len(paths)})", "color": [150, 150, 158], "trajs": paths}]
    for name, path, col in [("gmsig3 own-head", f"{RUN}/synthpin2_rows.npz", [255, 171, 66]),
                            ("xswap own-head", f"{RUN}/synthpin_xswap.npz", [96, 235, 160])]:
        Z = np.load(path, allow_pickle=True)
        meta = json.loads(str(Z["meta"]))
        groups.append({"label": name, "color": col,
                       "trajs": [Z[f"arr_{i}_headpin_traj"] for i, r in enumerate(meta)
                                 if r["side"] == side]})
    SECS.append((title, cloudviewer.viewer_html(side, groups, elem_id=f"v_{side}",
                                                max_pts=40000, note=note)))

body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Gmsig3 vs Xswap</title>
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
<h1>Gmsig3 vs Xswap</h1>
<p class="sub">Equal on the judge tier (40/40 both, same routes, same transit times). They
differentiate in exactly three places, one per viewer below: the CFR clearance tail in sim,
command quality on real right frames, and command style on real left frames.</p>
{body}
</main>
"""
open(f"{SP}/gmsig3_vs_xswap.html", "w").write(page)
print("wrote gmsig3_vs_xswap.html")
