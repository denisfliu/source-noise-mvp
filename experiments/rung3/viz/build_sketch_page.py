"""Sketch-prompting results over the compound scene clouds (2026-08-25): the hand-drawn
polylines (Denis, Sketchpad) vs the flights they produced, with the unguided failures for
contrast.

  python3 build_sketch_page.py   (writes sketch_results.html next to this file)
"""
import glob
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RD = os.path.dirname(SP)
RUN = "/home/dfliu/ctxrun"
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}
# compound-scene gate poses differ from the atomic scenes (the CMPR round-1 lesson):
# read gate_1 from the compound safety YAMLs for honest overlay
import yaml
FALSIFY = os.path.expanduser("~/code/falsify-pi")


def scene_marks(scene):
    safety = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{scene}.yaml"))
    gates = [np.asarray(g["corners"], np.float32) for g in safety["ordered_miss_gate"]["gates"]]
    trajs = [np.concatenate([g, g[:1]]) for g in gates]
    corners = np.array([[GOAL_C[0] + sx * GOAL_H[0], GOAL_C[1] + sy * GOAL_H[1], GOAL_C[2] + sz * GOAL_H[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    box = [corners[[a, b]] for a, b in idx]
    return [{"label": "gate apertures (judge)", "color": [124, 208, 240], "trajs": trajs},
            {"label": "goal box (judge)", "color": [248, 210, 90], "trajs": box}]


def rollouts(pattern):
    fs = sorted(glob.glob(pattern), key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return [np.load(f)[:, :3].astype(np.float32) for f in fs]


def sketch_line(path):
    pts = np.asarray(json.load(open(path))["points"], np.float32)[:, :3]
    return [pts]


SECS = []
for scene, tag, sk, note, extra in [
    ("left_and_center", "cmpl", f"{RD}/sketch_cmpl_denis.json",
     "Hand-drawn full-route sketch (orange line) -> 5/5 route-clean AND 5/5 clearance-clean "
     "(min 0.23-0.26 m): the mid-aperture left-gate crossing also cured the graze the "
     "corrective (machine-derived, switch-segment-only) sketch inherited from the "
     "compound-prompt crossing. Unguided (grey) parks at the goal after the atomic task.",
     [("corrective sketch flights (5/5 judge)", [150, 190, 150], f"{RUN}/traj_skcmpl_*.npy")]),
    ("right_and_center", "cmpr", f"{RD}/sketch_cmpr_denis_r1.json",
     "CORRECTED 2026-08-26: round 1 (red) was judged 0/5 by a half-width aperture box in the "
     "safety YAML (it covered only the west third of the real opening, shown fixed here) — "
     "under the corrected box round 1 is 5/5 ROUTE-CLEAN: the drawn crossing was mid-opening "
     "all along. r1 (green, the phantom 'repair') also flies 5/5 and its steer toward the "
     "west post explains its right-gate grazes. The cell never exceeded 1/10 under any "
     "autonomous arm.",
     [("round-1 flights (5/5 after aperture-bug fix)", [150, 190, 150], f"{RUN}/traj_skd_cmpr_*.npy")]),
]:
    groups = [{"label": "unguided gmsig3 (0/5)", "color": [140, 120, 130],
               "trajs": rollouts(f"{RUN}/traj_gmsig3_{tag}_*.npy")}]
    for lbl, col, pat in extra:
        groups.append({"label": lbl, "color": col, "trajs": rollouts(pat)})
    main_pat = f"{RUN}/traj_skd_cmpl_*.npy" if tag == "cmpl" else f"{RUN}/traj_skdr1_cmpr_*.npy"
    groups.append({"label": "hand-drawn sketch flights (5/5 route-clean)", "color": [96, 235, 160],
                   "trajs": rollouts(main_pat)})
    groups.append({"label": "the sketch (drawn command)", "color": [255, 171, 66],
                   "trajs": sketch_line(sk)})
    groups += scene_marks(scene)
    SECS.append((tag.upper() + " — hand-drawn", cloudviewer.viewer_html(
        scene, groups, note=note, elem_id=f"v_{tag}", max_pts=40000)))

# minimal-sketch study sections
g = [{"label": "min4 sketch (4 clicks)", "color": [255, 171, 66],
      "trajs": sketch_line(f"{RD}/sketch_cmpl_min4.json")},
     {"label": "min4 sigma=0 (5/5 route, 0/5 clearance)", "color": [240, 110, 110],
      "trajs": rollouts(f"{RUN}/traj_skm4_cmpl_*.npy")},
     {"label": "min4 sigma=0.5 (4/5 route, 4/5 clearance)", "color": [96, 235, 160],
      "trajs": rollouts(f"{RUN}/traj_skm4s_cmpl_*.npy")}] + scene_marks("left_and_center")
SECS.append(("CMPL — minimal (4 points)", cloudviewer.viewer_html(
    "left_and_center", g, elem_id="v_cmpl_min", max_pts=40000, note=
    "The straight diagonal between gate points (orange) pierces the center aperture ~15 cm "
    "from the west post. At sigma=0 (red) the flow tracks the line faithfully — completes the "
    "route but shaves the post (0.03-0.15 m). At sigma=0.5 (green) the trained trust dial "
    "lets the flow take a wider, slower crossing: clearances 0.18-0.26, at the cost of one "
    "flight losing the route entirely. Toggle groups to compare crossing geometry.")))
g = [{"label": "min4 sketch", "color": [255, 171, 66],
      "trajs": sketch_line(f"{RD}/sketch_cmpr_min4.json")},
     {"label": "min4 sigma=0 (0/5: post clip, passes outside)", "color": [240, 110, 110],
      "trajs": rollouts(f"{RUN}/traj_skm4_cmpr_*.npy")},
     {"label": "min4 sigma=0.5 (0/5: fails at center crossing)", "color": [200, 90, 170],
      "trajs": rollouts(f"{RUN}/traj_skm4s_cmpr_*.npy")},
     {"label": "min5 sketch (+1 staging point)", "color": [248, 210, 90],
      "trajs": sketch_line(f"{RD}/sketch_cmpr_min5.json")},
     {"label": "min5 sigma=0 (5/5 route-clean)", "color": [96, 235, 160],
      "trajs": rollouts(f"{RUN}/traj_skm5_cmpr_*.npy")}] + scene_marks("right_and_center")
SECS.append(("CMPR — minimal (4 vs 5 points)", cloudviewer.viewer_html(
    "right_and_center", g, elem_id="v_cmpr_min", max_pts=40000, note=
    "min4's gate1-to-gate2 diagonal (orange) pierces the center aperture 4 cm inside the "
    "west edge: at sigma=0 (red) the flights clip the post and pass outside — a LINE error "
    "no trust level fixes. sigma=0.5 (magenta) fails at the same center crossing (CORRECTED "
    "2026-08-26: these flights DID cross the right gate through its east half — the earlier "
    "'prior abandons the gate' reading was an artifact of the half-width aperture bug). "
    "min5 (yellow line) adds ONE staging point at (2.75,-0.9) so the approach pierces "
    "mid-aperture: 5/5 route-clean (green); its right-gate grazes come from the r1-style "
    "steer toward the west post.")))

body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Sketch Prompting Results</title>
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
<h1>Sketch Prompting Results</h1>
<p class="sub">A human-drawn polyline, projected into the source-noise command subspace and
served at full trust, flies both compound courses that no autonomous configuration completed
route-clean. The orange line is the drawn command; green flights execute it (tracking ~7 cm).
Both cells 5/5 route-clean; CMPL additionally 5/5 clearance-clean. Screen tier: 5 trials,
one sketch per cell, no video yet. The flow synthesizes all dynamics — the sketch carries
only coarse route topology.</p>
{body}
</main>
"""
out = f"{SP}/sketch_results.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
