"""Language-guidance composition recovery over the compound scene clouds (2026-08-25):
unguided (executes the atomic task, parks at the goal) vs CFG w=2 (not enough) vs w=4
(3/5 + 3/5 both-gates+dwell — the first zero-shot compound completions on clean data).

  python3 build_cfg_page.py   (writes cfg_compounds.html next to this file)
"""
import glob
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}
SCENE_APERTURES = {"left_and_center": ["left", "center"], "right_and_center": ["right", "center"]}


def box_edges(c, h):
    corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return [corners[[a, b]] for a, b in idx]


def markers(scene):
    return [{"label": "goal box (judge)", "color": [248, 210, 90], "trajs": box_edges(GOAL_C, GOAL_H)},
            {"label": "gate apertures (judge)", "color": [124, 208, 240],
             "trajs": [np.array(APERTURE[k] + [APERTURE[k][0]], np.float32)
                       for k in SCENE_APERTURES[scene]]}]


def rollouts(pattern):
    fs = sorted(glob.glob(pattern), key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return [np.load(f)[:, :3].astype(np.float32) for f in fs]


UNG, W2, W4, BAD = [140, 120, 130], [255, 171, 66], [96, 235, 160], [240, 110, 110]
SECS = []
for scene, tag, w4label, w4color, note in [
    ("left_and_center", "cmpl", "gmsig4 CFG w=4 (0/5 route-clean — backwards hoop pass)", BAD,
     "Unguided (grey): crosses the left gate then executes the plain left task — parks in the "
     "goal box and ignores the continuation. w=2 (amber): still captured by the atomic return. "
     "w=4 (red): VETOED (Denis, from this viewer) — the flights go to the goal FIRST, then "
     "punch through the center aperture BACKWARDS (+y, from the goal side), U-turn behind the "
     "gate, and re-cross forward; the legacy judge latched the U-turn. Route-clean rule "
     "(wrong-direction passes = fail, demos unanimously clean) scores this 0/5. Guidance "
     "fixed gate selection but not the around-the-far-side route topology CMPL needs."),
    ("right_and_center", "cmpr", "gmsig4 CFG w=4 (3/5 route-clean)", W4,
     "The genuine composition recovery — the task no arm in project history completed above "
     "1/10, and these flights are route-clean (zero wrong-direction passes): CMPR's correct "
     "center crossing (+y) points away from the goal side, so the guided pull at the gate is "
     "route-correct by geometry. Flights 4 and 5 go right gate (t~66) then center gate "
     "(t~138) direct; flight 2 loiters at the goal first, then crosses clean."),
]:
    groups = [
        {"label": "gmsig3 unguided (0/5)", "color": UNG, "trajs": rollouts(f"{RUN}/traj_gmsig3_{tag}_*.npy")},
        {"label": "gmsig4 CFG w=2 (0/5)", "color": W2, "trajs": rollouts(f"{RUN}/traj_cfg2g4_{tag}_*.npy")},
        {"label": w4label, "color": w4color, "trajs": rollouts(f"{RUN}/traj_cfg4g4_{tag}_*.npy")},
    ]
    for g in list(groups):
        groups.append({"label": g["label"].split(" (")[0] + " tails (last 100)",
                       "color": [min(255, c + 70) for c in g["color"]],
                       "trajs": [t[-100:] for t in g["trajs"]]})
    groups += markers(scene)
    SECS.append((tag.upper(), cloudviewer.viewer_html(scene, groups, note=note,
                                                      elem_id=f"v_{tag}", max_pts=40000)))

body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Guided Composition</title>
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
<h1>Guided Composition</h1>
<p class="sub">The language-sharpener progression on the compound tasks, RESCORED under the
route-clean rule (2026-08-25): a wrong-direction pass through an aperture fails the flight
(training demos are unanimously clean under it). Unguided (grey) parks after the atomic task;
w=2 (amber) is still captured by the return attractor; w=4 recovers composition on CMPR only
(green, 3/5) — the CMPL "successes" (red) thread the center hoop backwards and were vetoed.
Zero compound demonstrations in training; the guidance is a serve-time dial on gmsig4's
null-language branch. Toggle the tails groups to compare endings.</p>
{body}
</main>
"""
out = f"{SP}/cfg_compounds.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
