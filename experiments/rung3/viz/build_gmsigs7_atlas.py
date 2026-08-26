"""Seed-7 flight atlas: every gmsigs7 battery over its scene point cloud — the full picture of
the trust-dial recipe's replication seed in one page. Five viewers: left (10/10 strict), right
(0/10, planar overshoot), center CFL+CFR, compound-left x10 (10/10 judge, the seed-lottery
composition), compound-right x10.

  python3 build_gmsigs7_atlas.py     (writes gmsigs7_atlas.html next to this file)
"""
import glob
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
RD = os.path.dirname(SP)
TAIL = 100

DEMO_EPS = {"left": range(100, 150), "right": range(150, 200),
            "cfl": range(0, 50), "cfr": range(50, 100)}
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}
SCENE_APERTURES = {"left": ["left"], "right": ["right"], "center": ["center"],
                   "left_and_center": ["left", "center"], "right_and_center": ["right", "center"]}


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


def demos(key, n=10):
    return [np.load(f"{RD}/data_gate_synth/ep_{e:04d}.npz", allow_pickle=True)["state"][:, :3]
            .astype(np.float32) for e in list(DEMO_EPS[key])[:n]]


def rollouts(pattern):
    fs = sorted(glob.glob(pattern), key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return [np.load(f)[:, :3].astype(np.float32) for f in fs]


DEMO = [128, 136, 150]
MAIN, MAIN_T = [96, 205, 255], [210, 240, 255]     # gmsigs7 primary + tails
ALT, ALT_T = [255, 171, 66], [255, 226, 170]       # second group in center scene

SECTIONS = []
for scene, title, note, groups in [
    ("left", "Left — 10/10 strict (20/20 across seeds)",
     "Clean 0.36-0.40 m transits, settles inside the goal box every trial. This cell plus seed 42 "
     "is the record-board candidate.",
     [{"label": "demos", "color": DEMO, "trajs": demos("left")},
      {"label": "gmsigs7 left (10/10 strict)", "color": MAIN,
       "trajs": rollouts(f"{RUN}/traj_armgmsigs7_left_*.npy")}]),
    ("right", "Right — 0/10 goal, 10/10 clean",
     "Seed 7's miss mode: z is now in-box (1.18) but the flights overshoot +0.4 m in x past the "
     "goal box (seed 42 hovered 4 cm high at correct x/y). The settle miss persists across seeds; "
     "its geometry is lottery. Demos curl back and land at (1.52,-0.61,1.0).",
     [{"label": "demos", "color": DEMO, "trajs": demos("right")},
      {"label": "gmsigs7 right (0/10 goal)", "color": MAIN,
       "trajs": rollouts(f"{RUN}/traj_armgmsigs7_right_*.npy")}]),
    ("center", "Center — CFR 6/10 success (16/20 across seeds), CFL 3/10",
     "Both directions now complete sometimes. CFR's failures are grazes at the frame (0.03-0.07 m) "
     "and post-goal drift; CFL grazes on approach.",
     [{"label": "demos (CFL)", "color": DEMO, "trajs": demos("cfl")},
      {"label": "gmsigs7 CFL (3/10)", "color": ALT, "trajs": rollouts(f"{RUN}/traj_gmsigs7_cfl_*.npy")},
      {"label": "gmsigs7 CFR (6/10)", "color": MAIN, "trajs": rollouts(f"{RUN}/traj_gmsigs7_cfr_*.npy")}]),
    ("left_and_center", "Compound left->center — 10/10 judge, 0/10 clean",
     "The seed-lottery composition (seed 42: 0/10). Gate 1 at step 88-90, gate 2 at 232-241, dwell "
     "40-84 frames — near-identical trials (deterministic argmax route). Clearance fails AFTER the "
     "dwell latch: post-goal drift back toward the center gate frame (min 0.008-0.15 m at steps "
     "436-500) — watch the tails group.",
     [{"label": "gmsigs7 cmpl x10 (10/10 judge)", "color": MAIN,
       "trajs": rollouts(f"{RUN}/traj_c10gmsigs7_cmpl_*.npy")}]),
    ("right_and_center", "Compound right->center — 1/10",
     "Blocked at gate 1 by the same right-tail gap as the simple right cell; the one success "
     "threads both gates late (171/411).",
     [{"label": "gmsigs7 cmpr x10 (1/10)", "color": MAIN,
       "trajs": rollouts(f"{RUN}/traj_c10gmsigs7_cmpr_*.npy")}]),
]:
    for g in list(groups):
        if g["label"].startswith("gmsigs7"):
            groups.append({"label": g["label"].split(" (")[0] + " tails (last 100)",
                           "color": MAIN_T if g["color"] == MAIN else ALT_T,
                           "trajs": [t[-TAIL:] for t in g["trajs"]]})
    groups += markers(scene)
    html = cloudviewer.viewer_html(scene, groups, note=note, elem_id=f"v_{scene}", max_pts=45000)
    SECTIONS.append((title, html))

body = "".join(f'<h2>{t}</h2>\n<div class="vc">{h}</div>\n' for t, h in SECTIONS)
page = f"""<title>Seed-7 Flight Atlas</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;
padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-size:23px;margin:0 0 4px}} h2{{font-size:16px;margin:32px 0 8px;color:var(--acc)}}
.sub{{color:var(--mut);margin:0 0 18px;max-width:90ch}}
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
<h1>Seed-7 Flight Atlas</h1>
<p class="sub">Every gmsigs7 battery (trust-dial recipe: GMM x mh16 x sigma-conditioned pin,
seed 7) over its scene cloud. Judge goal box and aperture rectangles drawn; each rollout group
has a bright last-100-steps tails overlay. Drag to orbit, wheel to zoom, shift-drag to pan.</p>
{body}
</main>
"""
out = f"{SP}/gmsigs7_atlas.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
