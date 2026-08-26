"""Seed-7 flight atlas: every gmsig3 battery over its scene point cloud — the full picture of
the trust-dial recipe's replication seed in one page. Five viewers: left (10/10 strict), right
(0/10, planar overshoot), center CFL+CFR, compound-left x10 (10/10 judge, the seed-lottery
composition), compound-right x10.

  python3 build_gmsig3_atlas.py     (writes gmsig3_atlas.html next to this file)
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
    return [np.load(f"{RD}/data_gate_synth3/ep_{e:04d}.npz", allow_pickle=True)["state"][:, :3]
            .astype(np.float32) for e in list(DEMO_EPS[key])[:n]]


def rollouts(pattern):
    fs = sorted(glob.glob(pattern), key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return [np.load(f)[:, :3].astype(np.float32) for f in fs]


DEMO = [128, 136, 150]
MAIN, MAIN_T = [96, 205, 255], [210, 240, 255]     # gmsig3 primary + tails
ALT, ALT_T = [255, 171, 66], [255, 226, 170]       # second group in center scene

SECTIONS = []
for scene, title, note, groups in [
    ("left", "Left — 10/10 strict",
     "Dead-center transits, settles in-box every trial; demos are the regen-2 fans it trained on.",
     [{"label": "demos", "color": DEMO, "trajs": demos("left")},
      {"label": "gmsig3 left (10/10 strict)", "color": MAIN,
       "trajs": rollouts(f"{RUN}/traj_armgmsig3_left_*.npy")}]),
    ("right", "Right — 10/10 STRICT (the settle, closed)",
     "Endpoints (1.43,-0.40,0.99)+/-0.04 — inside the goal box, every trial. The cell no prior "
     "arm closed while keeping the others: the regenerated returns (no east-post skim), correct "
     "pacing, and the trust dial together.",
     [{"label": "demos", "color": DEMO, "trajs": demos("right")},
      {"label": "gmsig3 right (0/10 goal)", "color": MAIN,
       "trajs": rollouts(f"{RUN}/traj_armgmsig3_right_*.npy")}]),
    ("center", "Center — CFL 10/10 strict, CFR 10/10 judge (9/10 clean)",
     "Both center directions complete every trial; one CFR graze costs the clean join.",
     [{"label": "demos (CFL)", "color": DEMO, "trajs": demos("cfl")},
      {"label": "gmsig3 CFL (3/10)", "color": ALT, "trajs": rollouts(f"{RUN}/traj_gmsig3_cfl_*.npy")},
      {"label": "gmsig3 CFR (6/10)", "color": MAIN, "trajs": rollouts(f"{RUN}/traj_gmsig3_cfr_*.npy")}]),
    ("left_and_center", "Compound left->center — 0/5 (no compound demos)",
     "Composition remains the frontier: gate_nav3 contains no compound demonstrations.",
     [{"label": "gmsig3 cmpl x5 (0/5)", "color": MAIN,
       "trajs": rollouts(f"{RUN}/traj_gmsig3_cmpl_*.npy")}]),
    ("right_and_center", "Compound right->center — 0/5",
     "Same frontier; the simple right cell is solved, so gate-1 is no longer the blocker here.",
     [{"label": "gmsig3 cmpr x5 (0/5)", "color": MAIN,
       "trajs": rollouts(f"{RUN}/traj_gmsig3_cmpr_*.npy")}]),
]:
    for g in list(groups):
        if g["label"].startswith("gmsig3"):
            groups.append({"label": g["label"].split(" (")[0] + " tails (last 100)",
                           "color": MAIN_T if g["color"] == MAIN else ALT_T,
                           "trajs": [t[-TAIL:] for t in g["trajs"]]})
    groups += markers(scene)
    html = cloudviewer.viewer_html(scene, groups, note=note, elem_id=f"v_{scene}", max_pts=45000)
    SECTIONS.append((title, html))

body = "".join(f'<h2>{t}</h2>\n<div class="vc">{h}</div>\n' for t, h in SECTIONS)
page = f"""<title>gmsig3 Flight Atlas</title>
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
<h1>gmsig3 Flight Atlas</h1>
<p class="sub">Every gmsig3 battery (trust-dial recipe on gate_nav3, seed 42; 40/40 judge, 39/40 strict-tier) over its scene cloud. Judge goal box and aperture rectangles drawn; each rollout group
has a bright last-100-steps tails overlay. Drag to orbit, wheel to zoom, shift-drag to pan.</p>
{body}
</main>
"""
out = f"{SP}/gmsig3_atlas.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
