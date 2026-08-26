"""Regenerated demo trajectories (regen2, 2026-08-23) over the scene clouds, next to the OLD
demos they replace: wider real-matched start fans, CFR's pinned western return, right's
return_east berth, funneled approaches. Every shown new trajectory passed the posthoc judge
AND gate_clearance at plan time.

  python3 build_regen_page.py   (writes regen2_demos.html next to this file)
"""
import glob
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RD = os.path.dirname(SP)
KEPT = os.path.expanduser("~/code/falsify/runs/regen2/kept")
NPD = os.path.expanduser("~/code/falsify/runs/regen2/np")
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}


def box_edges(c, h):
    corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return [corners[[a, b]] for a, b in idx]


def markers(scene):
    return [{"label": "goal box (judge)", "color": [248, 210, 90], "trajs": box_edges(GOAL_C, GOAL_H)},
            {"label": "gate aperture (judge)", "color": [124, 208, 240],
             "trajs": [np.array(APERTURE[scene] + [APERTURE[scene][0]], np.float32)]}]


def old_demos(eps, n=15):
    return [np.load(f"{RD}/data_gate_synth/ep_{e:04d}.npz", allow_pickle=True)["state"][:, :3]
            .astype(np.float32) for e in list(eps)[:n]]


def new_kept(course):
    names = {os.path.basename(f).replace(".npz", "") for f in glob.glob(f"{KEPT}/{course}/*.npz")}
    return [np.load(f"{NPD}/{course}/{n}.npy").astype(np.float32) for n in sorted(names)]


OLD, NEW = [128, 136, 150], [96, 205, 255]
SECS = []
for course, scene, dem, title, note in [
    ("through_left_gate", "left", range(100, 150), "Left",
     "Two-class design: wide start fan + 0.2 m corridor tubes, converging to a 0.04-spread sphere at the gate. Recovery bloom restored (corrective on approach expresses again). Park at step 220 (old 239, regen1 was 96)."),
    ("through_right_gate", "right", range(150, 200), "Right",
     "return_east berth (>0.4 m from the east post) now with tube variance and old-profile pacing; 70/70 valid."),
    ("through_center_gate_from_left", "center", range(0, 50), "Center from left",
     "return_south berth replaces the old post_gate->hover line that ran 0.31 m from the west post."),
    ("through_center_gate_from_right", "center", range(50, 100), "Center from right",
     "Denis's fix: cross_west pins the southbound re-crossing at x=1.30 (old corridor: x~2.0-2.1, "
     "which the POLICY cut to x~2.4, through the frame). pre_gate z lowered + corrective capped "
     "for top-bar headroom on the inbound pass."),
]:
    groups = [
        {"label": "old demos", "color": OLD, "trajs": old_demos(dem)},
        {"label": f"regen2 kept 50 (judge+clearance-clean)", "color": NEW, "trajs": new_kept(course)},
    ] + markers(scene)
    SECS.append((title, cloudviewer.viewer_html(scene, groups, note=note,
                                                elem_id=f"v_{course}", max_pts=40000)))

body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Regen-2 Demo Fans</title>
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
<h1>Regen-2 Demo Fans</h1>
<p class="sub">The regen2 dataset's kept trajectories (blue, 50 per task, every one
judge+clearance-passing at plan time, real-matched start distribution) over the scene clouds,
next to the old demos they replace (grey). Toggle the old demos off to see the new corridors
alone; drag to orbit, wheel to zoom, shift-drag to pan.</p>
{body}
</main>
"""
out = f"{SP}/regen2_demos.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
