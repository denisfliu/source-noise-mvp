"""Overnight pin applications (2026-08-30): tempo verb, orbit, figure-8.

  python3 build_pinapps_page.py
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

def load(pat):
    return [np.load(f)[:, :3].astype(np.float32) for f in sorted(glob.glob(pat))]

def sketch(tag):
    return [np.asarray(json.load(open(f"{RD}/sketch_{tag}.json"))["points"], np.float32)[:, :3]]

SECS = []
g = [{"label": "0.6x tempo (gate at 88 steps)", "color": [124, 168, 255], "trajs": load(f"{RUN}/traj_app_tempo06_*.npy")},
     {"label": "1.0x tempo (53 steps)", "color": [96, 235, 160], "trajs": load(f"{RUN}/traj_app_tempo10_*.npy")},
     {"label": "1.5x tempo (37 steps)", "color": [255, 171, 66], "trajs": load(f"{RUN}/traj_app_tempo15_*.npy")}]
SECS.append(("the tempo verb — same route, three commanded speeds", cloudviewer.viewer_html(
    "right", g, elem_id="v_tempo", max_pts=40000, note=
    "Identical 4-point sketch, resampled at 0.6x / 1.0x / 1.5x demo speed. Realized "
    "crossing times: 88 / 53 / 37 steps — gains 0.60 and 1.43 against commanded 0.6 and "
    "1.5. Pace is a near-unit-gain command dial. All 15 flights 5/5 judge; note the "
    "clearance cost off-tempo (0/5 clean at both extremes vs 2/5 at 1.0x) — the flow's "
    "crossing finesse is tuned at demo pace.")))
for tag, title, note in [
    ("orbit", "the orbit — 1.5 loops around the gate (zero demos contain an orbit)",
     "Five flights, 100% arc completion, tracking median 0.07-0.10 m, all clearance-clean. "
     "The judge's 'fail' is expected — orbiting crosses the aperture plane repeatedly by "
     "design. A behavior class absent from every training demo, composed entirely from "
     "displacement words."),
    ("fig8", "the figure-8 — sustained curvature reversal in open space",
     "Two opposite-handed lobes: the hardest shape to speak in prefix-sum words. 100% "
     "completion, 0.09 m median tracking, clean.")]:
    g = [{"label": "sketch", "color": [255, 171, 66], "trajs": sketch(tag)},
         {"label": "flights (5)", "color": [96, 235, 160], "trajs": load(f"{RUN}/traj_app_{tag}_*.npy")}]
    SECS.append((title, cloudviewer.viewer_html("right", g, elem_id=f"v_{tag}",
                                                max_pts=40000, note=note)))
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>New Verbs</title>
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
<h1>New Verbs</h1>
<p class="sub">Overnight applications of the pin (xswap checkpoint, sigma=0, carrot on):
pace as a commandable dial, and two motion programs — orbit and figure-8 — that exist in
zero training demonstrations, flown at 8-10 cm fidelity from displacement words alone.</p>
{body}
</main>
"""
open(f"{SP}/new_verbs.html", "w").write(page)
print("wrote new_verbs.html")
