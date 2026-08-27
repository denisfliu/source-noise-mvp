"""Synth pins on real observations, in the cloud (2026-08-27).

  python3 build_synthpin_page.py   (writes synthpin.html)
"""
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

Z = np.load("/home/dfliu/ctxrun/synthpin2_rows.npz", allow_pickle=True)
meta = json.loads(str(Z["meta"]))
PATHS = np.load("/home/dfliu/ctxrun/real_pred_chunks.npz")

SECS = []
for side in ("right", "left"):
    paths = [PATHS[k] for k in PATHS.files if k.startswith(f"{side}_path_")]
    g = {t: [] for t in ("headpin", "synthheadpin", "synthpin", "realpin")}
    for i, r in enumerate(meta):
        if r["side"] == side:
            for t in g:
                g[t].append(Z[f"arr_{i}_{t}_traj"])
    groups = [
        {"label": f"real flights ({len(paths)})", "color": [150, 150, 158], "trajs": paths},
        {"label": "head pin (status quo: slow, scattered)", "color": [255, 171, 66], "trajs": g["headpin"]},
        {"label": "SYNTH-HEAD pin (sim twin's head command)", "color": [235, 110, 210], "trajs": g["synthheadpin"]},
        {"label": "synth-oracle pin (matched sim state's demo command)", "color": [124, 168, 255], "trajs": g["synthpin"]},
        {"label": "real-oracle pin (ceiling)", "color": [96, 235, 160], "trajs": g["realpin"]},
    ]
    note = ("Same real anchors, three commands. Blue = the matched SIM state's oracle pin "
            "served on the real observation: it flies the planner's route — visibly "
            "different line than the grey pilot flights, but full speed and 11/11 "
            "in-aperture at the right gate (s=0.32 vs pilots' 0.45). Amber = the head's own "
            "command (under-sped 0.60 m chunks, the scatter Denis flagged). Green = real "
            "oracle command (4 deg residual — the execution ceiling).")
    SECS.append((side.upper(), cloudviewer.viewer_html(side, groups, note=note,
                                                       elem_id=f"v_{side}", max_pts=40000)))

# closed-loop real-demo-as-sketch section
import glob as _glob
rd_trajs = [np.load(f)[:, :3].astype(np.float32)
            for f in sorted(_glob.glob("/home/dfliu/ctxrun/traj_realdemo_*.npy"))]
src = np.load("/home/dfliu/code/source-noise-mvp/experiments/rung3/data_gate_real/ep_0050.npz",
              allow_pickle=True)["state"][:, :3].astype(np.float32)
g2 = [
    {"label": "source REAL flight (ep050)", "color": [150, 150, 158], "trajs": [src]},
    {"label": "closed-loop sim flights under its pin sketch (5/5 strict)",
     "color": [96, 235, 160], "trajs": rd_trajs},
]
SECS.append(("REAL DEMO AS SKETCH — closed-loop in sim", cloudviewer.viewer_html(
    "right", g2, elem_id="v_rd", max_pts=40000, note=
    "The mirror direction, closed loop: a real pilot's flight (grey) downsampled to a "
    "10-point pin sketch and flown in the sim right cell — 5/5 judge, 5/5 clearance-clean "
    "(min 0.34-0.37 m). Real behavior is commandable and certifiable in sim.")))
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Synth Pins In Real</title>
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
<h1>Synth Pins In Real</h1>
<p class="sub">'Plan in sim, fly in real through the pin', visualized: at anchors on the real
flights (grey), the same real observation is served with three different commands. The blue
fans are sim-authored routes executing on real perception — task-valid at the right gate.
Right scene first (the cell in question); toggle groups to compare.</p>
{body}
</main>
"""
out = f"{SP}/synthpin.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
