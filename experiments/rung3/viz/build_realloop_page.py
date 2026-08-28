"""Real-in-the-loop emulator ladder (2026-08-28).

  python3 build_realloop_page.py
"""
import glob
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
LB = {"rlv0": ("V0 retrieval-SYNTH sanity", [124, 168, 255]),
      "rla1": ("A1 head on retrieved REAL obs", [255, 171, 66]),
      "rla2": ("A2 sim-twin commands, real obs", [96, 235, 160])}
SCORE = {("rlv0","left"):"5/5",("rla1","left"):"0/5",("rla2","left"):"1/5",
         ("rlv0","right"):"0/5 (SANITY FAIL)",("rla1","right"):"0/5",("rla2","right"):"3/5"}
SECS = []
for side in ("left", "right"):
    g = [{"label": "gmsig3 sim reference (10/10)", "color": [150, 150, 158],
          "trajs": [np.load(f)[:, :3].astype(np.float32)
                    for f in sorted(glob.glob(f"{RUN}/traj_armgmsig3_{side}_*.npy"))[:5]]}]
    for tag, (lab, col) in LB.items():
        g.append({"label": f"{lab} ({SCORE[(tag, side)]})", "color": col,
                  "trajs": [np.load(f)[:, :3].astype(np.float32)
                            for f in sorted(glob.glob(f"{RUN}/traj_{tag}_{side}_*.npy"))]})
    note = ("Closed-loop sim physics with retrieved corpus frames as the policy's eyes. "
            "V0 (blue) replays SYNTH frames — it must match the grey reference for the "
            "instrument to be valid: it does on left (5/5) and FAILS on right (0/5, "
            "retrieval frame-jumping breaks the narrow-gate servoing), so right-cell "
            "numbers are directional only. A1 (amber): real eyes, own head. A2 (green): "
            "real eyes, sim-twin commands.")
    SECS.append((f"{side} cell", cloudviewer.viewer_html(side, g, elem_id=f"v_{side}",
                                                         max_pts=40000, note=note)))
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Real-In-The-Loop Ladder</title>
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
<h1>Real-In-The-Loop Ladder</h1>
<p class="sub">First closed-loop measurement of real-perception flight, via retrieved real
frames inside sim physics. Read with the sanity caveat: the retrieval instrument itself
breaks the right cell (V0 0/5), so treat right as directional. The consistent signal:
sim-twin commands (green) beat the own-head arm (amber) in both cells and restore
trajectory coherence.</p>
{body}
</main>
"""
open(f"{SP}/realloop.html", "w").write(page)
print("wrote realloop.html")
