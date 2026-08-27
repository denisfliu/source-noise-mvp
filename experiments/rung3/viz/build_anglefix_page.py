"""Pin rotation-correction fans over the real flights (2026-08-27).

  python3 build_anglefix_page.py   (writes anglefix.html)
"""
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

C = np.load("/home/dfliu/ctxrun/angle_correct.npz", allow_pickle=True)
mc = json.loads(str(C["meta"]))
S = np.load("/home/dfliu/ctxrun/angle_scratch.npz", allow_pickle=True)
ms = json.loads(str(S["meta"]))
PATHS = np.load("/home/dfliu/ctxrun/real_pred_chunks.npz")

SECS = []
for side in ("left", "right"):
    paths = [PATHS[k] for k in PATHS.files if k.startswith(f"{side}_path_")]
    t0, tf = [], []
    for i, r in enumerate(mc):
        if r["side"] == side:
            t0.append(C[f"arr_{i}_traj0"])
            tf.append(C[f"arr_{i}_trajfix"])
    ts = [S[f"arr_{i}_traj"] for i, r in enumerate(ms) if r["side"] == side]
    groups = [
        {"label": f"real flights ({len(paths)})", "color": [150, 150, 158], "trajs": paths},
        {"label": "gmsig3 fans, uncorrected (|err| ~10-20 deg)", "color": [255, 171, 66], "trajs": t0},
        {"label": "gmsig3 fans, PIN-ROTATED by -dtheta (|err| ~3-5 deg)", "color": [96, 235, 160], "trajs": tf},
        {"label": "scratch fans (same anchors — same scatter class)", "color": [200, 120, 210], "trajs": ts},
    ]
    note = ("Toggle amber vs green with the same real path visible: each green fan is the "
            "SAME anchor and observation as its amber twin, with the pin command rotated by "
            "the measured heading error — 'your angle of approach is incorrect' spoken in "
            "c-space, sigma=0. Purple (scratch) shows the scatter is the shared stack's, "
            "not the pin path's. Heading errors: left 10.1 -> 2.9 deg, right 20.3 -> 4.9.")
    SECS.append((side.upper(), cloudviewer.viewer_html(side, groups, note=note,
                                                       elem_id=f"v_{side}", max_pts=40000)))

body = "".join(f"<h2>real {t} flights</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Pin Rotation Correction</title>
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
<h1>Pin Rotation Correction</h1>
<p class="sub">The angular scatter and its fix, on real observations: amber fans are the
policy's raw plans from anchors on the real flights (grey); green fans are the identical
anchors re-served with the pin command rotated by the measured heading error; purple fans
are scratch on the same anchors (same scatter class — the error belongs to the shared
stack, not the pin). Dose-response gain 0.76: the rotation is a calibratable dial.</p>
{body}
</main>
"""
out = f"{SP}/anglefix.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
