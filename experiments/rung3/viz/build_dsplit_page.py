"""dsplit verdict page (2026-08-28): stranded-head sim cells + real-anchor fans vs gmsig3.

  python3 build_dsplit_page.py   (writes dsplit_verdict.html)
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
SECS = []
for side in ("left", "right"):
    g = [
        {"label": f"gmsig3 (10/10)", "color": [96, 235, 160],
         "trajs": [np.load(f)[:, :3].astype(np.float32)
                   for f in sorted(glob.glob(f"{RUN}/traj_armgmsig3_{side}_*.npy"))]},
        {"label": "dsplit (0/10 — clean cruising, dead commands)", "color": [240, 110, 110],
         "trajs": [np.load(f)[:, :3].astype(np.float32)
                   for f in sorted(glob.glob(f"{RUN}/traj_armdsplit_{side}_*.npy"))]},
    ]
    SECS.append((f"sim {side} cell", cloudviewer.viewer_html(side, g, elem_id=f"v_{side}",
        max_pts=40000, note="Phase B (real-only fine-tune) moved the VLM features out from "
        "under the frozen phase-A head: readout R2 went negative, the served commands died, "
        "and the flights cruise cleanly without completing — representation drift inside "
        "one checkpoint.")))
Zg = np.load(f"{RUN}/synthpin2_rows.npz", allow_pickle=True)
Zd = np.load(f"{RUN}/synthpin_dsplit.npz", allow_pickle=True)
for name, Z, col in [("gmsig3", Zg, [255, 171, 66]), ("dsplit", Zd, [240, 110, 110])]:
    meta = json.loads(str(Z["meta"]))
    trajs = [Z[f"arr_{i}_headpin_traj"] for i, r in enumerate(meta) if r["side"] == "right"]
    if name == "gmsig3":
        groups = [{"label": "gmsig3 head fans (8 crossings)", "color": col, "trajs": trajs}]
    else:
        groups.append({"label": "dsplit head fans (3 crossings, 0.39 m — stranded)",
                       "color": col, "trajs": trajs})
SECS.append(("real right anchors — own-head fans, gmsig3 vs dsplit", cloudviewer.viewer_html(
    "right", groups, elem_id="v_fans", max_pts=40000,
    note="Same real anchors, each checkpoint's own head command: dsplit's head (red) "
    "produces short, aimless chunks — its features moved; its binding did not.")))
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Dsplit Verdict</title>
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
<h1>Dsplit Verdict</h1>
<p class="sub">The domain-split training experiment (synth learns the pin, real learns the
denoising, sequenced): a clean negative. Real-command execution was already saturated by
mixed training (no gain on any real-anchor metric) and the sequential phase stranded the
head (feature drift within one checkpoint). gmsig3 remains flagship; the domain split
survives as a SERVING doctrine, not a training one.</p>
{body}
</main>
"""
open(f"{SP}/dsplit_verdict.html", "w").write(page)
print("wrote dsplit_verdict.html")
