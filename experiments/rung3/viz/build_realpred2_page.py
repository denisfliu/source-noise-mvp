"""Predicted-from-real v2 (2026-08-27): side-prompt fans, counterfactual CENTER-prompt
fans, and max-distrust fans at the least-confident anchors, over the real flights.

  python3 build_realpred2_page.py   (writes realpred.html — same artifact URL)
"""
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

Z = np.load("/home/dfliu/ctxrun/real_cf_chunks.npz", allow_pickle=True)
meta = json.loads(str(Z["meta"]))
PATHS = np.load("/home/dfliu/ctxrun/real_pred_chunks.npz")

SECS = []
for side in ("left", "right"):
    n = meta[side]
    get = lambda i, k: Z[f"{side}_{i}_{k}"] if f"{side}_{i}_{k}" in Z.files else None
    paths = [PATHS[k] for k in PATHS.files if k.startswith(f"{side}_path_")]
    sigs = np.array([float(get(i, "side_sig")) for i in range(n)])
    med = float(np.median(sigs))
    side_lo = [get(i, "side_traj") for i in range(n) if sigs[i] <= med]
    side_hi = [get(i, "side_traj") for i in range(n) if sigs[i] > med]
    ctr = [get(i, "ctr_traj") for i in range(n)]
    dis = [get(i, "distrust_traj") for i in range(n) if get(i, "distrust_traj") is not None]
    groups = [
        {"label": f"real flights ({len(paths)})", "color": [150, 150, 158], "trajs": paths},
        {"label": f"side-prompt fans, confident (sigma*<={med:.1f})", "color": [96, 235, 160],
         "trajs": side_lo},
        {"label": "side-prompt fans, uncertain", "color": [255, 171, 66], "trajs": side_hi},
        {"label": "CENTER-prompt fans (counterfactual task)", "color": [124, 168, 255],
         "trajs": ctr},
        {"label": "max-distrust fans (sigma_serve=cap) at least-confident anchors",
         "color": [235, 110, 210], "trajs": dis},
    ]
    note = ("Blue fans: the SAME real observations with the prompt swapped to the center "
            "task — where they bend away from the side-task fans toward +x, the task "
            "binding is doing real counterfactual work on real perception; where they "
            "don't, the binding is state-dominated there. Magenta fans: the pin fully "
            "distrusted (sigma_serve at cap) at the anchors where the head is LEAST "
            "confident — the flow's own vote when told to ignore the command.")
    SECS.append((side.upper(), cloudviewer.viewer_html(side, groups, note=note,
                                                       elem_id=f"v_{side}", max_pts=40000)))

body = "".join(f"<h2>real {t} flights</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Predicted From Real</title>
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
<h1>Predicted From Real</h1>
<p class="sub">Real flights (grey) with the policy's 50-step chunks generated from the real
observations at anchors along them: the trained task's fans (green/amber by head
confidence), the counterfactual center-task fans (blue), and the flow's unpinned vote at
the least-confident anchors (magenta). Toggle groups to isolate each question.</p>
{body}
</main>
"""
out = f"{SP}/realpred.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
