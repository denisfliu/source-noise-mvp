"""What the policy would fly from the real flights (2026-08-27): real trajectories (grey)
with the policy's generated 50-step chunks anchored along them — full serve path (real obs
-> head c -> pinned noise -> flow at calibrated sigma). Chunk fans split by the head's own
sigma* (confident vs uncertain) at the pooled median.

  python3 build_realpred_page.py   (writes realpred.html)
"""
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

Z = np.load("/home/dfliu/ctxrun/real_pred_chunks.npz")
SECS = []
for side in ("left", "right"):
    paths = [Z[k] for k in Z.files if k.startswith(f"{side}_path_")]
    chunks = [Z[k] for k in Z.files if k.startswith(f"{side}_chunk_")]
    sig = Z[f"{side}_sig"]
    med = float(np.median(np.concatenate([Z["left_sig"], Z["right_sig"]])))
    lo = [c for c, s in zip(chunks, sig) if s <= med]
    hi = [c for c, s in zip(chunks, sig) if s > med]
    groups = [
        {"label": f"real flights ({len(paths)})", "color": [150, 150, 158], "trajs": paths},
        {"label": f"predicted chunks, confident head (sigma* <= {med:.1f})",
         "color": [96, 235, 160], "trajs": lo},
        {"label": "predicted chunks, uncertain head", "color": [255, 171, 66], "trajs": hi},
    ]
    note = ("Each green/amber fan starts ON a real flight (grey) and shows the 50 steps the "
            "policy would fly from that real observation — the real path's own continuation "
            "is the ground truth to compare against. Divergence between fan and grey "
            "continuation IS the behavior gap (0.62 cstd median, worst in the endgame); "
            "amber fans are where the trust dial already knows it is unsure.")
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
<p class="sub">The policy run on REAL observations: at anchors along each real flight (grey),
the full serve path (head-predicted pin, calibrated trust) generates a 50-step chunk, drawn
from the anchor. Where the fans hug the grey continuation, sim-trained prediction matches
real flying; where they peel off, you are looking at the behavior gap the pin-gap probe
measured. Chunks are colored by the head's own confidence.</p>
{body}
</main>
"""
out = f"{SP}/realpred.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
