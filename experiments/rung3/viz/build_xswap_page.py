"""S3 xswap verdict: own-head fans on real anchors, gmsig3 vs xswap (2026-08-28).

  python3 build_xswap_page.py
"""
import json
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
PATHS = np.load(f"{RUN}/real_pred_chunks.npz")
SECS = []
for side in ("right", "left"):
    paths = [PATHS[k] for k in PATHS.files if k.startswith(f"{side}_path_")]
    groups = [{"label": f"real flights ({len(paths)})", "color": [150, 150, 158], "trajs": paths}]
    for name, path, col in [("gmsig3 own-head fans", f"{RUN}/synthpin2_rows.npz", [255, 171, 66]),
                            ("XSWAP own-head fans", f"{RUN}/synthpin_xswap.npz", [96, 235, 160])]:
        Z = np.load(path, allow_pickle=True)
        meta = json.loads(str(Z["meta"]))
        groups.append({"label": name, "color": col,
                       "trajs": [Z[f"arr_{i}_headpin_traj"] for i, r in enumerate(meta)
                                 if r["side"] == side]})
    note = ("Same real anchors, each checkpoint's own head at calibrated trust... served "
            "sigma=0 in this probe. On RIGHT (the weak cell): xswap fans (green) are longer "
            "(0.79 vs 0.59 m), straighter (22.6 vs 26.8 deg), and reach the gate nearly "
            "twice as often (15 vs 8 crossings, 14/15 in-aperture, centered at s=0.40 vs "
            "the pilots' 0.45). On LEFT: functional parity; headings shift from pilot-style "
            "to planner-style — the trained-in twin.")
    SECS.append((f"{side} — own-head commands, before/after S3", cloudviewer.viewer_html(
        side, groups, elem_id=f"v_{side}", max_pts=40000, note=note)))
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Xswap Verdict</title>
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
<h1>Xswap Verdict</h1>
<p class="sub">S3 cross-supervised training (matched-pair chunk swap at p=0.5 on real
frames, mixed training): sim cells 40/40 route-clean AND 40/40 clearance-clean — the first
fully strict-clean sweep — and the real-frame own-head commands move toward twin grade
where they were weak. The deployed weights now carry the planner.</p>
{body}
</main>
"""
open(f"{SP}/xswap_verdict.html", "w").write(page)
print("wrote xswap_verdict.html")
