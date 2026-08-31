"""Ablation matrix page (2026-08-31): scratch-sketch attribution + the two new arms.

  python3 build_ablation_page.py
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

SECS = []
sk = [np.asarray(json.load(open(f"{RD}/sketch_orbit.json"))["points"], np.float32)[:, :3]]
g = [{"label": "orbit sketch", "color": [255, 171, 66], "trajs": sk},
     {"label": "xswap (reads the pin): 0.07-0.10 m tracking", "color": [96, 235, 160],
      "trajs": load(f"{RUN}/traj_app_orbit_*.npy")},
     {"label": "scratch (cannot read the pin): 0.45-0.68 m", "color": [240, 110, 110],
      "trajs": load(f"{RUN}/traj_scrsk_orbit_*.npy")}]
SECS.append(("the mechanism, in one picture — orbit through pin-reading vs pin-blind models",
             cloudviewer.viewer_html("right", g, elem_id="v_orbit", max_pts=40000, note=
    "Identical sketch pipeline, identical noise injection. The pin-trained model traces the "
    "circle at ~8 cm; scratch ignores it entirely (6x deviation, flies its habitual gate "
    "mission — and badly: the unreadable injected noise sits 4-6 sigma off its training "
    "distribution and costs it its usual 10/10, scoring 0/5). Sketches work THROUGH the "
    "source-noise channel; the channel is necessary, and injection without training is "
    "actively harmful.")))
groups = [{"label": "real flights", "color": [150, 150, 158],
           "trajs": [np.load(f"{RUN}/real_pred_chunks.npz")[k]
                     for k in np.load(f"{RUN}/real_pred_chunks.npz").files if k.startswith("right_path_")]}]
for arm, path, col in [("xswap (swap+sigma+real)", "synthpin_xswap.npz", [96, 235, 160]),
                       ("synthonly (no real data)", "synthpin_synthonly.npz", [124, 168, 255]),
                       ("nosig (no trust dial)", "synthpin_nosig.npz", [255, 171, 66])]:
    Z = np.load(f"{RUN}/{path}", allow_pickle=True)
    meta = json.loads(str(Z["meta"]))
    groups.append({"label": arm, "color": col,
                   "trajs": [Z[f"arr_{i}_headpin_traj"] for i, r in enumerate(meta)
                             if r["side"] == "right"]})
SECS.append(("real right anchors — each arm's own head", cloudviewer.viewer_html(
    "right", groups, elem_id="v_arms", max_pts=40000, note=
    "Own-head fans on the same real anchors. Gate-reaching: xswap 15, synthonly 12, nosig 7 "
    "— the SWAP drives the real-frame head gains, not sigma. Execution of real-oracle "
    "commands: nosig 3.3 deg (tightest — always-exact pin training), xswap 4.7, synthonly "
    "7.7 (real data buys ~1.6x execution fidelity but synth-only remains functional).")))
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Ablation Matrix</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;
padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-size:23px;margin:0 0 4px}} h2{{font-size:16px;margin:30px 0 8px;color:var(--acc)}}
.sub{{color:--mut;margin:0 0 18px;max-width:92ch;color:var(--mut)}}
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
<h1>Ablation Matrix</h1>
<p class="sub">Sim six cells: synthonly 40/40 judge (39/40 clean), nosig 40/40 + 40/40 —
sim atomics need neither real data nor the trust dial. The differences live on real frames
and in the mechanism control below.</p>
{body}
</main>
"""
open(f"{SP}/ablation_matrix.html", "w").write(page)
print("wrote ablation_matrix.html")
