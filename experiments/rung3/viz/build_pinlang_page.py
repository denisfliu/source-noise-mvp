"""Pin Language page (2026-08-29): (1) the whole extreme-pose sweep in ONE cloud,
(2) the 16-dim command vocabulary decoded as eigen-movements, (3) pin-intent curves
('the pin said THIS movement here') along the behind-the-start flight.

  python3 build_pinlang_page.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
RD = os.path.dirname(SP)
U = np.load(f"{RD}/pin_U_mh16.npy").astype(np.float32)
NS = json.load(open(os.path.join(os.path.dirname(RD), "..", "assets/openpi/pi0_gate3/local/gate_nav3/norm_stats.json")))["norm_stats"]["actions"]
amean = np.asarray(NS["mean"], np.float32); astd = np.asarray(NS["std"], np.float32)
GA = np.array([0.195, -1.348]); GB = np.array([0.924, -0.952]); CEN = (GA + GB) / 2

def decode_c(c, anchor):
    """U c -> denormalized coarse chunk -> integrated path from anchor (the pin's words)."""
    a = (U @ c).reshape(50, 32)[:, :3] * astd[:3] + 0.0   # deltas around mean~0
    return np.concatenate([anchor[None], anchor + np.cumsum(a, axis=0)]).astype(np.float32)

SECS = []
# ---- 1. the sweep at a glance -------------------------------------------------------
POSES = [("mx100_m1_26_1_50", 100, -1.26, 1.50, "behind start", [96, 235, 160]),
         ("mx180_0_0", 180, 0, 0, "180 flip", [255, 171, 66]),
         ("mxm35_1_34_0_75", -35, 1.34, 0.75, "goal doorstep", [124, 168, 255]),
         ("mx90_0_74_1_95", 90, 0.74, 1.95, "north 90", [235, 110, 210]),
         ("mx90_m0_26_0_85", 90, -0.26, 0.85, "close-in perp", [255, 230, 90]),
         ("mxm45_0_0", -45, 0, 0, "-45 rerun", [110, 220, 235]),
         ("mx0_0_5_m0_3", 0, 0.5, -0.3, "translated rerun", [200, 150, 120])]
groups = []
for tag, dyaw, dx, dy, name, col in POSES:
    th = math.radians(dyaw)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = CEN - R @ CEN + np.array([dx, dy])
    a2, b2 = R @ GA + t, R @ GB + t
    ap = np.array([[a2[0], a2[1], 0.2], [b2[0], b2[1], 0.2], [b2[0], b2[1], 1.95],
                   [a2[0], a2[1], 1.95], [a2[0], a2[1], 0.2]], np.float32)
    trajs = [np.load(f)[:, :3].astype(np.float32)
             for f in sorted(glob.glob(f"{RUN}/traj_{tag}_*.npy"))] + [ap]
    groups.append({"label": f"{name} (5/5)", "color": col, "trajs": trajs})
SECS.append(("the whole sweep, one room", cloudviewer.viewer_html(
    "right", groups, elem_id="v_all", max_pts=40000, note=
    "All seven extreme poses at once — each color is one gate pose (bright rectangle) with "
    "its five flights. The background cloud shows the gate at its ORIGINAL pose; each "
    "rectangle is where it actually stood for that cell. Toggle poses to unclutter.")))

# ---- 2. the command vocabulary ------------------------------------------------------
CH = {0: ("x", [240, 110, 110]), 1: ("y", [96, 235, 160]), 2: ("z", [124, 168, 255]),
      3: ("yaw", [255, 230, 90])}
H80 = [5, 11, 23, 45]
vg = []
for ch_i in range(4):
    name, col = CH[ch_i]
    trajs = []
    for h_i in range(4):
        k = h_i * 4 + ch_i
        raw = decode_c(np.eye(16, dtype=np.float32)[k], np.zeros(3, np.float32))
        ext = np.abs(raw).max()
        glyph = raw / max(ext, 1e-6) * 0.9
        origin = np.array([1.0 + 2.0 * h_i, 4.0 - 1.1 * ch_i, 1.0], np.float32)
        trajs.append(origin + glyph)
    vg.append({"label": f"{name}-words (columns: h~5, 11, 23, 45)", "color": col,
               "trajs": trajs})
SECS.append(("the 16-word vocabulary (eigen-movements)", cloudviewer.viewer_html(
    "vocab", vg, elem_id="v_vocab", max_pts=2000, note=
    "Type-specimen sheet, no room: each curve is one basis dimension decoded to the "
    "movement it commands (U e_k, denormalized, integrated; shape-normalized to ~0.9 m — "
    "true 2-sigma sizes are 5-35 cm). Rows are channels (x red, y green, z blue, yaw "
    "amber), columns are horizons, short to long left to right. Short-horizon words move "
    "then hold; long-horizon words keep moving through the whole chunk; yaw words curl "
    "because heading rotates the ongoing motion. Every command the pin ever carries is a "
    "weighted sentence of these sixteen.")))

# ---- 3. pin-intent along the behind-start flight ------------------------------------
sk = json.load(open(f"{RD}/sketch_mg_mx100_m1_26_1_50.json"))
pts = np.asarray(sk["points"], np.float32)
s_arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1))])
n = max(int(s_arc[-1] / sk["step_m"]), 52)
uu = np.linspace(0, s_arc[-1], n)
P = np.stack([np.interp(uu, s_arc, pts[:, k]) for k in range(3)], 1)
A = np.zeros((n - 1, 7), np.float32); A[:, :3] = np.diff(P, axis=0)
F = np.load(f"{RUN}/traj_mx100_m1_26_1_50_1.npy")[:, :3].astype(np.float32)
intents = []
i = 0
for t in range(0, min(len(F), 320), 50):
    pos = F[t]
    w = P[i:i + 90]
    i += min(int(np.linalg.norm(w - pos, axis=1).argmin()), 65)
    if i >= n - 52: break
    seg = np.zeros((50, 32), np.float32)
    off = pos - P[i]
    wts = np.maximum(0.0, 1.0 - np.arange(1, 51) / 20)[:, None]
    tgt = P[i + 1:i + 51] + off[None, :] * wts
    seg[:, :3] = (np.diff(np.concatenate([pos[None], tgt]), axis=0) - amean[:3]) / (astd[:3] + 1e-6)
    c = seg.reshape(-1) @ U
    intents.append(decode_c(c, pos))
g3 = [{"label": "sketch (orange)", "color": [255, 171, 66], "trajs": [pts[:, :3]]},
      {"label": "flight #1 (green)", "color": [96, 235, 160], "trajs": [F]},
      {"label": "pin intent at each replan ('the pin said THIS')", "color": [235, 110, 210],
       "trajs": intents}]
SECS.append(("what the pin said, replan by replan (behind-start flight)",
             cloudviewer.viewer_html("right", g3, elem_id="v_intent", max_pts=25000, note=
    "Magenta curves: the served command decoded back into its coarse movement at each "
    "replan anchor — the literal content of the noise. The green flight is what the flow "
    "made of it. Where magenta and green agree in shape but green is richer, you are "
    "seeing the factorization: coarse from the pin, residual from the model.")))

body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Pin Language</title>
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
<h1>Pin Language</h1>
<p class="sub">Three views of the same object: the sweep the pin just flew (one room, seven
gate poses), the sixteen words the pin speaks (each basis dimension decoded to its
movement), and a flight annotated with what the pin said at every replan.</p>
{body}
</main>
"""
open(f"{SP}/pin_language.html", "w").write(page)
print("wrote pin_language.html")
