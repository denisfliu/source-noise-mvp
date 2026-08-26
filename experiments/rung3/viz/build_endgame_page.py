"""Build the twin-endgame review page: ctl (MSE) vs gmm (MDN) trajectories over the scene point
clouds, tails emphasized, plus the phase-binned basis-capture table and the served-command drift
readout. The page exists to answer one question: is the failing tail a command-content problem,
an expressiveness problem, or both — and what should the pin change to capture those movements.

  python3 build_endgame_page.py         (writes twin_endgame.html next to this file)
"""
import glob
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RUN = "/home/dfliu/ctxrun"
RD = os.path.dirname(SP)
TAIL = 100  # steps of each rollout drawn as the emphasized tail

DEMO_EPS = {"left": range(100, 150), "right": range(150, 200), "cfl": range(0, 50)}
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}


def box_edges(c, h):
    corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return [corners[[a, b]] for a, b in idx]


def demos(key, n=10):
    out = []
    for e in list(DEMO_EPS[key])[:n]:
        d = np.load(f"{RD}/data_gate_synth/ep_{e:04d}.npz", allow_pickle=True)
        out.append(d["state"][:, :3].astype(np.float32))
    return out


def rollouts(pattern):
    fs = sorted(glob.glob(pattern), key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return [np.load(f)[:, :3].astype(np.float32) for f in fs]


def tails(trajs):
    return [t[-TAIL:] for t in trajs]


def markers(scene_key):
    return [{"label": "goal box (judge)", "color": [248, 210, 90], "trajs": box_edges(GOAL_C, GOAL_H)},
            {"label": "gate aperture (judge)", "color": [124, 208, 240],
             "trajs": [np.array(APERTURE[k] + [APERTURE[k][0]], np.float32)
                       for k in {"left": ["left"], "right": ["right"], "center": ["center"]}[scene_key]]}]


CTL, GMM, DEMO = [82, 200, 130], [255, 171, 66], [128, 136, 150]
CTL_T, GMM_T = [180, 255, 205], [255, 226, 170]
MH, MH_T = [232, 121, 249], [246, 200, 255]

V = []
for scene, dk, note in [
    ("left", "left",
     "ctl settles inside the goal box 10/10; gmm transits the same aperture then overshoots ~0.7 m "
     "past the box in -y, every trial (endpoints cluster at (1.98,-1.60) vs the box at (1.52,-0.62))."),
    ("right", "right",
     "Mirror image: gmm settles in-box 10/10 strict; ctl crosses the aperture then sails on, "
     "endpoints (2.66,-2.21) +/- 0.5 - the b2-right settle miss, reproduced. Demos show the demo "
     "route curls back to the same shared goal box."),
    ("center", "cfl",
     "Center-from-left: gmm recovers the ROUTE (8/10 transit, ctl 1/10) - mode commitment, not "
     "sampling, is doing it - but no arm settles; ctl's five grazes (min clearance 0.001-0.14 m) "
     "hug the frame's left post."),
]:
    groups = [{"label": "demos", "color": DEMO, "trajs": demos(dk)}]
    if scene in ("left", "right"):
        c = rollouts(f"{RUN}/traj_armctl_{scene}_*.npy")
        g = rollouts(f"{RUN}/traj_armgmm_{scene}_*.npy")
        m = rollouts(f"{RUN}/traj_armgmmmh_{scene}_*.npy")
    else:
        c = rollouts(f"{RUN}/traj_ctl_cfl_*.npy")
        g = rollouts(f"{RUN}/traj_gmm_cfl_*.npy")
        m = rollouts(f"{RUN}/traj_gmmmh_cfl_*.npy") + rollouts(f"{RUN}/traj_gmmmh_cfr_*.npy")
    groups += [
        {"label": "ctl (MSE head)", "color": CTL, "trajs": c},
        {"label": "ctl tails (last 100)", "color": CTL_T, "trajs": tails(c)},
        {"label": "gmm (MDN head)", "color": GMM, "trajs": g},
        {"label": "gmm tails (last 100)", "color": GMM_T, "trajs": tails(g)},
        {"label": "gmmmh (MDN x mh16)", "color": MH, "trajs": m},
        {"label": "gmmmh tails (last 100)", "color": MH_T, "trajs": tails(m)},
    ] + markers(scene)
    V.append((scene, cloudviewer.viewer_html(scene, groups, note=note, elem_id=f"v_{scene}",
                                             max_pts=45000)))

CAPTURE_ROWS = [
    ("center_from_left", 0.772, 0.784, 0.800, 0.810, 0.617),
    ("center_from_right", 0.765, 0.727, 0.826, 0.827, 0.658),
    ("left", 0.610, 0.785, 0.513, 0.668, 0.566),
    ("right", 0.648, 0.640, 0.428, 0.553, 0.432),
    ("ALL (pooled)", 0.757, 0.780, 0.837, 0.807, 0.611),
]


def shade(v):
    if v >= 0.75:
        return ""
    return ' class="warn"' if v >= 0.55 else ' class="bad"'


cap_rows = "".join(
    f"<tr><td>{t}</td>" + "".join(f"<td{shade(v)}>{v:.3f}</td>" for v in vals) + "</tr>"
    for t, *vals in CAPTURE_ROWS)

DRIFT_ROWS = [
    ("ctl / left", "10/10 strict", "14%", "0.44", "0.49 on-manifold"),
    ("gmm / left", "transit, 0/10 goal", "44%", "0.57", "0.72 on-manifold"),
    ("gmm / right", "10/10 strict", "59%", "0.21", "0.28 on-manifold"),
    ("ctl / right", "transit, 0/10 goal", "69%", "0.50", "0.59 off-manifold"),
]
drift_rows = "".join(
    f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>"
    for a, b, c, d, e in DRIFT_ROWS)

html = f"""<title>Endgame Ownership</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0;
--ok:#52c882;--warnc:#ffab42;--badc:#ff6b6b}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;
padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-size:23px;margin:0 0 4px;letter-spacing:.2px}}
h2{{font-size:16px;margin:34px 0 8px;color:var(--acc)}}
.sub{{color:var(--mut);margin:0 0 20px;max-width:88ch}}
.strip{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 6px}}
.cell{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:9px 13px;
font:12px/1.5 ui-monospace,Menlo,monospace}}
.cell b{{display:block;font-size:11px;color:var(--mut);letter-spacing:.6px}}
.cell .ok{{color:var(--ok)}} .cell .no{{color:var(--badc)}} .cell .pt{{color:var(--warnc)}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:14px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;
font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}}
.sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}}
.v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
table{{border-collapse:collapse;font:13px ui-monospace,Menlo,monospace;margin:10px 0}}
.twrap{{overflow-x:auto}}
th,td{{border:1px solid var(--line);padding:6px 11px;text-align:right;
font-variant-numeric:tabular-nums}}
th{{color:var(--mut);font-weight:600}} td:first-child,th:first-child{{text-align:left}}
td.warn{{color:var(--warnc)}} td.bad{{color:var(--badc);font-weight:700}}
p{{max-width:88ch}} li{{max-width:86ch;margin:5px 0}}
kbd{{font:12px ui-monospace,monospace;background:var(--card);border:1px solid var(--line);
border-radius:4px;padding:1px 5px}}
</style>
<main>
<h1>Endgame Ownership</h1>
<p class="sub">ctl (MSE joint head) vs gmm (MDN head, argmax serve) — same recipe, basis
(pin_U_gate_rrr_k5, sha ac49ae6b), seed 42, APC=50, strict scoring (posthoc judge + clearance).
Both twins transit both gates 10/10; the SETTLE is what flips owners. Drag to orbit, wheel to
zoom, shift-drag to pan; toggle groups in each legend.</p>

<div class="strip">
<div class="cell"><b>LEFT</b> ctl <span class="ok">10/10 strict</span> · gmm <span class="pt">transit only</span></div>
<div class="cell"><b>RIGHT</b> ctl <span class="pt">transit only</span> · gmm <span class="ok">10/10 strict</span></div>
<div class="cell"><b>CFL</b> ctl <span class="no">1/10 transit</span> · gmm <span class="pt">8/10 transit</span></div>
<div class="cell"><b>CFR</b> ctl <span class="no">0/10</span> · gmm <span class="no">1/10 transit</span></div>
<div class="cell"><b>CMPL</b> ctl <span class="no">0/5</span> · gmm <span class="pt">1/5 both-gates</span></div>
<div class="cell"><b>CMPR</b> ctl <span class="no">0/5</span> · gmm <span class="no">0/5</span></div>
</div>

<h2>Left scene — ctl settles, gmm overshoots</h2>
<div class="vc">{V[0][1]}</div>

<h2>Right scene — gmm settles, ctl sails past</h2>
<div class="vc">{V[1][1]}</div>

<h2>Center scene — routing recovered, no settle anywhere</h2>
<div class="vc">{V[2][1]}</div>

<h2>Can the pin even express the tail? Phase-binned basis capture</h2>
<p>Fraction of demo-chunk variance inside span(U) (normalized zero-padded H=50 chunks, the flow's
training target), by chunk-start phase and per task. Pooled capture looks healthy because
between-task variance is easy; the <em>within-task</em> numbers are what steering precision uses.</p>
<div class="twrap"><table>
<tr><th>task</th><th>[0,.25)</th><th>[.25,.5)</th><th>[.5,.75)</th><th>[.75,1]</th>
<th>stop (t&gt;T−H)</th></tr>
{cap_rows}
</table></div>
<p>The two failing settles live exactly in the weak cells: <b>right mid-flight and stop capture
0.43</b>, left 0.51–0.57. Roughly half the tail movement is invisible to the command channel on
the flat basis — the box measured the same structure (flat 0.34 vs mh16 0.81 on its
stop-segment definition).</p>

<h2>Is the failing tail command-content or expressiveness? Both — split by arm</h2>
<p>Served command vs the nearest demo state's own command (clog_analysis, chunk-displacement
metres), per arm and side:</p>
<div class="twrap"><table>
<tr><th>arm / side</th><th>outcome</th><th>&gt;0.3 m off demo manifold</th>
<th>mean |served−demo| c (m)</th><th>worst regime</th></tr>
{drift_rows}
</table></div>
<ul>
<li><b>Each twin is command-accurate exactly on the side it wins</b> (0.21–0.44 m mean error on
wins vs 0.50–0.57 m on losses).</li>
<li><b>gmm-left is wrong ON the demo manifold</b> (0.72 m error at &lt;0.15 m from demo states):
command content, not covariate shift.</li>
<li><b>ctl-right drifts off-manifold then commands worsen</b> (69% of replans &gt;0.3 m off,
error 0.59 m out there): accumulated mid-flight imprecision, the known restoring-field gap.</li>
</ul>

<h2>Reading</h2>
<ul>
<li>The settle failure is over-determined: the flat K=5 basis can only express ~43–57% of the
within-task tail movement (table above), and the head's tail commands are also wrong in-span.
Fixing either alone is unlikely to close a 0/10.</li>
<li>The measured, already-validated lever for expressiveness is the <b>multi-horizon basis
(mh16)</b>: box finding, stop-segment capture 0.34 → 0.81. Its rebuild recipe exists
(mh_basis_audit.py / RESEARCH_LOG 2026-08-13); tail-weighting and per-band soft pins stay
rejected as regime patches.</li>
<li>π(o) came back degenerate (one component, 0.74–1.0 everywhere): the MDN behaves as
heteroscedastic regression. Whatever helps its right-settle is not mixture routing — worth the
σ-by-phase probe before crediting a mechanism.</li>
<li>Seed replication for both twins is still required before "endgame ownership flipped on the
head axis" becomes a claim (gen16 precedent: ownership flipped on seed alone).</li>
</ul>
</main>
"""
out = f"{SP}/twin_endgame.html"
open(out, "w").write(html)
print(f"wrote {out} ({len(html)/1e6:.1f} MB)")
