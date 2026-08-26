"""Navigable trajectory explorer for every command source measured at claim tier.

One viewer per scene, all arms overlaid as toggleable groups, so the gate-bias split (coupled arms own
the left gate, the detached arm owns the right) can be inspected geometrically rather than as a table.
Drag to orbit, wheel to zoom, shift-drag to pan."""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu/code/source-noise-mvp/experiments/rung3")
import cloudviewer

SP = os.path.dirname(os.path.abspath(__file__))
RUN = "/home/ubuntu/ctxrun"
OUT = f"{SP}/trajectory_explorer.html"
THRESH = 0.18

# (label, traj prefix, scores file, traj tag in that file, colour, trials, note)
ARMS = [
    ("one-hot scaffold", "swap10", f"{RUN}/swap10_scores.txt", "traj_swap10_", [110, 200, 140], 10,
     "Enumerates the four tasks — the reference, not the goal."),
    ("B2 coupled (λ=1)", "evb210", f"{RUN}/ev_b210_scores.txt", "traj_evb210_", [90, 170, 245], 10,
     "Head loss backprops into the VLM. Owns the left gate."),
    ("B2 coupled (λ=0.3)", "armb2lam03", f"{RUN}/arm_b2lam03_scores.txt", "traj_armb2lam03_", [230, 80, 230], 10,
     "Partial coupling: left stays 10/10 strict AND the right route returns — "
     "all 10 transit clean; 7 overshoot the hover box (no stop)."),
    ("B1 detached", "b1x10", f"{RUN}/b1x10_scores.txt", "traj_b1x10_", [240, 170, 70], 10,
     "Head trained on stop-gradient features. Owns the right gate."),
    ("B1s detached + state", "evb1s10", f"{RUN}/ev_b1s10_scores.txt", "traj_evb1s10_", [200, 130, 210], 10,
     "Adding the 32-d state input costs clearance on both gates."),
    ("external prior, matched", "evbpr10", f"{RUN}/ev_bpr10_scores.txt", "traj_evbpr10_", [235, 120, 120], 10,
     "Fitted after training on the served flow's own features."),
    ("external prior, featfix", "evff10", f"{RUN}/ev_ff10_scores.txt", "traj_evff10_", [150, 150, 160], 10,
     "Cache rebuilt for the served flow; still 0/10 on both."),
    ("B1long (15k, annealed)", "armb1long", f"{RUN}/arm_b1long_scores.txt", "traj_armb1long_", [120, 190, 200], 5,
     "Three times the optimisation, lower loss, right gate lost."),
]


def verdicts(scores, tag, side):
    s = open(scores).read()
    i = s.index(f", {side}")
    j = s.find(", right")
    blk = s[i:j] if (side == "left" and j > i) else s[i:]
    tr = {m[0]: m[1] == "True" for m in
          re.findall(rf"({tag}\w+\.npy)\s+transit=\S+ wrong_dir=\d+ goal=\w+\s+SUCCESS=(\w+)", blk)}
    cl = {m[0]: float(m[1]) for m in re.findall(rf"({tag}\w+\.npy)\s+min-clearance ([0-9.]+) m", blk)}
    return tr, cl


def load(prefix, side):
    fs = sorted(glob.glob(f"{RUN}/traj_{prefix}_{side}_*.npy"),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return fs, [np.load(f)[:, :3] for f in fs]


GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])

# true gate apertures = the judge's miss_gate.corners rectangles (configs/safety/*.yaml, viz only)
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}
SCENE_APERTURES = {"left": ["left"], "right": ["right"], "center": ["center"],
                   "left_and_center": ["left", "center"], "right_and_center": ["right", "center"]}


def aperture_group(scene):
    quads = [np.array(APERTURE[k] + [APERTURE[k][0]], np.float32) for k in SCENE_APERTURES[scene]]
    return {"label": "gate aperture (judge)", "color": [90, 220, 255], "trajs": quads}


def box_edges(c, h):
    corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return [corners[[a, b]] for a, b in idx]


DATA = {}
for side in ("left", "right"):
    groups, table = [], []
    for label, prefix, scores, tag, colour, n, note in ARMS:
        fs, trajs = load(prefix, side)
        tr, cl = verdicts(scores, tag, side)
        strict, judge, clean = [], 0, 0
        for f, t in zip(fs, trajs):
            b = os.path.basename(f)
            ok, c = tr.get(b, False), cl.get(b, 0.0)
            judge += ok
            clean += c >= THRESH
            if ok and c >= THRESH:
                strict.append(t)
        # strict successes drawn solid; the rest dimmer so the successful routes stand out
        groups.append({"label": f"{label} — {len(strict)}/{n} strict", "color": colour, "trajs": trajs})
        table.append((label, note, judge, clean, len(strict), n))
    # goal box wireframe: success needs >=1 post-transit frame inside it — the λ=0.3 right-gate
    # failures are flights that sail past this box
    groups.append({"label": "goal box (hover target)", "color": [250, 210, 80],
                   "trajs": box_edges(GOAL_C, GOAL_H)})
    groups.append(aperture_group(side))
    DATA[side] = (groups, table)

# ---- center + compound sections (2026-08-12: lam=0.3 flown on CFL/CFR and the two novel
# compound prompts; one-hot record system and the langprior arm as references) ----

def parse_pairs(path, tag):
    """(judge SUCCESS, clearance CLEAN) per trajectory from a gate_success+gate_clearance dump."""
    s = open(path).read()
    tr = {m[0]: m[1] == "True" for m in
          re.findall(rf"({tag}\w+\.npy)\s+transit=\S+ wrong_dir=\d+ goal=\w+\s+SUCCESS=(\w+)", s)}
    cl = {m[0]: float(m[1]) for m in re.findall(rf"({tag}\w+\.npy)\s+min-clearance ([0-9.]+) m", s)}
    return tr, cl


def parse_gates(path, tag):
    s = open(path).read()
    g = {m[0]: int(m[1]) for m in re.findall(rf"({tag}\w+\.npy)\s+gates=(\d)/2", s)}
    ok = {m[0]: m[1] == "True" for m in re.findall(rf"({tag}\w+\.npy)\s+gates=.*SUCCESS=(\w+)", s)}
    c = open(f"{RUN}/cmp_clearance.txt").read()
    cl = {m[0]: float(m[1]) for m in re.findall(rf"({tag}\w+\.npy)\s+min-clearance ([0-9.]+) m", c)}
    return g, ok, cl


def trajs_of(prefix):
    fs = sorted(glob.glob(f"{RUN}/traj_{prefix}_*.npy"),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return fs, [np.load(f)[:, :3] for f in fs]


CENTER_SETS = {  # approach -> [(label, prefix, scores file, colour, note)]
    "CFL (center from left)": [
        ("one-hot scaffold", "cc_cfl", f"{RUN}/cc_center_rescore.txt", [110, 200, 140],
         "Record system (RRR pin flow + no-clock prior). The reference."),
        ("scratch π0 (no pin)", "scr_ctr_cfl", f"{RUN}/scr_center_scores.txt", [235, 120, 80],
         "Transits the judge 10/10 but hits the gate — the classic scratch signature, third scene."),
        ("B2 coupled (λ=0.3), APC=50", "l03_ctr_cfl", f"{RUN}/b2lam03_center_scores.txt", [230, 80, 230],
         "Trained prompt, in-distribution task — approaches the gate, never crosses."),
        ("B2 coupled (λ=0.3), APC=25", "l03a25_cfl", f"{RUN}/scr_center_scores.txt", [250, 140, 220],
         "Replanning twice as often does not fix the miss — the command itself is short/offset."),
    ],
    "CFR (center from right)": [
        ("one-hot scaffold", "cc_cfr", f"{RUN}/cc_center_rescore.txt", [110, 200, 140],
         "Record system, approach from the right."),
        ("scratch π0 (no pin)", "scr_ctr_cfr", f"{RUN}/scr_center_scores.txt", [235, 120, 80],
         "Transits 10/10, clean 0/10."),
        ("B2 coupled (λ=0.3), APC=50", "l03_ctr_cfr", f"{RUN}/b2lam03_center_scores.txt", [230, 80, 230],
         "Near-miss at 0.27–0.43 m, no transit."),
        ("B2 coupled (λ=0.3), APC=25", "l03a25_cfr", f"{RUN}/scr_center_scores.txt", [250, 140, 220],
         "Same at double the replan rate: 0/10, all clean, all short."),
    ],
}
CDATA = {}
for approach, sets in CENTER_SETS.items():
    crows, cgroups = [], []
    for label, prefix, scores, colour, note in sets:
        fs, ts = trajs_of(prefix)
        tr, cl = parse_pairs(scores, f"traj_{prefix}_")
        judge = sum(tr.get(os.path.basename(f), False) for f in fs)
        clean = sum(cl.get(os.path.basename(f), 0.0) >= THRESH for f in fs)
        strict = sum(tr.get(os.path.basename(f), False) and cl.get(os.path.basename(f), 0.0) >= THRESH
                     for f in fs)
        cgroups.append({"label": f"{label} — {strict}/{len(fs)} strict", "color": colour, "trajs": ts})
        crows.append((label, note, colour, judge, clean, strict, len(fs)))
    cgroups.append({"label": "goal box (hover target)", "color": [250, 210, 80],
                    "trajs": box_edges(GOAL_C, GOAL_H)})
    cgroups.append(aperture_group("center"))
    CDATA[approach] = (cgroups, crows)

CMP_SETS = [  # (scene, label, prefix, scores, colour, note)
    ("left_and_center", "B2 coupled (λ=0.3)", "l03_cmp_left", f"{RUN}/b2lam03_center_scores.txt",
     [230, 80, 230], "Novel conjoined prompt, never in training: first gate 5/5, both gates in order 2/5, dwell 0."),
    ("left_and_center", "langprior (external)", "a50_cmp_left", f"{RUN}/apc50full_scores.txt",
     [150, 150, 160], "The enumeration-free external prior: 0/5 — no first gate."),
    # compound-right was re-flown 2026-08-12 after TWO bugs were found: (1) the rollout renderer
    # selected the LEFT splat for right_and_center (no right gate existed in the scene the drone
    # saw) — the original l03 and a50 runs are INVALID and quarantined; (2) the safety YAML's
    # gate_1 corners spanned a 0.39 m segment matching no physical opening, so even valid
    # crossings scored gates=0. With both fixed: the novel right prompt selects the right gate.
    ("left_and_center", "scratch π0 (no pin)", "scr_cmp_left", f"{RUN}/scr_center_scores.txt",
     [235, 120, 80], "First gate 1/5; every flight contacts a gate (min clearance 0.003–0.134 m)."),
    ("right_and_center", "B2 coupled (λ=0.3), re-flown", "l03_cmp_right",
     f"{RUN}/cmp_right_refly_scores.txt", [230, 80, 230],
     "Corrected scene + corrected judge: first gate 5/5, all clearance-clean (0.26–0.39 m); "
     "center copy 0/5 — the second hop fails, matching center standalone."),
    ("right_and_center", "scratch π0 (no pin)", "scr_cmp_right", f"{RUN}/scr_center_scores.txt",
     [235, 120, 80], "2/5 pass the ordered judge INCLUDING dwell — but both graze the center copy "
     "at 0.010/0.035 m, so 0/5 clean. Scratch completes dirty; the judge alone overstates it."),
]
CMP = {}
for scene, label, prefix, scores, colour, note in CMP_SETS:
    fs, ts = trajs_of(prefix)
    g, ok, cl = parse_gates(scores, f"traj_{prefix}_")
    g1 = sum(v >= 1 for v in g.values()); g2 = sum(v == 2 for v in g.values())
    clean = sum(cl.get(os.path.basename(f), 0.0) >= THRESH for f in fs)
    strict = sum(ok.get(os.path.basename(f), False) and cl.get(os.path.basename(f), 0.0) >= THRESH
                 for f in fs)
    CMP.setdefault(scene, ([], []))
    CMP[scene][0].append({"label": f"{label} — {g2}/{len(fs)} both gates", "color": colour, "trajs": ts})
    CMP[scene][1].append((label, note, colour, g1, g2, clean, strict, len(fs)))

css = """
:root{ --paper:#f6f7f9; --card:#fff; --ink:#111820; --ink2:#48545f; --line:#dde2e8; --teal:#0e7c7b;
       --pass:#2c7f4f; --warn:#a8701a; --fail:#a83a3a; }
@media (prefers-color-scheme: dark){ :root{ --paper:#0d1116; --card:#141a21; --ink:#e6ebf1;
  --ink2:#94a2b0; --line:#242e39; --teal:#4fd1c9; --pass:#5fbe86; --warn:#d7a352; --fail:#e0787a; }}
:root[data-theme="dark"]{ --paper:#0d1116; --card:#141a21; --ink:#e6ebf1; --ink2:#94a2b0;
  --line:#242e39; --teal:#4fd1c9; --pass:#5fbe86; --warn:#d7a352; --fail:#e0787a; }
:root[data-theme="light"]{ --paper:#f6f7f9; --card:#fff; --ink:#111820; --ink2:#48545f;
  --line:#dde2e8; --teal:#0e7c7b; --pass:#2c7f4f; --warn:#a8701a; --fail:#a83a3a; }
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
main{max-width:1120px;margin:0 auto;padding:44px 20px 90px;display:flex;flex-direction:column;gap:36px}
header{display:flex;flex-direction:column;gap:10px;border-bottom:1px solid var(--line);padding-bottom:22px}
.eyebrow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--teal)}
h1{margin:0;font-size:clamp(25px,4vw,36px);line-height:1.15;letter-spacing:-.02em;font-weight:650;text-wrap:balance}
h2{margin:0 0 2px;font-size:20px;font-weight:640}
p{margin:0 0 10px;max-width:74ch} .sub{margin:0;color:var(--ink2);max-width:70ch}
section{display:flex;flex-direction:column;gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.chead{padding:14px 18px;border-bottom:1px solid var(--line)}
.tbl{overflow-x:auto;padding:0 18px 14px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink2);font-weight:600}
tbody tr:last-child td{border-bottom:none}
.good{color:var(--pass)} .warn{color:var(--warn)} .bad{color:var(--fail)}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:7px;vertical-align:-1px}
.hintbar{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--teal);
  border-radius:10px;padding:13px 16px;font-size:14px;color:var(--ink2)}
.v3dwrap{padding:14px 18px 18px;display:flex;flex-direction:column;gap:10px}
.v3dwrap canvas{width:100%;display:block;border-radius:8px;background:#05080b;border:1px solid var(--line);
  cursor:grab}
.v3dwrap canvas:active{cursor:grabbing}
.v3dui{display:flex;flex-wrap:wrap;gap:13px;align-items:center;font-size:12.5px;color:var(--ink2)}
.lg{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
.ct{opacity:.75} .hint{opacity:.7} .v3dnote{font-size:12.5px;color:var(--ink2);margin:0}
footer{color:var(--ink2);font-size:12.5px;border-top:1px solid var(--line);padding-top:16px}
"""

body = ""
for i, side in enumerate(("right", "left")):
    groups, table = DATA[side]
    rows = ""
    for (label, note, judge, clean, strict, n), (_, _, _, _, colour, _, _) in zip(table, ARMS):
        cls = "good" if strict >= n * 0.7 else ("warn" if strict else "bad")
        rows += (f'<tr><td><span class="sw" style="background:rgb({colour[0]},{colour[1]},{colour[2]})"></span>'
                 f'{label}<div style="font-size:12.5px;color:var(--ink2)">{note}</div></td>'
                 f'<td class="mono">{n}</td><td class="mono">{judge}/{n}</td>'
                 f'<td class="mono">{clean}/{n}</td>'
                 f'<td class="mono {cls}">{strict}/{n}</td></tr>')
    view = cloudviewer.viewer_html(
        side, groups, height=620, elem_id=f"x{i}", max_pts=24000,
        note="Drag to orbit · wheel to zoom · shift-drag to pan. Untick an arm to isolate the others; "
             "every flight starts at the origin at 1.5 m.")
    body += f"""
<section>
 <h2>{side.capitalize()} gate</h2>
 <div class="card">
  <div class="chead"><span class="mono">all command sources overlaid · APC=50</span></div>
  <div class="tbl"><table><thead><tr><th>command source</th><th>trials</th><th>judge</th>
   <th>clearance-clean</th><th>strict</th></tr></thead><tbody>{rows}</tbody></table></div>
  {view}
 </div>
</section>"""

for ai, (approach, (cgroups, crows)) in enumerate(CDATA.items()):
    crow_html = ""
    for label, note, colour, judge, clean, strict, n in crows:
        cls = "good" if strict >= n * 0.7 else ("warn" if strict else "bad")
        crow_html += (f'<tr><td><span class="sw" style="background:rgb({colour[0]},{colour[1]},{colour[2]})"></span>'
                      f'{label}<div style="font-size:12.5px;color:var(--ink2)">{note}</div></td>'
                      f'<td class="mono">{n}</td><td class="mono">{judge}/{n}</td>'
                      f'<td class="mono">{clean}/{n}</td><td class="mono {cls}">{strict}/{n}</td></tr>')
    cview = cloudviewer.viewer_html(
        "center", cgroups, height=620, elem_id=f"xc{ai}", max_pts=24000,
        note="Drag to orbit · wheel to zoom · shift-drag to pan. The gate sits mid-scene "
             "(cyan aperture, x 2.36–3.16); the yellow box is the hover target.")
    body += f"""
<section>
 <h2>Center gate — {approach}</h2>
 <div class="card">
  <div class="chead"><span class="mono">{approach} · APC=50 · 10 trials/arm</span></div>
  <div class="tbl"><table><thead><tr><th>command source</th><th>trials</th><th>judge</th>
   <th>clearance-clean</th><th>strict</th></tr></thead><tbody>{crow_html}</tbody></table></div>
  {cview}
 </div>
</section>"""

for si, (scene, title) in enumerate([("left_and_center", "Compound: left gate → center gate"),
                                     ("right_and_center", "Compound: right gate → center gate")]):
    groups, rows = CMP[scene]
    rh = ""
    for label, note, colour, g1, g2, clean, strict, n in rows:
        cls = "good" if g2 >= n * 0.7 else ("warn" if g2 else "bad")
        scls = "good" if strict >= n * 0.7 else ("warn" if strict else "bad")
        rh += (f'<tr><td><span class="sw" style="background:rgb({colour[0]},{colour[1]},{colour[2]})"></span>'
               f'{label}<div style="font-size:12.5px;color:var(--ink2)">{note}</div></td>'
               f'<td class="mono">{n}</td><td class="mono">{g1}/{n}</td>'
               f'<td class="mono {cls}">{g2}/{n}</td><td class="mono">{clean}/{n}</td>'
               f'<td class="mono {scls}">{strict}/{n}</td></tr>')
    view = cloudviewer.viewer_html(
        scene, groups + [{"label": "goal box (hover target)", "color": [250, 210, 80],
                          "trajs": box_edges(GOAL_C, GOAL_H)}, aperture_group(scene)],
        height=620, elem_id=f"xm{si}", max_pts=24000,
        note="One static prompt, no mid-flight switching — the judge requires both gates in order plus "
             "goal dwell. 5 trials: an exploratory screen (the compound prompts were never in training), "
             "not a claim cell.")
    warn = ("" if scene == "left_and_center" else
            '<div style="padding:10px 18px 0;font-size:13px;color:var(--warn)">The original compound-right '
            'runs (this arm and the langprior reference) were INVALID — the renderer showed the drone the '
            'left scene (scene-selection bug), and the judge’s gate_1 rectangle matched no physical '
            'opening (region-box bug). Both fixed 2026-08-12; the flights below are the re-flown set.</div>')
    body += f"""
<section>
 <h2>{title}</h2>
 <div class="card">
  <div class="chead"><span class="mono">novel conjoined prompt · single instruction · APC=50 · 5-trial screen</span></div>
  {warn}
  <div class="tbl"><table><thead><tr><th>command source</th><th>trials</th><th>first gate</th>
   <th>both gates in order</th><th>clearance-clean</th><th>strict (dwell + clean)</th></tr></thead>
   <tbody>{rh}</tbody></table></div>
  {view}
 </div>
</section>"""

html = f"""<title>Trajectory explorer — every command source</title>
<style>{css}</style>
<main>
<header>
 <div class="eyebrow">drone gate navigation · trajectory explorer · 2026-08-12</div>
 <h1>Every command source, both gates, in the scene</h1>
 <p class="sub">All flights from the claim-tier tables, overlaid on the Gaussian-splat geometry the
 clearance scorer measures against. Strict success means the transit judge <em>and</em> clearance at a
 0.18&nbsp;m body radius.</p>
</header>

<div class="hintbar">
 <strong>Drag to orbit, wheel to zoom, shift-drag to pan.</strong> Each arm is a checkbox — untick the
 rest to isolate one. Two things to look for. First, the coupling split: full coupling (λ=1) and the
 external priors bend through the <em>left</em> opening and sail past the right one, while the detached
 arm does the opposite. Second, the new λ=0.3 arm (magenta): on the right gate all ten flights now go
 <em>through</em> the opening at demo-level clearance — and seven of them then glide past the yellow
 goal box instead of stopping in it. The failure moved from aim to endgame.
</div>

{body}

<footer>
 Flows and heads: <span class="mono">gate_pin_joint_b1 / _b1s / _b2</span> (command head inside the
 checkpoint), <span class="mono">gate_pin_zeropad</span> + one-hot scaffold or featfix prior,
 <span class="mono">gate_both_pin_rrr</span> + matched external prior. Basis
 <span class="mono">pin_U_gate_rrr_k5</span> throughout, APC=50, judged by
 <span class="mono">rung3/gate_success.py</span> and <span class="mono">rung3/gate_clearance.py</span>.
 B1long is 5 trials, everything else 10. The yellow wireframe is the goal box (success needs a
 post-transit frame inside it). Rollouts are not reproducible run-to-run, so treat differences
 under a few trials as noise. Point clouds are simulator geometry, used for scoring and visualisation only.
</footer>
</main>
"""
open(OUT, "w").write(html)
print(OUT, f"{os.path.getsize(OUT) / 1e6:.2f} MB")
