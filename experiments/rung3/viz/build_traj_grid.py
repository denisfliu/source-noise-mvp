"""Compact synced trajectory grid (Denis, 2026-08-12): three arms only (one-hot scaffold,
B2 coupled lam=0.3, scratch), six scenes at once, ONE global legend that toggles every panel
simultaneously, middle-drag pan. Replaces the long per-scene explorer at the same artifact URL."""
import glob
import os
import re
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import gridviewer

RUN = "/home/ubuntu/ctxrun"
OUT = f"{SP}/trajectory_explorer.html"
THRESH = 0.18

ARMS = [
    {"key": "onehot", "label": "one-hot scaffold", "color": [110, 200, 140]},
    {"key": "lam03", "label": "B2 coupled (λ=0.3)", "color": [230, 80, 230]},
    {"key": "b2long", "label": "B2 coupled (λ=1, 15k annealed)", "color": [90, 170, 245]},
    {"key": "c2", "label": "C2 factored (c-loss-only VLM)", "color": [70, 190, 160]},
    {"key": "mh16", "label": "mh16 multi-horizon basis", "color": [245, 200, 60]},
    {"key": "logmag", "label": "b2logmag (scale-inv. loss)", "color": [255, 140, 100]},
    {"key": "gen1", "label": "gen1 (generative head, flat U)", "color": [160, 230, 90]},
    {"key": "gen16", "label": "gen16 (generative × multi-horizon)", "color": [255, 210, 130]},
    {"key": "scratch", "label": "scratch π0 (no pin)", "color": [235, 120, 80]},
    {"key": "goalbox", "label": "goal box", "color": [250, 210, 80]},
    {"key": "aperture", "label": "gate aperture (judge)", "color": [90, 220, 255]},
]

GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
APERTURE = {
    "left": [[0.65, 1.05, 0.20], [1.18, 0.45, 0.20], [1.18, 0.45, 1.95], [0.65, 1.05, 1.95]],
    "right": [[0.195, -1.348, 0.20], [0.924, -0.952, 0.20], [0.924, -0.952, 1.95], [0.195, -1.348, 1.95]],
    "center": [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875], [3.156, -0.328, 1.875]],
}
SCENE_AP = {"left": ["left"], "right": ["right"], "center": ["center"],
            "left_and_center": ["left", "center"], "right_and_center": ["right", "center"]}


def box_edges(c, h):
    k = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return [k[[a, b]] for a, b in idx]


def quads(scene):
    return [np.array(APERTURE[k] + [APERTURE[k][0]], np.float32) for k in SCENE_AP[scene]]


def trajs(prefix):
    fs = sorted(glob.glob(f"{RUN}/traj_{prefix}_*.npy"),
                key=lambda p: int(p.split("_")[-1].split(".")[0].lstrip("t")))
    return fs, [np.load(f)[:, :3] for f in fs]


def pairs(path, tag):
    """traj -> (judge SUCCESS, transit, min clearance) from gate_success + gate_clearance output."""
    s = open(path).read()
    m_ = re.findall(rf"({tag}[\w.]*?\.npy)\s+transit=(\S+) wrong_dir=\d+ goal=\w+\s+SUCCESS=(\w+)", s)
    tr = {m[0]: m[2] == "True" for m in m_}
    tx = {m[0]: m[1].startswith("True") for m in m_}
    cl = {m[0]: float(m[1]) for m in re.findall(rf"({tag}[\w.]*?\.npy)\s+min-clearance ([0-9.]+) m", s)}
    return tr, tx, cl


def gates(path, tag):
    s = open(path).read()
    g = {m[0]: int(m[1]) for m in re.findall(rf"({tag}[\w.]*?\.npy)\s+gates=(\d)/2", s)}
    ok = {m[0]: m[1] == "True" for m in re.findall(rf"({tag}[\w.]*?\.npy)\s+gates=.*SUCCESS=(\w+)", s)}
    c = open(f"{RUN}/cmp_clearance.txt").read()
    cl = {m[0]: float(m[1]) for m in re.findall(rf"({tag}[\w.]*?\.npy)\s+min-clearance ([0-9.]+) m", c)}
    return g, ok, cl


def strict_single(prefix, scores):
    fs, ts = trajs(prefix)
    tr, tx, cl = pairs(scores, f"traj_{prefix}_")
    s = sum(tr.get(os.path.basename(f), False) and cl.get(os.path.basename(f), 0.0) >= THRESH
            for f in fs)
    j = sum(tr.get(os.path.basename(f), False) for f in fs)
    x = sum(tx.get(os.path.basename(f), False) for f in fs)
    return ts, j, s, x, len(fs)


def strict_compound(prefix, scores):
    fs, ts = trajs(prefix)
    g, ok, cl = gates(scores, f"traj_{prefix}_")
    g2 = sum(v == 2 for v in g.values())
    s = sum(ok.get(os.path.basename(f), False) and cl.get(os.path.basename(f), 0.0) >= THRESH
            for f in fs)
    return ts, g2, s, len(fs)


# (panel id, title, scene, {armkey: (kind, prefix, scores)})
PANELS = [
    ("g_left", "Left gate", "left", {
        "onehot": ("s", "swap10_left", f"{RUN}/swap10_scores.txt"),
        "lam03": ("s", "armb2lam03_left", f"{RUN}/arm_b2lam03_scores.txt"),
        "b2long": ("s", "armb2long_left", f"{RUN}/arm_b2long_scores.txt"),
        "c2": ("s", "armc2_left", f"{RUN}/arm_c2_scores.txt"),
        "mh16": ("s", "armmh16_left", f"{RUN}/arm_mh16_scores.txt"),
        "logmag": ("s", "armb2logmag_left", f"{RUN}/arm_b2logmag_scores.txt"),
        "gen1": ("s", "armgen1_left", f"{RUN}/arm_gen1_scores.txt"),
        "gen16": ("s", "armgen16_left", f"{RUN}/arm_gen16_scores.txt"),
        "scratch": ("s", "scr_left", f"{RUN}/scr_lr_scores.txt")}),
    ("g_right", "Right gate", "right", {
        "onehot": ("s", "swap10_right", f"{RUN}/swap10_scores.txt"),
        "lam03": ("s", "armb2lam03_right", f"{RUN}/arm_b2lam03_scores.txt"),
        "b2long": ("s", "armb2long_right", f"{RUN}/arm_b2long_scores.txt"),
        "c2": ("s", "armc2_right", f"{RUN}/arm_c2_scores.txt"),
        "mh16": ("s", "armmh16_right", f"{RUN}/arm_mh16_scores.txt"),
        "logmag": ("s", "armb2logmag_right", f"{RUN}/arm_b2logmag_scores.txt"),
        "gen1": ("s", "armgen1_right", f"{RUN}/arm_gen1_scores.txt"),
        "gen16": ("s", "armgen16_right", f"{RUN}/arm_gen16_scores.txt"),
        "scratch": ("s", "scr_right", f"{RUN}/scr_lr_scores.txt")}),
    ("g_cfl", "Center from left", "center", {
        "onehot": ("s", "cc_cfl", f"{RUN}/cc_center_rescore.txt"),
        "lam03": ("s", "l03_ctr_cfl", f"{RUN}/b2lam03_center_scores.txt"),
        "c2": ("s", "c2_cfl", f"{RUN}/ctr_c2_scores.txt"),
        "mh16": ("s", "mh16_cfl", f"{RUN}/ctr_mh16_scores.txt"),
        "gen1": ("s", "gen1_cfl", f"{RUN}/ctr_gen1_scores.txt"),
        "gen16": ("s", "gen16_cfl", f"{RUN}/ctr_gen16_scores.txt"),
        "scratch": ("s", "scr_ctr_cfl", f"{RUN}/scr_center_scores.txt")}),
    ("g_cfr", "Center from right", "center", {
        "onehot": ("s", "cc_cfr", f"{RUN}/cc_center_rescore.txt"),
        "lam03": ("s", "l03_ctr_cfr", f"{RUN}/b2lam03_center_scores.txt"),
        "c2": ("s", "c2_cfr", f"{RUN}/ctr_c2_scores.txt"),
        "mh16": ("s", "mh16_cfr", f"{RUN}/ctr_mh16_scores.txt"),
        "gen1": ("s", "gen1_cfr", f"{RUN}/ctr_gen1_scores.txt"),
        "gen16": ("s", "gen16_cfr", f"{RUN}/ctr_gen16_scores.txt"),
        "scratch": ("s", "scr_ctr_cfr", f"{RUN}/scr_center_scores.txt")}),
    ("g_cmpl", "Compound: left → center", "left_and_center", {
        "lam03": ("c", "l03_cmp_left", f"{RUN}/b2lam03_center_scores.txt"),
        "c2": ("c", "c2_cmpl", f"{RUN}/ctr_c2_scores.txt"),
        "mh16": ("c", "mh16_cmpl", f"{RUN}/ctr_mh16_scores.txt"),
        "gen1": ("c", "gen1_cmpl", f"{RUN}/ctr_gen1_scores.txt"),
        "gen16": ("c", "gen16_cmpl", f"{RUN}/ctr_gen16_scores.txt"),
        "scratch": ("c", "scr_cmp_left", f"{RUN}/scr_center_scores.txt")}),
    ("g_cmpr", "Compound: right → center", "right_and_center", {
        "lam03": ("c", "l03_cmp_right", f"{RUN}/cmp_right_refly_scores.txt"),
        "c2": ("c", "c2_cmpr", f"{RUN}/ctr_c2_scores.txt"),
        "mh16": ("c", "mh16_cmpr", f"{RUN}/ctr_mh16_scores.txt"),
        "gen1": ("c", "gen1_cmpr", f"{RUN}/ctr_gen1_scores.txt"),
        "gen16": ("c", "gen16_cmpr", f"{RUN}/ctr_gen16_scores.txt"),
        "scratch": ("c", "scr_cmp_right", f"{RUN}/scr_center_scores.txt")}),
]

panels, table = [], {}
for pid, title, scene, spec in PANELS:
    groups = {"goalbox": box_edges(GOAL_C, GOAL_H), "aperture": quads(scene)}
    for key, (kind, prefix, scores) in spec.items():
        if kind == "s":
            ts, j, s, x, n = strict_single(prefix, scores)
            extra = f" ({x} transit)" if x > max(j, s) else (f" ({j} judge)" if j != s else "")
            table.setdefault(key, {})[pid] = f"{s}/{n}{extra}"
        else:
            ts, g2, s, n = strict_compound(prefix, scores)
            table.setdefault(key, {})[pid] = f"{s}/{n} ({g2} both gates)"
        groups[key] = ts
    panels.append({"scene": scene, "title": title, "id": pid, "groups": groups})

legend, grid, js = gridviewer.grid_html(panels, ARMS)

cols = [(pid, title) for pid, title, _, _ in PANELS]
rows = ""
for a in [x for x in ARMS if x["key"] not in ("goalbox", "aperture")]:
    cells = "".join(f'<td class="mono">{table.get(a["key"], {}).get(pid, "—")}</td>'
                    for pid, _ in cols)
    rows += (f'<tr><td><span class="sw" style="background:rgb({a["color"][0]},{a["color"][1]},'
             f'{a["color"][2]})"></span>{a["label"]}</td>{cells}</tr>')
head = "".join(f"<th>{t}</th>" for _, t in cols)

html = f"""<title>Trajectory grid — three command sources, six scenes</title>
<style>
:root{{ --paper:#f6f7f9; --card:#fff; --ink:#111820; --ink2:#48545f; --line:#dde2e8; --teal:#0e7c7b; }}
@media (prefers-color-scheme: dark){{ :root{{ --paper:#0d1116; --card:#141a21; --ink:#e6ebf1;
  --ink2:#94a2b0; --line:#242e39; --teal:#4fd1c9; }} }}
:root[data-theme="dark"]{{ --paper:#0d1116; --card:#141a21; --ink:#e6ebf1; --ink2:#94a2b0;
  --line:#242e39; --teal:#4fd1c9; }}
:root[data-theme="light"]{{ --paper:#f6f7f9; --card:#fff; --ink:#111820; --ink2:#48545f;
  --line:#dde2e8; --teal:#0e7c7b; }}
*{{box-sizing:border-box}} .mono{{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}}
body{{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:28px 18px 70px}}
h1{{font-size:clamp(20px,3vw,27px);margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--ink2);margin:0 0 16px;max-width:90ch}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:16px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:right;padding:5px 9px;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
thead th{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2)}}
tbody tr:last-child td{{border-bottom:none}}
.sw{{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:7px;vertical-align:-1px}}
.tbl{{overflow-x:auto}}
{gridviewer.STYLE}
</style>
<main>
<h1>Three command sources · six scenes</h1>
<p class="sub">Drag to orbit · wheel to zoom · <strong>middle-drag (or shift-drag) to pan</strong>.
The legend below controls every panel at once. Strict = transit judge + 0.18&nbsp;m clearance
(+ dwell on compounds); yellow box = hover target, cyan rectangle = the judge's aperture.
Compound cells are 5-trial screens; everything else is 10 trials.</p>
<div class="glegend">{legend}</div>
<div class="ggrid">{grid}</div>
<div class="card tbl">
 <table><thead><tr><th>command source (strict)</th>{head}</tr></thead><tbody>{rows}</tbody></table>
</div>
{js}
</main>
"""
open(OUT, "w").write(html)
print(OUT, f"{os.path.getsize(OUT) / 1e6:.2f} MB")
