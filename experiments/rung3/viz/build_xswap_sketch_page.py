"""xswap re-attribution of the sketched-compound rows (2026-09-02): every sketch row in the
paper draft was flown on gmsig3/gmsig3s7; this page shows the same five sketches flown on
the flagship xswap checkpoints (both training seeds), with the original gmsig3 flights for
reference and the per-trial judge + clearance verdicts verbatim, so Denis can grade the
trajectories himself. Editorial content is deliberately minimal.

  python3 build_xswap_sketch_page.py   (writes xswap_sketches.html next to this file)
"""
import glob
import html
import json
import os
import re
import sys

import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RD = os.path.dirname(SP)
RUN = "/home/dfliu/ctxrun"
FALSIFY = os.path.expanduser("~/code/falsify-pi")
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])


def scene_marks(scene):
    safety = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{scene}.yaml"))
    gates = [np.asarray(g["corners"], np.float32) for g in safety["ordered_miss_gate"]["gates"]]
    trajs = [np.concatenate([g, g[:1]]) for g in gates]
    corners = np.array([[GOAL_C[0] + sx * GOAL_H[0], GOAL_C[1] + sy * GOAL_H[1], GOAL_C[2] + sz * GOAL_H[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [{"label": "gate apertures (judge)", "color": [124, 208, 240], "trajs": trajs},
            {"label": "goal box (judge)", "color": [248, 210, 90], "trajs": [corners[[a, b]] for a, b in idx]}]


def rollouts(pattern):
    fs = sorted(glob.glob(pattern), key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return [np.load(f)[:, :3].astype(np.float32) for f in fs]


def sketch_line(path):
    return [np.asarray(json.load(open(path))["points"], np.float32)[:, :3]]


def parse_scores(path):
    """{tag: {'judge': 'k/5', 'clean': 'k/5', 'trials': {n: [judge_line, clearance_line]}}}"""
    out, tag = {}, None
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.rstrip()
        m = re.match(r"== xswap sketch re-attribution: (\S+)", line)
        if m:
            tag = m.group(1); out[tag] = {"judge": "?", "clean": "?", "trials": {}}; continue
        if tag is None:
            continue
        m = re.match(r"== (\d+/\d+) success", line)
        if m: out[tag]["judge"] = m.group(1); continue
        m = re.match(r"== (\d+/\d+) clearance-clean", line)
        if m: out[tag]["clean"] = m.group(1); continue
        m = re.match(r"traj_\S+?_(\d+)\.npy\s+(.*)", line)
        if m:
            out[tag]["trials"].setdefault(int(m.group(1)), []).append(m.group(2).strip())
    return out


SC = parse_scores(f"{RUN}/xsk_scores.txt")


def label(name, tag):
    s = SC.get(tag)
    return f"{name} ({s['judge']} route, {s['clean']} clear)" if s else f"{name} (pending)"


def trial_table(tags):
    rows = []
    for t in tags:
        s = SC.get(t)
        if not s:
            rows.append(f"<tr><td colspan=3>{html.escape(t)}: pending</td></tr>"); continue
        for n in sorted(s["trials"]):
            j = s["trials"][n]
            rows.append(f"<tr><td>{html.escape(t)} #{n}</td><td>{html.escape(j[0] if j else '')}</td>"
                        f"<td>{html.escape(j[1] if len(j) > 1 else '')}</td></tr>")
    return ("<table class='tt'><tr><th>trial</th><th>judge</th><th>clearance (0.18 m body)</th></tr>"
            + "".join(rows) + "</table>")


# (title, scene, sketch json, [(cell tag xswap42, xswaps7)], reference gmsig3 patterns)
ROWS = [
    ("Hand-drawn sketch, Left -> Center", "left_and_center", "sketch_cmpl_denis.json",
     ("xsk42_cmpl_denis", "xsks7_cmpl_denis"),
     [("gmsig3 reference (5/5 route, 5/5 clear)", f"{RUN}/traj_skd_cmpl_*.npy")]),
    ("Hand-drawn sketch r1, Right -> Center", "right_and_center", "sketch_cmpr_denis_r1.json",
     ("xsk42_cmpr_r1", "xsks7_cmpr_r1"),
     [("gmsig3 reference (5/5 route, 3/5 clear)", f"{RUN}/traj_skdr1_cmpr_*.npy")]),
    ("4-click sketch, sigma 0, Left -> Center", "left_and_center", "sketch_cmpl_min4.json",
     ("xsk42_cmpl_min4", "xsks7_cmpl_min4"),
     [("gmsig3 reference (5/5 route, 0/5 clear)", f"{RUN}/traj_skm4_cmpl_*.npy"),
      ("gmsig3s7 reference (5/5 route, 0/5 clear)", f"{RUN}/traj_s7m4_cmpl_*.npy")]),
    ("4-click sketch, sigma 0.5, Left -> Center", "left_and_center", "sketch_cmpl_min4s.json",
     ("xsk42_cmpl_min4s", "xsks7_cmpl_min4s"),
     [("gmsig3 reference (4/5 route, 4/5 clear)", f"{RUN}/traj_skm4s_cmpl_*.npy"),
      ("gmsig3s7 reference (3/5 route, 5/5 clear)", f"{RUN}/traj_s7m4s_cmpl_*.npy")]),
    ("5-click corrected sketch, Right -> Center", "right_and_center", "sketch_cmpr_min5f.json",
     ("xsk42_cmpr_min5f", "xsks7_cmpr_min5f"),
     [("gmsig3 reference (5/5 route, 2/5 clear)", f"{RUN}/traj_m5f42_cmpr_*.npy"),
      ("gmsig3s7 reference (5/5 route, 1/5 clear)", f"{RUN}/traj_m5fs7_cmpr_*.npy")]),
]

SECS, summary = [], []
for title, scene, sk, (t42, ts7), refs in ROWS:
    groups = [{"label": "the sketch (drawn command)", "color": [255, 171, 66],
               "trajs": sketch_line(f"{RD}/{sk}")}]
    for lbl, pat in refs:
        groups.append({"label": lbl, "color": [150, 150, 158], "trajs": rollouts(pat)})
    groups.append({"label": label("xswap seed 42", t42), "color": [96, 235, 160],
                   "trajs": rollouts(f"{RUN}/traj_{t42}_*.npy")})
    groups.append({"label": label("xswap seed 7", ts7), "color": [124, 168, 255],
                   "trajs": rollouts(f"{RUN}/traj_{ts7}_*.npy")})
    groups += scene_marks(scene)
    view = cloudviewer.viewer_html(scene, groups, elem_id=f"v_{t42}", max_pts=40000,
                                   note="Grey = the original gmsig3 flights the draft reports. Green/blue = "
                                        "the same sketch on the flagship xswap checkpoints. Toggle groups.")
    SECS.append((title, view + trial_table([t42, ts7])))
    for t, seed in ((t42, "42"), (ts7, "7")):
        s = SC.get(t)
        summary.append(f"<tr><td>{html.escape(title)}</td><td>{seed}</td>"
                       f"<td>{s['judge'] if s else 'pending'}</td><td>{s['clean'] if s else 'pending'}</td></tr>")

body = "".join(f"<h2>{html.escape(t)}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>xswap Sketch Re-attribution</title>
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
table{{border-collapse:collapse;font:12px ui-monospace,Menlo,monospace;margin:10px 0;width:100%}}
td,th{{border:1px solid var(--line);padding:3px 7px;text-align:left;vertical-align:top}}
th{{color:var(--acc)}}
.tt{{overflow-x:auto;display:block}}
</style>
<main>
<h1>xswap Sketch Re-attribution</h1>
<p class="sub">The five sketched-compound rows, re-flown on the flagship xswap checkpoints (training
seeds 42 and 7), 5 trials per cell, sketch JSONs and serving settings unchanged from the original
gmsig3 rows (carrot 0, sigma per sketch, APC 50). Judge = route-clean ordered two-gate transit + dwell;
clearance = 0.18 m body radius against the scene cloud. Verdicts are printed verbatim per trial.</p>
<table><tr><th>row</th><th>xswap seed</th><th>route-clean</th><th>clearance-clean</th></tr>{''.join(summary)}</table>
{body}
</main>
"""
out = f"{SP}/xswap_sketches.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB); cells parsed: {len(SC)}")
