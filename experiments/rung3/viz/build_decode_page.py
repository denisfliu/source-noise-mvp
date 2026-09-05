"""Decode-only diagnostic page (2026-09-04): what the 16 command words encode on their own. Each
scene shows the flights that EXECUTE U c directly (no denoising) beside the pin arm's flights
(same head, flow reads the command) and, for the sketch cell, the drawn line.

  /home/dfliu/code/tv/bin/python build_decode_page.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
import cloudviewer  # noqa: E402
from catalogue import AUTO  # noqa: E402

RD = os.path.dirname(SP); RUN = "/home/dfliu/ctxrun"
FALSIFY = os.path.expanduser("~/code/falsify-pi")
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
SAFETY = {"left": "left_gate", "center": "center_gate", "left_and_center": "left_and_center"}


def marks(scene):
    saf = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{SAFETY[scene]}.yaml"))
    gates = [np.asarray(g["corners"], np.float32) for g in saf["ordered_miss_gate"]["gates"]] if "ordered_miss_gate" in saf \
        else [np.asarray(saf["miss_gate"]["corners"], np.float32)]
    co = np.array([[GOAL_C[0] + sx * GOAL_H[0], GOAL_C[1] + sy * GOAL_H[1], GOAL_C[2] + sz * GOAL_H[2]]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [{"label": "gate apertures (judge)", "color": [124, 208, 240], "trajs": [np.concatenate([g, g[:1]]) for g in gates]},
            {"label": "goal box (judge)", "color": [248, 210, 90], "trajs": [co[[a, b]] for a, b in E]}]


def rollouts(pat):
    fs = sorted(glob.glob(pat), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    return [np.load(f)[:, :3].astype(np.float32) for f in fs]


def realize(prefix):
    """median over flights of (speed p95 m/s, accel p95 m/s^2, path length m). gate_nav3 is 10 Hz (meta/info.json
    fps=10; one action per frame) -- the 2026-09-04 page and log entry scaled these as 25 Hz. realism.py is the full suite."""
    rows = []
    for f in glob.glob(f"{RUN}/traj_{prefix}_*.npy"):
        P = np.load(f)[:, :3]; v = np.diff(P, axis=0); sp = np.linalg.norm(v, axis=1); acc = np.linalg.norm(np.diff(v, axis=0), axis=1)
        rows.append((np.percentile(sp, 95) * 10, np.percentile(acc, 95) * 100, sp.sum()))
    if not rows:
        return "pending"
    m = np.median(np.array(rows), axis=0)
    return f"speed p95 {m[0]:.2f} m/s · accel p95 {m[1]:.0f} m/s² · path {m[2]:.1f} m"


def tally(prefix):
    r = c = n = m = 0
    for f in glob.glob(f"{RUN}/traj_{prefix}_*.npy"):
        au = AUTO.get(os.path.basename(f)[:-4], {})
        if au.get("judge"): n += 1; r += ("SUCCESS=True" in au["judge"]) and ("wrong_dir=0" in au["judge"] or "wrong_dir" not in au["judge"])
        if au.get("clear"): m += 1; c += "CLEAN=True" in au["clear"]
    return f"{r}/{n} route, {c}/{m} clean" if n else "pending"


SECS = []
SRC = {"dec_cfr": "src_cfr", "dec_cmpl": "src_cmpl"}
for title, scene, dec, pin, sketch, note in [
    ("Center from right", "center", "dec_cfr", "xswap_cfr", None,
     "Orange: the head's command U c executed verbatim every replan, no flow. Green: the same head's command read by the pin-trained flow. "
     "The difference between the two is everything the flow contributes."),
    ("Left gate", "left", "dec_left", "armxswap_left", None, "As above, left gate."),
    ("Compound L->C, hand-drawn sketch", "left_and_center", "dec_cmpl", "xsk42_cmpl_denis", f"{RD}/sketch_cmpl_denis.json",
     "Orange: the sketch's own command U c executed verbatim (the coarse, 4-band approximation of the drawn line). "
     "Green: the pin flights of the same sketch. Yellow-orange line: the drawing."),
]:
    g = [{"label": f"decoded command U c executed, no flow, deterministic ({tally(dec)})", "color": [255, 140, 40], "trajs": rollouts(f"{RUN}/traj_{dec}_*.npy")}]
    if dec in SRC:
        g.append({"label": f"pinned SOURCE SAMPLE z executed, no flow, 5 noise draws ({tally(SRC[dec])})", "color": [240, 90, 120],
                  "trajs": rollouts(f"{RUN}/traj_{SRC[dec]}_*.npy")})
    g.append({"label": f"pin arm: flow reads the same command ({tally(pin)})", "color": [96, 235, 160], "trajs": rollouts(f"{RUN}/traj_{pin}_*.npy")})
    if sketch:
        g.append({"label": "the sketch (drawn command)", "color": [255, 200, 90],
                  "trajs": [np.asarray(json.load(open(sketch))["points"], np.float32)[:, :3]]})
    g += marks(scene)
    rows = [("decoded U c, no flow", dec), ("pinned source sample z, no flow", SRC.get(dec)), ("pin arm (flow denoises z)", pin)]
    tab = "<table class='rt'><tr><th>trajectory source</th><th>route / clean</th><th>realizability (median over flights)</th></tr>" + "".join(
        f"<tr><td>{n}</td><td>{tally(k)}</td><td>{realize(k)}</td></tr>" for n, k in rows if k) + "</table>"
    SECS.append((title, cloudviewer.viewer_html(scene, g, elem_id=f"v_{dec}", max_pts=40000, note=note) + tab))

body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Command Without Denoising</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:23px;margin:0 0 4px}} h2{{font-size:16px;margin:30px 0 8px;color:var(--acc)}}
.sub{{color:var(--mut);margin:0 0 18px;max-width:92ch}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
table.rt{{border-collapse:collapse;font:12px ui-monospace,Menlo,monospace;margin:10px 0 0;font-variant-numeric:tabular-nums}}
.rt td,.rt th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}} .rt th{{color:var(--acc);font-weight:500}}
</style>
<main><h1>Command Without Denoising</h1>
<p class="sub">Three ways to turn the same command into motion. Orange: the decoded minimum-norm chunk U c executed directly (deterministic,
piecewise-constant velocity per horizon band). Pink: the pinned source sample itself, z = g - UU^T g + U c, executed directly (exact command,
unit-variance Gaussian jitter in the orthogonal complement, a fresh draw each replan). Green: the pin arm, where the flow denoises z.
U c is the minimum-norm 50-step chunk consistent with the 16 words: piecewise-constant velocity over the four horizon bands per axis.
Replanned every 50 steps from the head (atomics) or the sketch window (compound), so the orange paths are the chained coarse plan.</p>
{body}</main>"""
open(f"{SP}/decode_only.html", "w").write(page)
print(f"wrote decode_only.html ({len(page)/1e6:.1f} MB)")
