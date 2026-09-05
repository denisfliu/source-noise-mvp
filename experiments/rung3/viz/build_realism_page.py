"""Flight-realism page (2026-09-05): per cell, the point-cloud view of every trajectory source, speed
traces that expose the command's step structure, and the realism ledger from realism.py with the
demonstrations as the reference rows.

  /home/dfliu/miniforge3/bin/python3 build_realism_page.py     # reads ../realism_results.json
"""
import html
import json
import os
import sys

import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(SP))
import cloudviewer  # noqa: E402
import realism as R  # noqa: E402

RD = os.path.dirname(SP)
FALSIFY = os.path.expanduser("~/code/falsify-pi")
SAFETY = {"left": "left_gate", "right": "right_gate", "center": "center_gate", "left_and_center": "left_and_center"}
RES = json.load(open(f"{RD}/realism_results.json"))

# fixed colour per arm role (same hues as the decode-only page: command orange, source pink, pin green)
COL = {"dec": "#ff8c28", "src": "#f05a78", "pin": "#60eba0", "pin_s7": "#2fb874",
       "sde03": "#8fc4f5", "sde05": "#5aa9f0", "sde07": "#3a86d6", "sde09": "#2a63ad",
       "scratch": "#b48cff", "scratch_s7": "#8d66d9", "pinoff": "#9aa3b2", "real": "#e4e9f1", "synth": "#f2d25a"}


def rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def marks(scene):
    saf = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{SAFETY[scene]}.yaml"))
    gates = [np.asarray(g["corners"], np.float32) for g in saf["ordered_miss_gate"]["gates"]] if "ordered_miss_gate" in saf \
        else [np.asarray(saf["miss_gate"]["corners"], np.float32)]
    co = np.array([[R.GOAL_C[0] + sx * R.GOAL_H[0], R.GOAL_C[1] + sy * R.GOAL_H[1], R.GOAL_C[2] + sz * R.GOAL_H[2]]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [{"label": "gate apertures (judge)", "color": [124, 208, 240], "trajs": [np.concatenate([g, g[:1]]) for g in gates]},
            {"label": "goal box (judge)", "color": [248, 210, 90], "trajs": [co[[a, b]] for a, b in E]}]


# ------------------------------------------------------------------ speed traces (inline SVG small multiples)
W, H, PL, PB, PT = 300, 120, 34, 22, 8
TMAX, VMAX = 30.0, 1.0


def trace_svg(P, color, label, sub, seams=True):
    A, _ = R.split_segment(P)
    v = np.linalg.norm(np.diff(A, axis=0), axis=1) / R.DT
    t = np.arange(len(v)) * R.DT
    keep = t <= TMAX; t, v = t[keep], np.clip(v[keep], 0, VMAX)
    x = lambda s: PL + (W - PL - 6) * s / TMAX
    y = lambda s: PT + (H - PT - PB) * (1 - s / VMAX)
    pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(t, v))
    grid = "".join(f'<line x1="{PL}" x2="{W - 6}" y1="{y(g):.1f}" y2="{y(g):.1f}" class="g"/>'
                   f'<text x="{PL - 5}" y="{y(g) + 3.5:.1f}" class="tk" text-anchor="end">{g:.1f}</text>' for g in (0.0, 0.5, 1.0))
    ticks = "".join(f'<text x="{x(g):.1f}" y="{H - 7}" class="tk" text-anchor="middle">{g:.0f}</text>' for g in (0, 10, 20, 30))
    seam = "".join(f'<line x1="{x(s):.1f}" x2="{x(s):.1f}" y1="{PT}" y2="{H - PB}" class="s"/>'
                   for s in np.arange(R.CHUNK * R.DT, TMAX, R.CHUNK * R.DT)) if seams else ""
    return (f'<figure class="tr"><svg viewBox="0 0 {W} {H}" role="img" aria-label="speed trace, {html.escape(label)}">'
            f'{grid}{seam}{ticks}<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/>'
            f'<text x="{W - 6}" y="{H - 7}" class="tk" text-anchor="end">s</text></svg>'
            f'<figcaption><span class="sw" style="background:{color}"></span>{html.escape(label)}<span class="ct">{html.escape(sub)}</span></figcaption></figure>')


# ------------------------------------------------------------------ ledger table with inline bars
COLS = [  # key, header, unit, fmt, bar scale (None = auto max), lower-is-better
    ("speed_p95", "speed p95", "m/s", "{:.2f}", 1.0),
    ("acc_p95", "accel p95 (SG)", "m/s²", "{:.2f}", 1.5),
    ("jerk_p95", "jerk p95 (SG)", "m/s³", "{:.1f}", 15.0),
    ("dimless_jerk", "dimless jerk (log10)", "", "{:.2f}", 9.0),
    ("hf_frac", "velocity power ≥1 Hz", "", "{:.2f}", 0.4),
    ("zero_acc_frac", "zero-accel steps", "", "{:.2f}", 1.0),
    ("seam_acc", "replan-seam |Δv|/dt", "m/s²", "{:.2f}", 5.0),
    ("tilt_p99", "required tilt p99", "°", "{:.1f}", 12.0),
    ("rate_p99", "required body rate p99", "°/s", "{:.0f}", 100.0),
    ("env_jerk_viol", "steps beyond real jerk p99", "", "{:.3f}", 0.25),
    ("track_rmse", "PX4 tracking RMSE", "m", "{:.3f}", 0.12),
]
AUCS = [("auc_vs_real", "AUC vs real"), ("auc_vs_real_shape", "AUC vs real (shape)"),
        ("auc_vs_synth", "AUC vs synth"), ("auc_vs_synth_shape", "AUC vs synth (shape)")]


def cell_html(key, label, s, color, ref=False):
    g = lambda k: s.get(k, {}).get("med")
    tds = []
    for k, _, _, fmt, scale in COLS:
        v = g(k)
        if v is None or not np.isfinite(v):
            tds.append("<td class='na'>–</td>"); continue
        w = min(100, 100 * v / scale)
        tds.append(f"<td><span class='bar' style='width:{w:.0f}%'></span><span class='num'>{fmt.format(v)}</span></td>")
    for k, _ in AUCS:
        v = s.get(k)
        if v is None:
            tds.append("<td class='na'>–</td>"); continue
        w = max(0, min(100, 200 * (v - 0.5)))
        tds.append(f"<td><span class='bar' style='width:{w:.0f}%'></span><span class='num'>{v:.2f}</span></td>")
    cls = " class='ref'" if ref else ""
    return (f"<tr{cls}><td class='lab'><span class='sw' style='background:{color}'></span>{html.escape(label)}"
            f"<span class='ct'>n={s['n']}</span></td>" + "".join(tds) + "</tr>")


def ledger(cell):
    c = RES["cells"][cell]
    head = "<tr><th>trajectory source</th>" + "".join(f"<th>{h}<span class='u'>{u}</span></th>" for _, h, u, _, _ in COLS) \
        + "".join(f"<th>{h}</th>" for _, h in AUCS) + "</tr>"
    rows = [cell_html("real", "real demos (mocap, 100 flights)", RES["demos"]["real"], COL["real"], ref=True)]
    task = c["task"]
    synth = RES["demos"].get(f"synth_task{task}", RES["demos"]["synth"]) if task is not None else RES["demos"]["synth"]
    rows.append(cell_html("synth", "synth demos (planner" + (f", this task" if task is not None else "") + ")", synth, COL["synth"], ref=True))
    for key, s in c["arms"].items():
        rows.append(cell_html(key, s["label"], s, COL[key]))
    return f"<div class='tw'><table class='rt'>{head}{''.join(rows)}</table></div>"


# ------------------------------------------------------------------ page
def tally(key, s):
    return ""


SECS = []
for cell, (title, scene, task, arms) in R.CELLS.items():
    c = RES["cells"].get(cell)
    if not c:
        continue
    groups, traces = [], []
    for key, label, tag in arms:
        if key not in c["arms"]:
            continue
        trajs = R.load_arm(tag)
        groups.append({"label": f"{label} (n={len(trajs)})", "color": rgb(COL[key]), "trajs": [t.astype(np.float32) for t in trajs]})
        s = c["arms"][key]
        traces.append(trace_svg(trajs[0], COL[key], label, f"zero-accel {s['zero_acc_frac']['med']:.2f} · jerk p95 {s['jerk_p95']['med']:.1f}"))
    if cell == "cmpl":
        groups.append({"label": "the sketch (drawn command)", "color": [255, 200, 90],
                       "trajs": [np.asarray(json.load(open(f"{RD}/sketch_cmpl_denis.json"))["points"], np.float32)[:, :3]]})
    groups += marks(scene)
    demos = R.load_demos()
    dtr = []
    if task is not None:
        for name, src in (("real", "real"), ("synth", "synth")):
            ps = demos[src].get(task, [])
            if ps:
                dtr.append(trace_svg(ps[0], COL[name], f"{name} demo, this task", "reference", seams=False))
    if not dtr:
        dtr = [trace_svg(demos["real"][0][0], COL["real"], "real demo (left gate)", "reference", seams=False),
               trace_svg(demos["synth"][2][0], COL["synth"], "synth demo (CFL)", "reference", seams=False)]
    note = ("Every source is a 10 Hz position stream in the same scene; the ledger below scores the active segment "
            "(start to first goal-box entry + 1 s) of every flight and reports the median over flights.")
    SECS.append((title, cloudviewer.viewer_html(scene, groups, elem_id=f"v_{cell}", max_pts=40000, note=note)
                 + "<h3>Speed traces, first flight of each source</h3><p class='sub'>Speed from raw position differences, "
                 "one flight per source; faint verticals are the 50-step replan seams (5 s). The command alone is a staircase; "
                 "the demonstrations are bells.</p><div class='trs'>" + "".join(dtr + traces) + "</div>"
                 + "<h3>Realism ledger</h3>" + ledger(cell)))

d = RES["demos"]; env = RES["envelope"]
cal = (f"real-vs-real {d['auc_real_vs_real']:.2f} (shape {d['auc_real_vs_real_shape']:.2f}) · "
       f"synth-vs-real {d['auc_synth_vs_real']:.2f} (shape {d['auc_synth_vs_real_shape']:.2f})")
body = "".join(f"<h2>{t}</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Flight Realism Ledger</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0;--bar:#2b3a4a;--ref:#1a222c;
  --sans:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans);padding:28px 18px 70px}}
main{{max-width:1120px;margin:0 auto}} h1{{font-size:24px;font-weight:600;margin:0 0 4px;text-wrap:balance}}
h2{{font-size:17px;font-weight:600;margin:34px 0 8px;color:var(--acc)}} h3{{font-size:14px;font-weight:600;margin:22px 0 4px;letter-spacing:.02em;text-transform:uppercase;color:var(--mut)}}
.sub{{color:var(--mut);margin:0 0 14px;max-width:92ch}} .kv{{font:13px var(--mono);color:var(--mut);margin:0 0 6px}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:12px 14px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px var(--mono)}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:6px;flex:none}}
.ct{{color:var(--mut);margin-left:8px;font:12px var(--mono)}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
.trs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px 16px}}
.tr{{margin:0}} .tr svg{{width:100%;height:auto;display:block;background:var(--bg);border-radius:6px}}
.tr figcaption{{font:12px var(--mono);color:var(--ink);margin-top:5px;display:flex;align-items:center;flex-wrap:wrap}}
.g{{stroke:var(--line);stroke-width:1}} .s{{stroke:var(--line);stroke-width:1;stroke-dasharray:2 3}} .tk{{fill:var(--mut);font:10px var(--mono)}}
.tw{{overflow-x:auto}} table.rt{{border-collapse:collapse;font:12px var(--mono);margin:8px 0 0;font-variant-numeric:tabular-nums;min-width:100%}}
.rt td,.rt th{{border-bottom:1px solid var(--line);padding:5px 9px;text-align:left;white-space:nowrap;vertical-align:middle}}
.rt th{{color:var(--mut);font-weight:500;font-size:11px;vertical-align:bottom;max-width:10ch;white-space:normal;overflow-wrap:anywhere;line-height:1.25}}
.rt th .u{{display:block;color:var(--mut);opacity:.7}} .rt td.lab{{min-width:26ch;white-space:normal}}
.rt td{{position:relative}} .bar{{position:absolute;left:4px;top:50%;height:8px;margin-top:-4px;background:var(--bar);border-radius:2px;max-width:calc(100% - 8px)}}
.num{{position:relative}} .na{{color:var(--mut)}} tr.ref td{{background:var(--ref)}}
</style>
<main><h1>Flight Realism Ledger</h1>
<p class="sub">Is a flight something the drone actually flies like? Each cell compares the coarse command executed by itself
(no denoising), the pinned source sample executed by itself, the pin arm (the flow denoises the command), SDEdit on the unpinned
flow guided by the same command, the unpinned flow alone, and the demonstrations that define realistic. Metrics: <code>experiments/rung3/realism.py</code>.</p>
<p class="kv">reference: real demos' smoothed p99 envelope |a| {env['acc_p99']:.2f} m/s² · |j| {env['jerk_p99']:.1f} m/s³ · 0.7 s Savitzky-Golay (SG) derivatives · PX4 defaults acc 3 m/s², jerk 8 m/s³, tilt 45°</p>
<p class="kv">classifier calibration (1 s windows, grouped 5-fold CV): {cal}. 0.5 = indistinguishable; shape = magnitudes divided by window mean speed.</p>
{body}</main>"""
open(f"{SP}/realism.html", "w").write(page)
print(f"wrote realism.html ({len(page)/1e6:.1f} MB)")
