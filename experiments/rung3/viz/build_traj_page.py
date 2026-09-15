"""Generic trajectory page: any set of ctxrun trajectory tags over a scene cloud, each flight labelled
with its route-clean judge and clearance result from the score files, optionally with a sketch.

  python build_traj_page.py --scene left_and_center --title "..." --out min4v2.html \
      --group "xswap s42=xsk42_cmpl_min4v2" --group "xswap s7=xsks7_cmpl_min4v2" \
      --sketch ../sketch_cmpl_min4v2.json --note "..."
Flights that are route-clean AND clearance-clean are drawn in the group colour; grazes (route-clean
but clearance < 0.18 m) in orange; route failures in red. The legend lists per-flight min clearance.
"""
import argparse
import html
import json
import os
import re
import sys

import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(SP))
import cloudviewer  # noqa: E402
from catalogue import AUTO, trial_files  # noqa: E402

RD = os.path.dirname(SP)
FALSIFY = os.path.expanduser("~/code/falsify-pi")
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
SAFETY = {"left": "left_gate", "right": "right_gate", "center": "center_gate",
          "left_and_center": "left_and_center", "right_and_center": "right_and_center"}
GROUP_COLS = [[96, 235, 160], [90, 170, 240], [180, 140, 255], [240, 210, 90]]
GRAZE, FAIL = [255, 140, 40], [240, 80, 80]


def axes(length=0.5):
    """Mocap-frame origin with x (red), y (green), z (blue) axes of `length` metres, as three viewer groups."""
    o = np.zeros(3, np.float32)
    return [{"label": "origin: +x axis (0.5 m)", "color": [240, 80, 80], "trajs": [np.stack([o, o + np.array([length, 0, 0], np.float32)])]},
            {"label": "origin: +y axis (0.5 m)", "color": [80, 220, 80], "trajs": [np.stack([o, o + np.array([0, length, 0], np.float32)])]},
            {"label": "origin: +z axis (0.5 m)", "color": [90, 140, 255], "trajs": [np.stack([o, o + np.array([0, 0, length], np.float32)])]}]


def marks(scene):
    saf = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{SAFETY[scene]}.yaml"))
    gates = [np.asarray(g["corners"], np.float32) for g in saf["ordered_miss_gate"]["gates"]] if "ordered_miss_gate" in saf \
        else [np.asarray(saf["miss_gate"]["corners"], np.float32)]
    co = np.array([[GOAL_C[0] + sx * GOAL_H[0], GOAL_C[1] + sy * GOAL_H[1], GOAL_C[2] + sz * GOAL_H[2]]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [{"label": "gate apertures (judge)", "color": [124, 208, 240], "trajs": [np.concatenate([g, g[:1]]) for g in gates]},
            {"label": "goal box (judge)", "color": [248, 210, 90], "trajs": [co[[a, b]] for a, b in E]}]


def verdict(fname):
    a = AUTO.get(fname, {}); j, k = a.get("judge", ""), a.get("clear", "")
    route = ("SUCCESS=True" in j) and ("wrong_dir=0" in j or "wrong_dir" not in j)
    m = re.search(r"min-clearance ([0-9.]+)", k); clr = float(m.group(1)) if m else float("nan")
    return route, ("CLEAN=True" in k), clr


def section(scene, specs, sketch, elem, judge=True, extra=()):
    """One viewer + table for a list of 'label=tag' specs. judge=False colours by clearance only."""
    groups, rows = list(extra), []
    for gi, spec in enumerate(specs):
        label, tag = spec.split("=", 1)
        ok, gz, bad = [], [], []
        for f in trial_files(tag):
            P = np.load(f)[:, :3].astype(np.float32); name = os.path.basename(f)[:-4]
            route, clean, clr = verdict(name)
            if not judge:
                route = True
            rows.append((label, name.rsplit("_", 1)[1], route, clean, clr))
            (ok if route and clean else gz if route else bad).append(P)
        col = GROUP_COLS[gi % len(GROUP_COLS)]
        okl = "route-clean + clearance-clean" if judge else "clearance-clean"
        gzl = "route-clean, graze < 0.18 m" if judge else "contact < 0.18 m"
        if ok: groups.append({"label": f"{label}: {okl} ({len(ok)})", "color": col, "trajs": ok})
        if gz: groups.append({"label": f"{label}: {gzl} ({len(gz)})", "color": GRAZE if judge else col, "trajs": gz})
        if bad: groups.append({"label": f"{label}: route failure ({len(bad)})", "color": FAIL, "trajs": bad})
    if sketch:
        groups.append({"label": "the sketch (drawn command)", "color": [255, 200, 90],
                       "trajs": [np.asarray(json.load(open(sketch))["points"], np.float32)[:, :3]]})
    groups += marks(scene) + axes()
    rc = "<th>route-clean</th>" if judge else ""
    tab = f"<table class='rt'><tr><th>arm</th><th>trial</th>{rc}<th>clearance-clean</th><th>min clearance</th></tr>" + "".join(
        f"<tr class='{'ok' if r and c else 'gz' if r else 'bad'}'><td>{html.escape(l)}</td><td>{t}</td>"
        + (f"<td>{'yes' if r else 'no'}</td>" if judge else "")
        + f"<td>{'yes' if c else 'no'}</td><td>{clr:.3f} m</td></tr>" for l, t, r, c, clr in rows) + "</table>"
    n_ok = sum(1 for *_, r, c, _ in rows if r and c); n_r = sum(1 for *_, r, _, _ in rows if r)
    note = ("Orange flights are route-clean but pass within 0.18 m of a gate post; red are route failures." if judge
            else "Colour by arm; the table gives each flight's closest approach to a gate post. Toggle groups in the legend.")
    head = (f"{n_r}/{len(rows)} route-clean, " if judge else "") + f"{n_ok}/{len(rows)} clearance-clean (0.18 m body radius)."
    return f"<p class='kv'>{head}</p>" + cloudviewer.viewer_html(scene, groups, elem_id=elem, max_pts=40000, note=note) + tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene"); ap.add_argument("--title", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--group", action="append", default=[], help="label=tag (single-section page)")
    ap.add_argument("--section", action="append", default=[], help="title|scene|sketch-or-empty|label=tag,label=tag")
    ap.add_argument("--sketch"); ap.add_argument("--note", default=""); ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--demo-task", type=int, help="also draw the synth demonstrations of this gate_nav3 task index (faint)")
    ap.add_argument("--demo-n", type=int, default=10)
    a = ap.parse_args()
    secs = []
    if a.group:
        extra = []
        if a.demo_task is not None:
            import realism as R
            ps = R.load_demos()["synth"].get(a.demo_task, [])[:a.demo_n]
            extra.append({"label": f"synth demonstrations, task {a.demo_task} ({len(ps)} of 50)", "color": [150, 150, 160],
                          "trajs": [p.astype(np.float32) for p in ps]})
        secs.append(("", section(a.scene, a.group, a.sketch, "v0", not a.no_judge, extra)))
    for i, spec in enumerate(a.section):
        t, sc, sk, gs = spec.split("|")
        secs.append((t, section(sc, gs.split(","), sk or None, f"v{i + 1}", not a.no_judge)))
    body = "".join((f"<h2>{html.escape(t)}</h2>" if t else "") + f"<div class='vc'>{h}</div>" for t, h in secs)
    page = f"""<title>{html.escape(a.title)}</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:23px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 18px;max-width:92ch}} .kv{{font:13px ui-monospace,Menlo,monospace;color:var(--mut);margin:0 0 8px}} h2{{font-size:17px;margin:28px 0 8px;color:var(--acc)}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
table.rt{{border-collapse:collapse;font:12px ui-monospace,Menlo,monospace;margin:10px 0 0;font-variant-numeric:tabular-nums}}
.rt td,.rt th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}} .rt th{{color:var(--acc);font-weight:500}}
tr.gz td{{color:#ffb070}} tr.bad td{{color:#f07070}}
</style>
<main><h1>{html.escape(a.title)}</h1>
<p class="sub">{html.escape(a.note)}</p>
{body}</main>"""
    open(os.path.join(SP, a.out), "w").write(page)
    print(f"wrote {a.out} ({len(page)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
