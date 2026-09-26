"""One representative flight per method on the flawed left-then-center sketch, mixed-data checkpoints (2026-09-25).
Each legend entry names the method and what happened to that flight. The representative is the completed flight with
the median closest approach when the method completed any, otherwise the flight with the median closest approach.

  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
      ~/code/openpi/.venv/bin/python build_flawed_one_page.py
"""
import glob, html, json, os, sys

import numpy as np
import torch

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP)
sys.path.insert(0, SP); sys.path.insert(0, RD)
import cloudviewer  # noqa: E402
import gate_clearance as G  # noqa: E402
import gate_success as GS  # noqa: E402
import route_contact as RC  # noqa: E402
from build_traj_page import axes, marks  # noqa: E402

RUN = "/home/dfliu/ctxrun"; SCENE = "left_and_center"; SK = f"{RD}/sketch_cmpl_min4.json"
ARMS = [("Ours, σ 0", "bsm25_pin0_L", [120, 200, 255]), ("Ours, σ 0.5", "bsm25_pin05_L", [96, 235, 160]),
        ("Velocity projection on π0", "bsm25_vproj_L", [190, 140, 255]), ("Velocity projection on π0, s 0.1", "bsm25_vps01_L", [215, 170, 255]),
        ("Velocity projection on π0, s 0.3", "bsm25_vps03_L", [235, 200, 255]), ("Velocity projection on π0, s 0.5", "bsm25_vps05_L", [250, 225, 255]),
        ("SDEdit on π0", "bsm25_sde05_L", [255, 170, 90])]


def verdict(P, S, cloud):
    j = GS.judge_compound(P, SCENE); e, dmin = RC.contact(P, S, cloud)
    if j["gates_latched"] < 2:
        return False, dmin, ("never reaches the center gate" if j["gates_latched"] == 1 else "never passes the left gate")
    if sum(j["wrong_crossings"]):
        return False, dmin, "passes a gate the wrong way"
    if dmin < RC.BODY:
        return False, dmin, f"touches a post ({dmin * 100:.0f} cm)"
    if e >= len(P) - 1:
        return False, dmin, "does not reach the end of the sketch"
    return True, dmin, f"completed, {dmin * 100:.0f} cm from the nearest post"


def main():
    S = np.asarray(json.load(open(SK))["points"], np.float64)[:, :3]
    cloud = torch.tensor(np.asarray(G.gate_cloud(SCENE), np.float32))
    groups, rows = [], []
    for name, tag, col in ARMS:
        fs = sorted(glob.glob(f"{RUN}/traj_{tag}_[0-9]*.npy"))
        if not fs:
            continue
        v = [(f,) + verdict(np.load(f)[:, :3].astype(np.float64), S, cloud) for f in fs]
        pool = [x for x in v if x[1]] or v
        pool.sort(key=lambda x: x[2]); f, ok, dmin, why = pool[len(pool) // 2]
        n_ok = sum(x[1] for x in v)
        groups.append({"label": f"{name}: {why} ({n_ok}/{len(v)} completed)", "color": col,
                       "trajs": [np.load(f)[:, :3].astype(np.float32)]})
        rows.append((name, n_ok, len(v), why, os.path.basename(f)))
    groups += [{"label": "the flawed sketch", "fixed": True, "color": [255, 215, 80], "trajs": [S.astype(np.float32)]}] + marks(SCENE) + axes()
    view = cloudviewer.viewer_html(SCENE, groups, elem_id="v0", max_pts=40000,
                                   note="One flight per method. The legend says what that flight did and how many of the method's 10 flights completed. Yellow is the flawed sketch.")
    tab = ("<table class='rt'><tr><th>method</th><th>completed</th><th>the flight shown</th><th>file</th></tr>"
           + "".join(f"<tr><td>{html.escape(n)}</td><td>{k}/{m}</td><td>{html.escape(w)}</td><td>{html.escape(fn)}</td></tr>" for n, k, m, w, fn in rows)
           + "</table>")
    src = open(f"{SP}/build_traj_page.py").read()
    css = src[src.index("<style>"):src.index("</style>") + 8].replace("{{", "{").replace("}}", "}")
    css = css.replace("</style>", ".bgbtn{font:12px ui-monospace,Menlo,monospace;padding:3px 9px;border-radius:5px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer}</style>")
    page = f"""<title>Flawed Center-Gate Sketch</title>{css}<main><h1>Flawed Center-Gate Sketch</h1>
<p class="sub">A left-then-center sketch that passes 7 cm from the center gate's post, flown by the policies trained on real and
simulated demonstrations (25-step replan, same seed). One representative flight per method.</p>
<div class='vc'>{view}{tab}</div></main>"""
    open(f"{SP}/flawed_center_sketch.html", "w").write(page); print("wrote flawed_center_sketch.html")


if __name__ == "__main__" and not {"--figure", "--panels"} & set(sys.argv):
    main()


# ---- figure version (2026-09-25, for the paper figure): white background, legend drawn over the image, every flight on,
# darker colours, and a red blob where each flight touches a post. Screenshot the viewer at the angle you want.
FIG_ARMS = [("Ours, σ = 0.5", "bsm25_pin05_L", [26, 150, 90]), ("Ours, σ = 0", "bsm25_pin0_L", [40, 110, 200]),
            ("Velocity projection", "bsm25_vproj_L", [120, 60, 190]), ("Velocity projection, s = 0.1", "bsm25_vps01_L", [190, 110, 220]),
            ("SDEdit", "bsm25_sde05_L", [220, 120, 20])]
SHORT = {"never reaches the center gate": "misses the center gate", "never passes the left gate": "misses the left gate",
         "passes a gate the wrong way": "wrong-way pass", "does not reach the end of the sketch": "stops short"}


def blob(c, r=0.05, n=400):
    v = np.random.default_rng(0).normal(size=(n, 3)); v = v / np.linalg.norm(v, axis=1, keepdims=True) * r * np.cbrt(np.random.default_rng(1).random((n, 1)))
    return (c + v).astype(np.float32)


def figure():
    S = np.asarray(json.load(open(SK))["points"], np.float64)[:, :3]
    cloud = torch.tensor(np.asarray(G.gate_cloud(SCENE), np.float32))
    groups, legend, crashes = [], [], []
    for name, tag, col in FIG_ARMS:
        fs = sorted(glob.glob(f"{RUN}/traj_{tag}_[0-9]*.npy"))
        v = [(f,) + verdict(np.load(f)[:, :3].astype(np.float64), S, cloud) for f in fs]
        pool = [x for x in v if x[1]] or v
        pool.sort(key=lambda x: x[2]); f, ok, dmin, why = pool[len(pool) // 2]
        P = np.load(f)[:, :3].astype(np.float64)
        e = RC.route_end(P, S); d = torch.cdist(torch.from_numpy(P[:e + 1].astype(np.float32)), cloud).min(1).values.numpy()
        if d.min() < RC.BODY:
            crashes.append(blob(P[int(d.argmin())]))
        outcome = "completes" if ok else ("hits the post" if why.startswith("touches") else SHORT.get(why, why))
        groups.append({"label": f"{name}: {outcome}", "color": col, "trajs": [P.astype(np.float32)]})
        legend.append((f"{name}: {outcome}", col))
    if crashes:
        groups.append({"label": "contact with a post", "fixed": True, "color": [214, 39, 40], "trajs": crashes})
        legend.append(("contact with a post", [214, 39, 40]))
    groups += [{"label": "the flawed sketch", "fixed": True, "color": [200, 150, 0], "trajs": [S.astype(np.float32)]}] + marks(SCENE)
    legend.insert(0, ("the flawed sketch", [200, 150, 0]))
    view = cloudviewer.viewer_html(SCENE, groups, elem_id="v0", max_pts=40000, default_on=True, white=True, height=640)
    ovl = "".join(f'<div><span style="background:rgb({c[0]},{c[1]},{c[2]})"></span>{html.escape(t)}</div>' for t, c in legend)
    i = view.index('<canvas id="v0"'); j = view.index("</canvas>", i) + len("</canvas>")
    view = view[:i] + f'<div class="figwrap">{view[i:j]}<div class="ovl">{ovl}</div></div>' + view[j:]
    page = f"""<title>Flawed Sketch Figure</title>
<style>
body{{margin:0;background:#ffffff;color:#1b1f24;font:15px/1.5 system-ui,sans-serif;padding:20px 16px}}
main{{max-width:1200px;margin:0 auto}} .sub{{color:#5d6570;margin:0 0 12px}}
.v3dwrap{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}} .v3dwrap canvas{{width:100%;display:block;border-radius:6px}}
.figwrap{{position:relative}}
.ovl{{position:absolute;top:12px;left:12px;background:rgba(255,255,255,.92);border:1px solid #cfcfcf;border-radius:6px;padding:8px 12px;
  font:600 14px system-ui,sans-serif;color:#1b1f24;pointer-events:none;display:grid;gap:4px}}
.ovl span{{display:inline-block;width:26px;height:5px;border-radius:3px;margin-right:8px;vertical-align:middle}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace;color:#5d6570}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.bgbtn{{font:12px ui-monospace,Menlo,monospace;padding:3px 9px;border-radius:5px;border:1px solid #cfcfcf;background:transparent;color:inherit;cursor:pointer}}
.hint{{margin-left:auto}}
</style>
<main><p class="sub">Figure view: drag to orbit, wheel to zoom, shift-drag to pan, then screenshot. One representative flight per method on the flawed
left-then-center sketch; red blobs mark where a flight touches a post.</p>{view}</main>"""
    open(f"{SP}/flawed_center_figure.html", "w").write(page); print("wrote flawed_center_figure.html")


if __name__ == "__main__" and "--figure" in sys.argv:
    figure()


# ---- small multiples (2026-09-25): one panel per method, same scene, one shared camera; every label is editable in
# place (click it and type) so the figure's wording can be set before the screenshot.
PANELS = [("Ours, σ = 0.5", "bsm25_pin05_L"), ("Ours, σ = 0.3", "bsm25_pin03_L"), ("Velocity projection", "bsm25_vproj_L"),
          ("Velocity projection, s = 0.3", "bsm25_vps03_L"), ("SDEdit", "bsm25_sde05_L")]
TRAJ_COL = [26, 110, 200]


def panels():
    S = np.asarray(json.load(open(SK))["points"], np.float64)[:, :3]
    cloud = torch.tensor(np.asarray(G.gate_cloud(SCENE), np.float32))
    cells = []
    for k, (name, tag) in enumerate(PANELS):
        fs = sorted(glob.glob(f"{RUN}/traj_{tag}_[0-9]*.npy"))
        if not fs:
            print("skipping (not flown yet):", tag); continue
        v = [(f,) + verdict(np.load(f)[:, :3].astype(np.float64), S, cloud) for f in fs]
        pool = [x for x in v if x[1]] or v
        pool.sort(key=lambda x: x[2]); f, ok, dmin, why = pool[len(pool) // 2]
        P = np.load(f)[:, :3].astype(np.float64)
        e = RC.route_end(P, S); d = torch.cdist(torch.from_numpy(P[:e + 1].astype(np.float32)), cloud).min(1).values.numpy()
        groups = [{"label": "flight", "fixed": True, "color": TRAJ_COL, "trajs": [P.astype(np.float32)]},
                  {"label": "the flawed sketch", "fixed": True, "color": [200, 150, 0], "trajs": [S.astype(np.float32)]}]
        if d.min() < RC.BODY:
            groups.append({"label": "contact", "fixed": True, "color": [214, 39, 40], "trajs": [blob(P[int(d.argmin())])]})
        groups += [dict(g, fixed=True) for g in marks(SCENE)]
        n_ok = sum(x[1] for x in v)
        outcome = "completes" if ok else ("hits the post" if why.startswith("touches") else SHORT.get(why, why))
        view = cloudviewer.viewer_html(SCENE, groups, elem_id=f"p{k}", max_pts=30000, default_on=True, white=True, height=420, sync="flawed")
        i = view.index(f'<canvas id="p{k}"'); j = view.index("</canvas>", i) + len("</canvas>")
        title = f'<div class="ttl" contenteditable="true" spellcheck="false">{html.escape(name)}: {html.escape(outcome)} ({n_ok}/{len(v)})</div>'
        cells.append(f'<div class="cell"><div class="figwrap">{view[i:j]}{title}</div>{view[j:]}</div>')
    key = ('<div class="key" contenteditable="true" spellcheck="false"><span style="background:rgb(200,150,0)"></span>flawed sketch'
           '<span style="background:rgb(26,110,200)"></span>flight<span class="dot"></span>contact with a post</div>')
    page = f"""<title>Flawed Sketch Panels</title>
<style>
body{{margin:0;background:#ffffff;color:#1b1f24;font:15px/1.5 system-ui,sans-serif;padding:20px 16px}}
main{{max-width:1500px;margin:0 auto}} .sub{{color:#5d6570;margin:0 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px}}
.cell .v3dwrap,.cell{{background:#fff}} .figwrap{{position:relative}} canvas{{width:100%;display:block;border-radius:6px;border:1px solid #e3e3e3;cursor:grab}}
.ttl{{position:absolute;top:8px;left:10px;background:rgba(255,255,255,.9);border-radius:5px;padding:3px 8px;font:600 14px system-ui,sans-serif;outline:none}}
.ttl:focus,.key:focus{{box-shadow:0 0 0 2px #1a73e8}}
.key{{display:flex;gap:6px 16px;flex-wrap:wrap;align-items:center;font:600 14px system-ui,sans-serif;margin:0 0 10px;outline:none}}
.key span{{display:inline-block;width:26px;height:5px;border-radius:3px;margin-right:6px;vertical-align:middle}}
.key .dot{{width:12px;height:12px;border-radius:50%;background:rgb(214,39,40)}}
.v3dui{{display:none}} .v3dnote{{display:none}}
</style>
<main><p class="sub">Drag any panel to rotate; every panel follows (wheel zooms, shift-drag pans). Click any label to edit its text,
then screenshot. One representative flight per method on the flawed left-then-center sketch.</p>{key}<div class="grid">{"".join(cells)}</div></main>"""
    open(f"{SP}/flawed_center_panels.html", "w").write(page); print("wrote flawed_center_panels.html")


if __name__ == "__main__" and "--panels" in sys.argv:
    panels()
