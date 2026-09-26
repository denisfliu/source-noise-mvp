"""Point-cloud page for one Sketch Lab sketch flown by lab_fly.py (2026-09-25): the sketch, every method's flights
(orange if they touch a gate before the route ends, with a red tick at the closest approach), and a results table.

  /home/dfliu/code/tv/bin/python build_lab_page.py <slug> <scene>
"""
import glob, html, json, os, sys

import numpy as np
import torch

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP)
sys.path.insert(0, SP); sys.path.insert(0, RD)
import cloudviewer  # noqa: E402
import gate_clearance as G  # noqa: E402
import route_contact as RC  # noqa: E402
from build_traj_page import axes, marks  # noqa: E402

RUN = "/home/dfliu/ctxrun"
NAMES = {"ours": "Ours", "vproj": "Velocity projection on π0", "sdedit": "SDEdit on π0", "inject": "π0 + injected sketch"}
COLS = [[96, 235, 160], [60, 190, 130], [30, 140, 100], [180, 140, 255], [90, 170, 240], [230, 120, 200], [245, 215, 110], [230, 180, 60], [200, 140, 30]]


def main():
    slug, scene = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]                      # more slugs flown on the same sketch (e.g. <slug>-gmsig3), shown alongside
    res = json.load(open(f"{RUN}/lab_{slug}_results.json"))
    S = np.asarray(json.load(open(f"{RD}/sketch_lab_{slug}_s0.json"))["points"], np.float64)[:, :3]
    items = [(slug, key, r) for key, r in res["results"].items()]
    for sl in extra:
        items += [(sl, key, r) for key, r in json.load(open(f"{RUN}/lab_{sl}_results.json"))["results"].items()]
    cloud = torch.tensor(np.asarray(G.gate_cloud(scene), np.float32))
    groups, ticks, rows = [], [], []
    for k, (sl, key, r) in enumerate(items):
        arm, _, sig = key.partition("@"); label = NAMES[arm] + (f" σ {sig}" if sig else "")
        if arm == "ours":
            label = label.replace("Ours", "Ours (mixed-data gmsig3)" if sl.endswith("-gmsig3") else "Ours (real-only)")
        ok, bad = [], []
        for f in sorted(glob.glob(f"{RUN}/traj_lab_{sl}_{key.replace('@', '_s')}_[0-9]*.npy")):
            P = np.load(f)[:, :3].astype(np.float64); e = RC.route_end(P, S)
            d = torch.cdist(torch.from_numpy(P[:e + 1].astype(np.float32)), cloud).min(1).values.numpy(); j = int(d.argmin())
            (ok if d[j] >= RC.BODY else bad).append(P.astype(np.float32))
            if d[j] < RC.BODY:
                ticks.append(np.stack([P[j] - [0, 0, 0.12], P[j] + [0, 0, 0.12]]).astype(np.float32))
        if ok: groups.append({"label": f"{label}: no contact ({len(ok)})", "color": COLS[k % len(COLS)], "trajs": ok})
        if bad: groups.append({"label": f"{label}: touches a gate ({len(bad)})", "color": [255, 140, 40], "trajs": bad})
        rows.append((label, r))
    if ticks:
        groups.append({"label": "closest approach of each touching flight", "fixed": True, "color": [240, 80, 80], "trajs": ticks})
    groups += [{"label": "the sketch", "fixed": True, "color": [255, 200, 90], "trajs": [S.astype(np.float32)]}] + marks(scene) + axes()
    v = cloudviewer.viewer_html(scene, groups, elem_id="v0", max_pts=40000,
                                note="Orange flights touch a gate before the route ends; red ticks mark where. Yellow is the sketch.")
    tab = ("<table class='rt'><tr><th>method</th><th>completed</th><th>closest to a gate (median)</th><th>tracking</th></tr>"
           + "".join(f"<tr><td>{html.escape(l)}</td><td>{r['completed']}/{r['n']}</td><td>{r['closest_m'] * 100:.1f} cm</td>"
                     f"<td>{r['tracking_cm']:.1f} cm</td></tr>" for l, r in rows) + "</table>")
    src = open(f"{SP}/build_traj_page.py").read()
    css = src[src.index("<style>"):src.index("</style>") + 8].replace("{{", "{").replace("}}", "}")
    css = css.replace("</style>", ".bgbtn{font:12px ui-monospace,Menlo,monospace;padding:3px 9px;border-radius:5px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer}</style>")
    page = f"""<title>{html.escape(res['name'])}</title>{css}<main><h1>{html.escape(res['name'])}</h1>
<p class="sub">A Sketch Lab sketch in the {scene.replace('_', ' ')} scene, flown in simulation by each method with a 25-step replan and the
same seed. A flight is completed if it passes the gate the right way and reaches the end of the sketch without touching a gate.</p>
<div class='vc'>{v}{tab}</div></main>"""
    open(f"{SP}/lab_{slug}.html", "w").write(page); print(f"wrote lab_{slug}.html")


if __name__ == "__main__":
    main()
