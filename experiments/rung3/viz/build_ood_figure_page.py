"""Figure panels for the Table 1 OOD sketches (2026-09-26): one representative flight of our policy on the orbit and the
figure-eight, drawn like the flawed-sketch panels (white ground, gates as cropped blue frames, black sketch). The
representative is the completed flight with the median tracking error to the sketch. Both panels share one camera
only if you want them to; each rotates on its own here, so set each angle and screenshot.

  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
      ~/code/openpi/.venv/bin/python build_ood_figure_page.py
"""
import glob, html, json, os, sys

import numpy as np
import torch
import yaml

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP)
sys.path.insert(0, SP); sys.path.insert(0, RD)
import cloudviewer  # noqa: E402
import gate_clearance as G  # noqa: E402
import route_contact as RC  # noqa: E402
from build_flawed_one_page import GATE_COL, SKETCH_COL  # noqa: E402

RUN = "/home/dfliu/ctxrun"
FLIGHT_COL = [47, 111, 168]    # "ours" blue of results_bars
CELLS = [("orbit", "Orbit Gate", "right", "right_gate", "ro25_orbit_wide", "sketch_orbit_wide.json"),
         ("fig8", "Figure-Eight", "left_and_center", "left_and_center", "ro25_fig8_denis3_cx15", "sketch_fig8_denis3_cx15.json")]


def gate_corners(safety):
    s = yaml.safe_load(open(os.path.expanduser(f"~/code/falsify-pi/configs/safety/{safety}.yaml")))
    gs = s["ordered_miss_gate"]["gates"] if "ordered_miss_gate" in s else [s["miss_gate"]]
    return [np.asarray(g["corners"], np.float64) for g in gs]


def near_frame(pts, corners, reach):
    res = []
    for C in corners:
        a, b, top = C[0, :2], C[1, :2], C[:, 2].max()
        L = np.linalg.norm(b - a); u = (b - a) / L; rel = pts[:, :2] - a
        res.append((np.abs(rel @ np.array([-u[1], u[0]])) < reach, rel @ u, L, top))
    return res


def figure_cloud(scene, safety, out, gate_col=GATE_COL):
    """The scene cloud with its gate points swapped for saturated blue gate frames cropped to the opening."""
    Z = np.load(f"{SP}/scene_cloud_{scene}.npz"); pts, rgb = Z["pts"].astype(np.float32), Z["rgb"].copy()
    corners = gate_corners(safety); drop = pts[:, 2] > 1.8
    for near, along, L, top in near_frame(pts, corners, 0.15):
        drop |= near & (pts[:, 2] > 0.15) & (along > -0.25) & (along < L + 0.25)
        drop |= near & (pts[:, 2] > 1.5) & (along > -0.6) & (along < L + 0.6)
    pts, rgb = pts[~drop], rgb[~drop]
    g = np.asarray(G.gate_cloud(scene), np.float32); keep = np.zeros(len(g), bool)
    for near, along, L, top in near_frame(g, corners, 0.15):
        keep |= near & (along > -0.07) & (along < L + 0.07) & (g[:, 2] < top + 0.09)
    g = g[keep]
    _, inv, cnt = np.unique(np.floor(g / 0.04).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    g = g[cnt[inv.ravel()] >= 15]; g = g[np.random.default_rng(0).permutation(len(g))[:12000]]
    np.savez(f"{SP}/scene_cloud_{out}.npz", pts=np.concatenate([pts, g]),
             rgb=np.concatenate([rgb, np.tile(np.array(gate_col, np.uint8), (len(g), 1))]))


def representative(tag, S, cloud):
    Q, _ = RC.dense(S); rows = []
    for f in sorted(glob.glob(f"{RUN}/traj_{tag}_[0-9]*.npy")):
        P = np.load(f)[:, :3].astype(np.float64); e, dmin = RC.contact(P, S, cloud)
        ok = e < len(P) - 1 and dmin >= RC.BODY
        trk = np.linalg.norm(P[:e + 1, None] - Q[None], axis=2).min(1).mean()
        rows.append((ok, trk, P[:e + 1], os.path.basename(f), len(rows)))
    pool = sorted([r for r in rows if r[0]] or rows, key=lambda r: r[1])
    return pool[len(pool) // 2], sum(r[0] for r in rows), len(rows)


def main():
    cells, notes = [], []
    for k, (key, title, scene, safety, tag, sk) in enumerate(CELLS):
        S = np.asarray(json.load(open(f"{RD}/{sk}"))["points"], np.float64)[:, :3]
        cloud = torch.tensor(np.asarray(G.gate_cloud(scene), np.float32))
        (ok, trk, P, fn, _), n_ok, n = representative(tag, S, cloud)
        out = f"oodfig_{key}"; figure_cloud(scene, safety, out)
        groups = [{"label": "flight", "fixed": True, "color": FLIGHT_COL, "trajs": [P.astype(np.float32)]},
                  {"label": "sketch", "fixed": True, "color": SKETCH_COL, "trajs": [S.astype(np.float32)]}]
        view = cloudviewer.viewer_html(out, groups, elem_id=f"o{k}", max_pts=None, default_on=True, white=True, square=True)
        os.remove(f"{SP}/scene_cloud_{out}.npz")
        i = view.index(f'<canvas id="o{k}"'); j = view.index("</canvas>", i) + len("</canvas>")
        ttl = f'<div class="ttl" contenteditable="true" spellcheck="false">{html.escape(title)}</div>'
        cells.append(f'<div class="cell">{view[:i]}<div class="figwrap">{view[i:j]}{ttl}</div>{view[j:]}</div>')
        notes.append(f"{title}: {fn}, tracking {trk * 100:.1f} cm (median of the {n_ok}/{n} completed flights)")
    page = f"""<title>OOD Sketch Panels</title>
<style>
body{{margin:0;background:#ffffff;color:#1b1f24;font:15px/1.5 system-ui,sans-serif;padding:20px 16px}}
main{{max-width:1100px;margin:0 auto}} .sub{{color:#5d6570;margin:0 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
@media (max-width:700px){{.grid{{grid-template-columns:1fr}}}}
.cell{{min-width:0}} .cell .v3dwrap{{background:#fff;border:0;padding:0;margin:0}} .figwrap{{position:relative}}
canvas{{width:100%;display:block;border-radius:6px;border:1px solid #e3e3e3;cursor:grab}}
.ttl{{position:absolute;top:8px;left:10px;background:rgba(255,255,255,.9);border-radius:5px;padding:3px 8px;font:600 14px system-ui,sans-serif;outline:none}}
.ttl:focus,.key:focus{{box-shadow:0 0 0 2px #1a73e8}}
.key{{display:flex;gap:6px 16px;flex-wrap:wrap;align-items:center;font:600 14px system-ui,sans-serif;margin:0 0 10px;outline:none}}
.key span{{display:inline-block;width:26px;height:5px;border-radius:3px;margin-right:6px;vertical-align:middle}}
.v3dui,.v3dnote{{display:none}} ul{{color:#5d6570;font-size:13px}}
</style>
<main><p class="sub">Drag to rotate, wheel to zoom, shift-drag to pan; click a label to edit it; then screenshot. One representative
flight of our policy per sketch (the completed flight with the median tracking error).</p>
<div class="key" contenteditable="true" spellcheck="false"><span style="background:rgb(15,15,15)"></span>sketch<span style="background:rgb(47,111,168)"></span>ours<span style="background:rgb(0,70,230)"></span>gate</div>
<div class="grid">{"".join(cells)}</div><ul>{"".join(f"<li>{html.escape(t)}</li>" for t in notes)}</ul></main>"""
    open(f"{SP}/ood_sketch_panels.html", "w").write(page); print("wrote ood_sketch_panels.html"); print("\n".join(notes))


if __name__ == "__main__":
    main()
