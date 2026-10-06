"""Table 1's demonstrated-task flights in the point cloud (paper figure, 2026-09-26): every one of the n=100 simulation
flights per policy and gate, one panel per policy, so the spread through the opening can be compared. Panels of the
same gate share one camera. Drawn like the flawed-sketch panels (white ground, gates cropped to the opening), gates in
dark grey. Successful flights (Table 1's rule, as rescore_tables: correct-direction transit, no wrong-way crossing,
>= 0.18 m from the gate cloud) keep the policy colour (pi0 green, ours blue). Failures are split: crashed (comes
within 0.18 m of the gate cloud; red, black blob at the closest point) and missed (no contact, but no correct-direction
transit or a wrong-way crossing; orange). Failures are drawn on top.

  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
      ~/code/openpi/.venv/bin/python build_crossing_cloud_page.py
"""
import glob, html, os, sys

import numpy as np
import torch

SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(SP))
import cloudviewer  # noqa: E402
from build_ood_figure_page import figure_cloud  # noqa: E402
import gate_clearance as G  # noqa: E402
import gate_success as GS  # noqa: E402
from build_flawed_one_page import blob  # noqa: E402
from catalogue import AUTO  # noqa: E402

RUN = "/home/dfliu/ctxrun"
GATE_COL, PI0_COL, OURS_COL, FAIL_COL, MISS_COL = [70, 70, 70], [30, 150, 80], [47, 111, 168], [214, 39, 40], [245, 150, 0]
GATES = [("left", "left", "left_gate", "Left Gate", "armscrreal_apc25_left", "armrealonly_apc25_left"),
         ("right", "right", "right_gate", "Right Gate", "armscrreal_apc25_right", "armrealonly_apc25_right")]


def main():
    cells = []
    for key, scene, safety, gate, pre_pi0, pre_ours in GATES:
        out = f"xcloud_{key}"; figure_cloud(scene, safety, out, GATE_COL)
        cloud = torch.tensor(np.asarray(G.gate_cloud(scene), np.float32))
        for name, pre, col in (("π0", pre_pi0, PI0_COL), ("Ours", pre_ours, OURS_COL)):
            ok, crash, miss, hits = [], [], [], []
            for f in sorted(glob.glob(f"{RUN}/traj_{pre}_[0-9]*.npy")):
                P = np.load(f)[:, :3].astype(np.float64); j = GS.judge(P, key)
                good = j["transit"] and j["wrong_dir_crossings"] == 0 and AUTO[os.path.basename(f)[:-4]]["minclr"] >= 0.18
                if good:
                    ok.append(P.astype(np.float32)); continue
                d = torch.cdist(torch.from_numpy(P.astype(np.float32)), cloud).min(1).values
                if d.min() < 0.18:
                    crash.append(P.astype(np.float32)); hits.append(blob(P[int(d.argmin())], r=0.04, n=250))
                else:
                    miss.append(P.astype(np.float32))
            T = ok + crash + miss; k = len(cells)
            groups = [{"label": "succeeded", "fixed": True, "color": col, "trajs": ok},
                      {"label": "missed", "fixed": True, "color": MISS_COL, "trajs": miss},
                      {"label": "crashed", "fixed": True, "color": FAIL_COL, "trajs": crash},
                      {"label": "contact", "fixed": True, "color": [0, 0, 0], "trajs": hits}]
            print(gate, name, f"{len(ok)}/{len(T)} succeeded, {len(crash)} crashed, {len(miss)} missed")
            view = cloudviewer.viewer_html(out, groups, elem_id=f"x{k}",
                                           max_pts=None, default_on=True, white=True, sync=f"gate_{key}", square=True)
            i = view.index(f'<canvas id="x{k}"'); j = view.index("</canvas>", i) + len("</canvas>")
            ttl = f'<div class="ttl" contenteditable="true" spellcheck="false">{html.escape(f"{gate}, {name}: {len(ok)} succeed, {len(crash)} crash, {len(miss)} miss")}</div>'
            cells.append(f'<div class="cell">{view[:i]}<div class="figwrap">{view[i:j]}{ttl}</div>{view[j:]}</div>')
        os.remove(f"{SP}/scene_cloud_{out}.npz")
    page = f"""<title>Gate Crossings</title>
<style>
body{{margin:0;background:#ffffff;color:#1b1f24;font:15px/1.5 system-ui,sans-serif;padding:20px 16px}}
main{{max-width:1100px;margin:0 auto}} .sub{{color:#5d6570;margin:0 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
@media (max-width:700px){{.grid{{grid-template-columns:1fr}}}}
.cell{{min-width:0}} .cell .v3dwrap{{background:#fff;border:0;padding:0;margin:0}} .figwrap{{position:relative}}
canvas{{width:100%;display:block;border-radius:6px;border:1px solid #e3e3e3;cursor:grab}}
.ttl{{position:absolute;top:8px;left:10px;background:rgba(255,255,255,.9);border-radius:5px;padding:3px 8px;font:600 14px system-ui,sans-serif;outline:none}}
.ttl:focus{{box-shadow:0 0 0 2px #1a73e8}}
.v3dui,.v3dnote{{display:none}}
</style>
<main><p class="sub">All 100 simulated flights per policy on the demonstrated tasks (Table 1); crashed flights in red (black blob at the contact), missed flights in orange. Drag to rotate: the two panels of
each gate turn together (wheel zooms, shift-drag pans). Click a label to edit it, then screenshot.</p>
<div class="grid">{"".join(cells)}</div></main>"""
    open(f"{SP}/gate_crossings.html", "w").write(page); print("wrote gate_crossings.html")


if __name__ == "__main__":
    main()
