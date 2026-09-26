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


if __name__ == "__main__":
    main()
