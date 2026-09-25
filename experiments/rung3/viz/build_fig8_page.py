"""Figure-eight through the left and centre gates (2026-09-25): the hand-drawn sketch flown by ours, SDEdit and velocity
projection, and all three again on the same sketch with the centre-gate crossing moved +0.15 m in x. Flights that touch a gate
before the route end are orange, with a red tick at the closest approach.

  /home/dfliu/code/tv/bin/python build_fig8_page.py
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
SECTIONS = [("The hand-drawn sketch", "fig8_denis3",
             [("ours", "ro25_fig8_denis3", [96, 235, 160]), ("SDEdit on pi0", "sde25_fig8_denis3", [90, 170, 240]),
              ("velocity projection on pi0", "vproj25_fig8_denis3", [180, 140, 255])]),
            ("Centre-gate crossing moved +0.15 m in x", "fig8_denis3_cx15",
             [("ours", "ro25_fig8_denis3_cx15", [96, 235, 160]), ("SDEdit on pi0", "sde25_fig8_denis3_cx15", [90, 170, 240]),
              ("velocity projection on pi0", "vproj25_fig8_denis3_cx15", [180, 140, 255])])]


def section(k, title, sk, arms, cloud):
    S = np.asarray(json.load(open(f"{RD}/sketch_{sk}.json"))["points"], np.float64)[:, :3]
    groups, rows, ticks = [], [], []
    for lab, pre, col in arms:
        ok, bad = [], []
        for f in sorted(glob.glob(f"{RUN}/traj_{pre}_[0-9]*.npy"), key=lambda f: int(f.rsplit("_", 1)[1][:-4])):
            P = np.load(f)[:, :3].astype(np.float64); e = RC.route_end(P, S)
            d = torch.cdist(torch.from_numpy(P[:e + 1].astype(np.float32)), cloud).min(1).values.numpy()
            j = int(d.argmin()); dm = float(d[j])
            (ok if dm >= RC.BODY else bad).append(P.astype(np.float32))
            rows.append((lab, f.rsplit("_", 1)[1][:-4], dm, j, P[j]))
            if dm < RC.BODY:
                ticks.append(np.stack([P[j] - [0, 0, 0.12], P[j] + [0, 0, 0.12]]).astype(np.float32))
        if ok: groups.append({"label": f"{lab}: no contact ({len(ok)})", "color": col, "trajs": ok})
        if bad: groups.append({"label": f"{lab}: touches a gate ({len(bad)})", "color": [255, 140, 40], "trajs": bad})
    if ticks:
        groups.append({"label": "closest approach of each touching flight", "fixed": True, "color": [240, 80, 80], "trajs": ticks})
    groups += [{"label": "the sketch", "fixed": True, "color": [255, 200, 90], "trajs": [S.astype(np.float32)]}] + marks("left_and_center") + axes()
    Q, _ = RC.dense(S); ds = torch.cdist(torch.from_numpy(Q.astype(np.float32)), cloud).min(1).values.numpy()
    v = cloudviewer.viewer_html("left_and_center", groups, elem_id=f"v{k}", max_pts=40000,
                                note="Orange flights touch a gate before the route ends; red ticks mark where. Yellow is the sketch.")
    tab = ("<table class='rt'><tr><th>method</th><th>flight</th><th>closest to a gate before the route end</th><th>at step</th>"
           "<th>where (x, y, z)</th></tr>" + "".join(
               f"<tr class='{'' if dm >= RC.BODY else 'gz'}'><td>{html.escape(l)}</td><td>{t}</td><td>{dm:.2f} m</td><td>{j}</td>"
               f"<td>({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})</td></tr>" for l, t, dm, j, p in rows) + "</table>")
    sub = f"The sketch itself comes {ds.min():.2f} m from a gate at ({Q[ds.argmin()][0]:.2f}, {Q[ds.argmin()][1]:.2f})."
    return f"<h2>{html.escape(title)}</h2><p class='sub'>{sub}</p><div class='vc'>{v}{tab}</div>"


def main():
    cloud = torch.tensor(np.asarray(G.gate_cloud("left_and_center"), np.float32))
    body = "".join(section(k, *s, cloud) for k, s in enumerate(SECTIONS))
    src = open(f"{SP}/build_traj_page.py").read()
    css = src[src.index("<style>"):src.index("</style>") + 8].replace("{{", "{").replace("}}", "}")
    page = f"""<title>Figure-Eight Through Two Gates</title>{css}<main><h1>Figure-Eight Through Two Gates</h1>
<p class="sub">The hand-drawn figure-eight through the left and centre gates, flown in simulation 10 times per method with a
25-step replan and the same seeds. SDEdit and velocity projection steer the plain real-only pi0. On the original sketch every
one of our flights cuts the turn at the centre gate and touches its post; the second section moves the four sketch points
around that crossing +0.15 m in x (neighbours +0.075 m).</p>{body}</main>"""
    open(f"{SP}/fig8_denis3_three.html", "w").write(page); print("wrote fig8_denis3_three.html")


if __name__ == "__main__":
    main()
