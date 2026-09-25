"""Gate-generalization sweep helper (2026-08-28): for a given SE(2) gate move (dyaw about
the aperture centroid + dxy), emit (a) the auto 4-point sketch through the NEW pose,
(b) the transformed aperture corners, and score trajectories against them.

  python3 moved_gate_cell.py --make --dyaw 45 --dx 0 --dy 0 --tag g45
  python3 moved_gate_cell.py --score --tag g45 --traj ...npy ...
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import yaml

RD = os.path.dirname(os.path.abspath(__file__))
# Right-gate aperture, read from the judge's own safety YAML so the moved-gate scorer and gate_success agree
# (corners re-measured 2026-09-25: posts' inner edges, top bar's underside).
_C = np.asarray(yaml.safe_load(open(os.path.expanduser("~/code/falsify-pi/configs/safety/right_gate.yaml")))["miss_gate"]["corners"])
GA, GB = _C[0, :2], _C[1, :2]   # post A, post B (mocap xy)
CEN = (GA + GB) / 2
GOAL = np.array([1.525, -0.615, 1.0]); HALF = np.array([0.3, 0.3, 0.5])
START = np.array([0.0, 0.0, 1.5])
# The renderer (gsplat_scene_edit.apply_arbitrary_gate) rotates the gate about the centroid of its selected
# gaussians, not about the aperture midpoint CEN. Sketch and scorer must use the same pivot or every rotated pose
# is aimed off-centre by |(R - I)(PIVOT - CEN)|, up to 0.15 m at 180 deg (found 2026-09-24).
PIVOT = np.array([0.4842, -1.1597])
RUN_IN, RUN_OUT = 0.8, 0.5   # straight approach before and exit after the aperture (was 0.45 / 0.30)
ZC = 1.45
INSET, ZLO, ZHI = 0.0, float(_C[:, 2].min()), float(_C[:, 2].max())   # scorer: span inset, aperture bottom and top


def se2(dyaw_deg, dx, dy):
    th = math.radians(dyaw_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = PIVOT - R @ PIVOT + np.array([dx, dy])
    return R, t


def moved_geometry(dyaw_deg, dx, dy):
    R, t = se2(dyaw_deg, dx, dy)
    a, b = R @ GA + t, R @ GB + t
    tv = (b - a) / np.linalg.norm(b - a)
    n = np.array([tv[1], -tv[0]])          # original crossing was -y-ish: n points to exit side
    mid = (a + b) / 2
    return a, b, mid, tv, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--dyaw", type=float, default=0)
    ap.add_argument("--dx", type=float, default=0)
    ap.add_argument("--dy", type=float, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--traj", nargs="*", default=[])
    a = ap.parse_args()
    ga, gb, mid, tv, n = moved_geometry(a.dyaw, a.dx, a.dy)
    if a.make:
        sk = make_sketch(a.dyaw, a.dx, a.dy)
        out_p = f"{RD}/sketch_mg_{a.tag}.json"
        json.dump(sk, open(out_p, "w"), indent=1)
        print(f"wrote {out_p} ({len(sk['points'])} pts); aperture {np.round(ga, 2)}..{np.round(gb, 2)}")
        return
    nsucc = 0
    for f in a.traj:
        r = score_traj(np.load(f)[:, :3], a.dyaw, a.dx, a.dy)
        nsucc += r["ok"]
        print(f"  {os.path.basename(f):26s} cross={r['cross']} wrong={r['wrong']} goal={r['goal']} "
              f"min-post-dist={r['dpost']:.2f}  {'OK' if r['ok'] else 'fail'}")
    print(f"== {a.tag} (dyaw {a.dyaw}, dxy {a.dx},{a.dy}): {nsucc}/{len(a.traj)} route-clean")


def make_sketch(dyaw, dx, dy):
    """The auto sketch through the moved gate: straight run-in and run-out through the aperture midpoint, with
    around-post detours wherever another leg would cross the moved gate plane."""
    ga, gb, mid, tv, n = moved_geometry(dyaw, dx, dy)
    appr = mid - RUN_IN * n
    thru = mid + RUN_OUT * n
    L = np.linalg.norm(gb - ga)

    def crosses(p, q):
        """does segment p->q cross the aperture line within span+-0.25?"""
        dp_, dq_ = (p - ga) @ n, (q - ga) @ n
        if dp_ * dq_ >= 0:
            return False
        f = dp_ / (dp_ - dq_)
        sxx = ((p + f * (q - p)) - ga) @ tv
        return -0.25 < sxx < L + 0.25

    def detour(p, q):
        """route around the nearer post: exterior points beyond each post, pick shorter."""
        cands = [ga - 0.55 * tv, gb + 0.55 * tv]
        best = min(cands, key=lambda c: np.linalg.norm(p - c) + np.linalg.norm(c - q))
        return best

    wp = [START[:2], appr, thru, GOAL[:2]]
    # leg routing (2026-08-29): insert around-post detours where straight legs would
    # cross the moved plane — covers gates behind the start and across the return
    out = [wp[0]]
    for i, (p, q) in enumerate(zip(wp[:-1], wp[1:])):
        if i != 1 and crosses(p, q):     # never reroute the approach->thru crossing leg
            out.append(detour(p, q))
        out.append(q)
    zs = {0: 1.5, len(out) - 1: 1.2}
    pts = []
    for i, p in enumerate(out):
        nxt = out[min(i + 1, len(out) - 1)]
        prv = out[max(i - 1, 0)]
        seg = (nxt - p) if i < len(out) - 1 else (p - prv)
        yawv = math.atan2(seg[1], seg[0])
        pts.append([round(float(p[0]), 3), round(float(p[1]), 3),
                    zs.get(i, ZC), round(yawv, 3)])
    sk = {"points": pts,
          "prompt_after": "go through the gate on the right and hover over the stuffed animal",
          "enter_radius": 0.5, "step_m": 0.025, "sigma_serve": 0.0,
          "end_margin_m": 0.1, "carrot": 20}
    return sk


def score_traj(P, dyaw, dx, dy):
    """Route-clean on the moved gate: first crossing of the moved aperture along +n (inside the posts and between
    its bottom and top), no crossing the other way, and the goal box entered after the crossing."""
    ga, gb, mid, tv, n = moved_geometry(dyaw, dx, dy)
    L = np.linalg.norm(gb - ga)
    rel = P[:, :2] - ga
    s, d = rel @ tv, rel @ n
    inspan = (s > INSET) & (s < L - INSET)
    inz = (P[:, 2] > ZLO) & (P[:, 2] < ZHI)
    cross, wrongs = None, 0
    for i in range(len(P) - 1):
        if d[i] < 0 <= d[i + 1] and inspan[i + 1] and inz[i]:
            cross = i if cross is None else cross
        if d[i] >= 0 > d[i + 1] and inspan[i + 1] and inz[i]:
            wrongs += 1
    goal = cross is not None and bool(np.all(np.abs(P - GOAL) <= HALF, axis=1)[cross:].any())
    dpost = min(np.linalg.norm(P[:, :2] - ga, axis=1).min(), np.linalg.norm(P[:, :2] - gb, axis=1).min())
    return dict(cross=cross, wrong=wrongs, goal=goal, dpost=float(dpost), ok=cross is not None and wrongs == 0 and goal)


if __name__ == "__main__":
    main()
