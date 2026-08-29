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

RD = os.path.dirname(os.path.abspath(__file__))
GA = np.array([0.195, -1.348]); GB = np.array([0.924, -0.952])   # right gate posts (mocap)
CEN = (GA + GB) / 2
GOAL = np.array([1.525, -0.615, 1.0]); HALF = np.array([0.3, 0.3, 0.5])
START = np.array([0.0, 0.0, 1.5])
ZC = 1.45


def se2(dyaw_deg, dx, dy):
    th = math.radians(dyaw_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = CEN - R @ CEN + np.array([dx, dy])
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
        appr = mid - 0.45 * n
        thru = mid + 0.30 * n
        yaw0 = math.atan2(*(appr - START[:2])[::-1])
        yaw1 = math.atan2(*n[::-1])
        yaw2 = math.atan2(*(GOAL[:2] - thru)[::-1])
        sk = {"points": [[0.0, 0.0, 1.5, round(yaw0, 3)],
                         [round(appr[0], 3), round(appr[1], 3), ZC, round(yaw1, 3)],
                         [round(thru[0], 3), round(thru[1], 3), ZC, round(yaw1, 3)],
                         [round(GOAL[0], 3), round(GOAL[1], 3), 1.2, round(yaw2, 3)]],
              "prompt_after": "go through the gate on the right and hover over the stuffed animal",
              "enter_radius": 0.5, "step_m": 0.025, "sigma_serve": 0.0,
              "end_margin_m": 0.1, "carrot": 20}
        out = f"{RD}/sketch_mg_{a.tag}.json"
        json.dump(sk, open(out, "w"), indent=1)
        print(f"wrote {out}; aperture now {np.round(ga,2)}..{np.round(gb,2)}")
        return
    # score: crossing along +n within post span, wrong-dir count, goal dwell, post distance
    L = np.linalg.norm(gb - ga)
    nsucc = 0
    for f in a.traj:
        P = np.load(f)[:, :3]
        rel = P[:, :2] - ga
        s = rel @ tv
        d = rel @ n
        inspan = (s > 0.05) & (s < L - 0.05)
        cross = wrong = None
        wrongs = 0
        for i in range(len(P) - 1):
            if d[i] < 0 <= d[i + 1] and inspan[i + 1] and 0.2 < P[i, 2] < 1.95:
                cross = i if cross is None else cross
            if d[i] >= 0 > d[i + 1] and inspan[i + 1] and 0.2 < P[i, 2] < 1.95:
                wrongs += 1
        ing = np.all(np.abs(P - GOAL) <= HALF, axis=1)
        goal_after = cross is not None and bool(ing[cross:].any())
        dpost = min(np.linalg.norm(P[:, :2] - ga, axis=1).min(),
                    np.linalg.norm(P[:, :2] - gb, axis=1).min())
        ok = cross is not None and wrongs == 0 and goal_after
        nsucc += ok
        print(f"  {os.path.basename(f):26s} cross={cross} wrong={wrongs} goal={goal_after} "
              f"min-post-dist={dpost:.2f}  {'OK' if ok else 'fail'}")
    print(f"== {a.tag} (dyaw {a.dyaw}, dxy {a.dx},{a.dy}): {nsucc}/{len(a.traj)} route-clean")


if __name__ == "__main__":
    main()
