"""Completed routes on the three OOD cells of Table 1 for one arm (2026-09-25). A route is completed when the flight
reaches the end of its sketch without touching a gate (route_contact); Arbitrary Gates also needs moved_gate_cell's
route check (right gate, right direction, goal), and the figure-eight must cross the left and then the center gate in
the sketch's direction with no wrong-way pass.

  /home/dfliu/code/tv/bin/python score_ood_cells.py --reloc rr25mg --orbit ro25_orbit_wide --fig8 ro25_fig8_denis3_cx15
"""
import argparse, glob, json, os, sys

import numpy as np
import torch

RD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, RD)
import gate_clearance as G  # noqa: E402
import moved_gate_cell as MG  # noqa: E402
import route_contact as RC  # noqa: E402
import yaml  # noqa: E402

RUN = "/home/dfliu/ctxrun"
POSES = [("-163,0.04,-1.09", 51), ("174,2.81,1.37", 52), ("2,1.06,2.68", 53), ("153,2.46,-0.3", 54), ("-58,1.55,1.76", 55),
         ("132,1.69,-0.83", 56), ("-49,-0.06,2.97", 57), ("44,2.03,2.6", 58), ("130,0.42,-0.36", 59), ("-86,2.1,0.87", 60),
         ("10,-1.73,2.16", 61), ("34,-2.6,2.2", 62), ("107,-1.29,0.24", 63)]


def trajs(pre):
    return sorted(glob.glob(f"{RUN}/traj_{pre}_[0-9]*.npy"), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))


def reached_clean(P, S, cloud):
    e, dmin = RC.contact(P, S, cloud)
    return e < len(P) - 1 and dmin >= RC.BODY


def crossings(P, corners):
    """(+n crossings, -n crossings) of the aperture, with n oriented to +y (the sketch flies both gates toward +y)."""
    C = np.asarray(corners, np.float64); c = C.mean(0); u = C[1] - C[0]; v = C[3] - C[0]
    n = np.cross(u, v); n /= np.linalg.norm(n); n *= np.sign(n[1])
    hu, hv = np.linalg.norm(u) / 2, np.linalg.norm(v) / 2; u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    s = (P - c) @ n; fwd = back = 0; first = None
    for i in np.where(s[:-1] * s[1:] < 0)[0]:
        q = P[i] + s[i] / (s[i] - s[i + 1]) * (P[i + 1] - P[i]) - c
        if abs(q @ u) <= hu and abs(q @ v) <= hv:
            if s[i + 1] > s[i]:
                fwd += 1; first = i if first is None else first
            else:
                back += 1
    return fwd, back, first


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reloc"); ap.add_argument("--orbit"); ap.add_argument("--fig8")
    a = ap.parse_args()
    if a.reloc:
        ok = n = 0
        for spec, seed in POSES:
            dyaw, dx, dy = map(float, spec.split(",")); pt = spec.translate(str.maketrans({"-": "m", ",": "_", ".": "_"})) + f"_{seed}"
            cloud = torch.tensor(np.asarray(G.moved_gate_cloud(dyaw, (dx, dy)), np.float32))
            S = np.asarray(json.load(open(f"{RD}/sketch_mg_rr25mg{pt}.json"))["points"], np.float64)[:, :3]
            for f in trajs(a.reloc + pt):
                P = np.load(f)[:, :3].astype(np.float64); n += 1
                ok += bool(MG.score_traj(P, dyaw, dx, dy)["ok"]) and reached_clean(P, S, cloud)
        print(f"Arbitrary Gates {a.reloc}: {ok}/{n}")
    if a.orbit:
        cloud = torch.tensor(np.asarray(G.gate_cloud("right"), np.float32))
        S = np.asarray(json.load(open(f"{RD}/sketch_orbit_wide.json"))["points"], np.float64)[:, :3]
        fs = trajs(a.orbit); ok = sum(reached_clean(np.load(f)[:, :3].astype(np.float64), S, cloud) for f in fs)
        print(f"Orbit Gate {a.orbit}: {ok}/{len(fs)}")
    if a.fig8:
        cloud = torch.tensor(np.asarray(G.gate_cloud("left_and_center"), np.float32))
        S = np.asarray(json.load(open(f"{RD}/sketch_fig8_denis3_cx15.json"))["points"], np.float64)[:, :3]
        gates = [g["corners"] for g in yaml.safe_load(open(os.path.expanduser(
            "~/code/falsify-pi/configs/safety/left_and_center.yaml")))["ordered_miss_gate"]["gates"]]
        fs = trajs(a.fig8); ok = 0
        for f in fs:
            P = np.load(f)[:, :3].astype(np.float64)
            (lf, lb, l0), (cf, cb, c0) = crossings(P, gates[0]), crossings(P, gates[1])
            ok += reached_clean(P, S, cloud) and lf >= 1 and cf >= 1 and lb == 0 and cb == 0 and l0 < c0
        print(f"Figure-Eight {a.fig8}: {ok}/{len(fs)}")


if __name__ == "__main__":
    main()
