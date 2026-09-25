"""Gate contact up to the end of a sketched route (2026-09-25, Denis: contact after the route ends is an artefact of
how the sketch hands control back, not of the policy, so it is not counted).

The route end is where the flight has progressed along the sketch to within END_M of its last point. Progress is
tracked monotonically (nearest sketch point within a window ahead of the current progress), so loops that end near
their start (the hand-drawn figure-eight) do not end at takeoff.

  /home/dfliu/code/tv/bin/python route_contact.py --scene left_and_center --sketch sketch_fig8_denis3.json \
      --traj ~/ctxrun/traj_ro25_fig8_denis3_*.npy [--gate-tf=-45,0,0]
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_clearance as G  # noqa: E402

BODY, END_M, BACK, AHEAD = 0.18, 0.25, 0.3, 1.0


def dense(S, step=0.02):
    pts = [S[0]]
    for a, b in zip(S[:-1], S[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    Q = np.array(pts); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    return Q, s


def route_end(P, S):
    """Index of the first step whose sketch progress reaches the last END_M of the sketch (len(P)-1 if never)."""
    Q, s = dense(S); prog = 0.0
    for i, p in enumerate(P):
        w = (s >= prog - BACK) & (s <= prog + AHEAD)
        j = np.where(w)[0][np.argmin(np.linalg.norm(Q[w] - p, axis=1))]
        prog = max(prog, s[j])
        if prog >= s[-1] - END_M:
            return i
    return len(P) - 1


def contact(P, S, cloud):
    e = route_end(P, S)
    d = torch.cdist(torch.from_numpy(P[:e + 1].astype(np.float32)), cloud).min(1).values.numpy()
    return e, float(d.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--sketch", required=True)
    ap.add_argument("--traj", nargs="+", required=True); ap.add_argument("--gate-tf", default="")
    a = ap.parse_args()
    if a.gate_tf:
        dyaw, dx, dy = map(float, a.gate_tf.split(",")); C = G.moved_gate_cloud(dyaw, (dx, dy))
    else:
        C = G.gate_cloud(a.scene)
    cloud = torch.tensor(np.asarray(C, np.float32))
    S = np.asarray(json.load(open(a.sketch))["points"], np.float64)[:, :3]
    n = 0
    for f in a.traj:
        e, dmin = contact(np.load(f)[:, :3].astype(np.float64), S, cloud)
        n += dmin >= BODY
        print(f"{os.path.basename(f):36s} route end @step {e:4d}  closest to gate {dmin:.3f} m  {'clean' if dmin >= BODY else 'CONTACT'}")
    print(f"== {n}/{len(a.traj)} contact-free to the route end")


if __name__ == "__main__":
    main()
