"""Tracking error of flights against a sketch: per-flight median and max distance from the
flown positions to the sketch polyline (densified at 1 cm) while the sketch is ACTIVE: from
the first approach within enter_radius to the handback (first step within 0.6 m of the
sketch's last point, the SketchPrompt DONE rule), or to the end of the flight if it never
arrives. The number the log calls 'tracking'.

  python sketch_track.py --sketch sketch_orbit.json --traj T1.npy [T2.npy ...]
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", required=True)
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--from-step", type=int, default=0, help="ignore steps before this (post-kick rejoin)")
    a = ap.parse_args()
    d = json.load(open(a.sketch))
    P = np.asarray(d["points"], np.float64)[:, :3]
    enter = float(d.get("enter_radius", 0.5))
    seg = [np.linspace(P[i], P[i + 1], max(2, int(np.linalg.norm(P[i + 1] - P[i]) / 0.01) + 1)) for i in range(len(P) - 1)]
    dense = np.concatenate(seg)
    meds = []
    for f in a.traj:
        T = np.load(f)[:, :3].astype(np.float64)
        dist = np.sqrt(((T[:, None, :] - dense[None, :, :]) ** 2).sum(-1)).min(1)
        on = np.where(dist < enter)[0]
        if len(on) == 0:
            print(f"{os.path.basename(f):32s} never within enter_radius of the sketch"); continue
        end_d = np.linalg.norm(T - P[-1], axis=1)
        arrived = np.where(end_d[on[0]:] < 0.6)[0]
        stop = on[0] + int(arrived[0]) + 1 if len(arrived) else len(T)
        active = dist[max(on[0], a.from_step):stop]
        if len(active) == 0:
            print(f"{os.path.basename(f):32s} no active steps after --from-step"); continue
        meds.append(np.median(active))
        print(f"{os.path.basename(f):32s} track median {np.median(active):.3f} m  p90 {np.percentile(active, 90):.3f}  max {active.max():.3f}  (steps {on[0]}-{stop}{'' if len(arrived) else ', never arrived'})")
    if meds:
        print(f"== tracking median over flights {np.median(meds):.3f} m (range {min(meds):.3f}-{max(meds):.3f})")


if __name__ == "__main__":
    main()
