"""How much of the demo chunk variance does the pin basis capture, BY TRAJECTORY PHASE?

The endgame question in one number per (task, phase): capture = ||P_U (y-ybar)||^2 / ||y-ybar||^2
over chunks starting in that phase bin, with y the normalized zero-padded H=50 chunk exactly as
the flow trains on (SNMVP_ZERO_PAD_ACTIONS=1 convention) and ybar the bin mean. Box reference
(RESEARCH_LOG 2026-08-13): flat K=5 captured 0.34 of stop-segment variance, mh16 0.81 — the flat
basis could not EXPRESS the stop. This reruns that measurement against the rebuilt local basis so
the tail-capacity question is grounded in current data before any basis work.

  python3 basis_phase_capture.py [--u pin_U_gate_rrr_k5.npy]
"""
import argparse
import json
import os

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
H, AD = 50, 32
STRIDE = 4
NS = json.load(open(os.path.expanduser(
    "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
AMEAN, ASTD = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
TASKS = {"center_from_left": range(0, 50), "center_from_right": range(50, 100),
         "left": range(100, 150), "right": range(150, 200)}
# phase bins by chunk-start fraction t/T; "stop" additionally = chunks reaching past the episode
# end (t > T-H), i.e. the windows that contain the settle + zero-pad stop signature
BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def seg_to_Y(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - AMEAN[:r]) / (ASTD[:r] + 1e-6)
    return ch.reshape(-1)


def capture(Y, U):
    Yc = Y - Y.mean(0)
    return float(((Yc @ U) ** 2).sum() / ((Yc ** 2).sum() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", default=f"{RD}/pin_U_gate_rrr_k5.npy")
    a = ap.parse_args()
    U = np.load(a.u).astype(np.float32)
    print(f"basis {os.path.basename(a.u)} K={U.shape[1]}   capture = in-span variance fraction")
    hdr = "".join(f"  [{lo:.2f},{hi:.2f})" for lo, hi in BINS)
    print(f"{'task':18s}{hdr}      stop(t>T-H)   n_stop")
    allY = {i: [] for i in range(len(BINS))}
    allS = []
    for task, eps in TASKS.items():
        Yb = {i: [] for i in range(len(BINS))}
        Ys = []
        for e in eps:
            d = np.load(f"{RD}/data_gate_synth/ep_{e:04d}.npz", allow_pickle=True)
            ac = d["action"].astype(np.float32)
            T = len(ac)
            for t in range(0, T, STRIDE):
                y = seg_to_Y(ac[t:])
                f = t / T
                for i, (lo, hi) in enumerate(BINS):
                    if lo <= f < hi:
                        Yb[i].append(y)
                        allY[i].append(y)
                if t > T - H:
                    Ys.append(y)
                    allS.append(y)
        row = "".join(f"  {capture(np.stack(Yb[i]), U):11.3f}" for i in range(len(BINS)))
        print(f"{task:18s}{row}      {capture(np.stack(Ys), U):11.3f}   {len(Ys):6d}")
    row = "".join(f"  {capture(np.stack(allY[i]), U):11.3f}" for i in range(len(BINS)))
    print(f"{'ALL':18s}{row}      {capture(np.stack(allS), U):11.3f}   {len(allS):6d}")


if __name__ == "__main__":
    main()
