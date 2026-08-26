"""Multi-horizon basis: build it and audit it offline (Denis approved 2026-08-12).

Builds the 16-column family — cumulative displacement over the first {6,12,25,50} steps for each
of {x,y,z,yaw}, hand-written linear functionals of the normalized chunk — orthonormalizes with QR
(nested sums -> per-band sums), saves pin_U_mh16.npy, and answers the two go/no-go questions
before any flow train:

  1. Conditioning: per-component ridge R2 (pi0_base post-fusion features -> c) split by
     trajectory segment (early/transit/tail). The claim: short-horizon components stay
     predictable at the tail where the flat 50-step basis degrades.
  2. Stop signature: does short-horizon ~= long-horizon displacement separate stopping frames
     from slow-cruising frames in the demos?
"""
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
from refit_rrr_basis import chunks  # zero-padded normalized chunks, cache row order

H, AD = 50, 32
HORIZONS = [6, 12, 25, 50]
AXES = [0, 1, 2, 3]  # x, y, z, yaw in the actuated leading dims


def build_family():
    W = []
    for h in HORIZONS:
        for j in AXES:
            w = np.zeros((H, AD), np.float32)
            w[:h, j] = 1.0
            W.append(w.reshape(-1))
    return np.stack(W, 1)  # (1600, 16)


def ridge_r2(X, Y, tr, te, lam=100.0):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    A = Xb[tr].T @ Xb[tr] + lam * np.eye(Xb.shape[1], dtype=np.float32)
    Wm = np.linalg.solve(A, Xb[tr].T @ Y[tr])
    P = Xb[te] @ Wm
    return P, np.array([1 - ((Y[te][:, k] - P[:, k]) ** 2).sum()
                        / (((Y[te][:, k] - Y[te][:, k].mean()) ** 2).sum() + 1e-9)
                        for k in range(Y.shape[1])])


def main():
    W = build_family()
    U, _ = np.linalg.qr(W)
    np.save(f"{RD}/pin_U_mh16.npy", U.astype(np.float32))
    labels = [f"h{h}:{'xyzw'[j]}" for h in HORIZONS for j in AXES]
    print(f"saved pin_U_mh16.npy {U.shape} (QR of nested sums -> per-band components)")

    z = np.load(f"{RD}/langprior_feats_base.npz")
    E, ep, frac = z["E"].astype(np.float32), z["ep"], z["frac"]
    Y = chunks(ep, None, "/home/ubuntu/hf_bundle/gate-drone-pi0/assets/gate_nav")
    C_mh = Y @ U
    Uflat = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32)
    C_fl = Y @ Uflat

    rng = np.random.default_rng(0)
    tr_eps = set(rng.permutation(200)[:160].tolist())
    tr = np.array([e in tr_eps for e in ep]); te = ~tr
    segs = {"early": frac < 0.33, "transit": (frac >= 0.33) & (frac < 0.75), "tail": frac >= 0.75}

    for name, C in (("multi-horizon (16)", C_mh), ("flat RRR k5 (deployed)", C_fl)):
        P, _ = ridge_r2(E, C, tr, te)
        print(f"\n== {name}: held-out per-component R2 by segment ==")
        hdr = labels if C is C_mh else [f"c{k}" for k in range(C.shape[1])]
        for sname, sm in segs.items():
            m = sm[te]
            r2s = [1 - ((C[te][m, k] - P[m, k]) ** 2).sum()
                   / (((C[te][m, k] - C[te][m, k].mean()) ** 2).sum() + 1e-9)
                   for k in range(C.shape[1])]
            row = " ".join(f"{h}={r:+.2f}" for h, r in zip(hdr, r2s))
            print(f"  {sname:8s} {row}")

    # stop signature: |6-step| / |50-step| displacement ratio, stopping vs cruising frames
    d6 = np.linalg.norm((Y.reshape(len(Y), H, AD)[:, :6, :3]).sum(1), axis=1)
    d50 = np.linalg.norm((Y.reshape(len(Y), H, AD)[:, :, :3]).sum(1), axis=1)
    ratio = d6 / (d50 + 1e-6)
    tail_m, mid_m = segs["tail"], segs["transit"]
    print(f"\nstop signature |d6|/|d50|: tail mean {ratio[tail_m].mean():.3f} "
          f"(p10 {np.percentile(ratio[tail_m], 10):.3f}) vs transit mean {ratio[mid_m].mean():.3f} "
          f"(p90 {np.percentile(ratio[mid_m], 90):.3f})")
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
