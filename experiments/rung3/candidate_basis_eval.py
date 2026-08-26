"""Candidate pin bases, evaluated offline on BOTH purposes before any GPU is spent
(Denis, 2026-08-20: "other pins that might be even more expressive?").

Every candidate is a set of linear functionals of the normalized zero-padded H=50 chunk
(linearity is structural: it is what makes c ride the interpolant exactly), orthonormalized
by QR. Axes restricted to the live dims {x,y,z,yaw} (4-6 have zero std in the demos).

Reported per candidate, on identical stride-8 rows matching vlm_feat_gate_prefix_local.npz:
  K            pinned dims (noise DOF handed to the command source)
  cap L/R mid  within-task capture, [0.5,0.75) chunk-start phase, left / right task
  cap L/R stop within-task capture, chunks reaching past episode end (t > T-H)
  R2 all/tail  held-out ridge R2 (cached VLM prefix feats -> c), mean over components,
               all rows / tail rows (frac > 0.7)  [proxy: cache is from gate_both_pin]
  R2 worst     the worst single component's held-out R2 (a component the head cannot
               predict is a channel serving noise)

  python3 candidate_basis_eval.py
"""
import json
import os

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
H, AD = 50, 32
STRIDE = 8
AXES = [0, 1, 2, 3]
NS = json.load(open(os.path.expanduser(
    "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
AMEAN, ASTD = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
TASKS = {"cfl": range(0, 50), "cfr": range(50, 100), "left": range(100, 150), "right": range(150, 200)}


def seg_to_Y(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - AMEAN[:r]) / (ASTD[:r] + 1e-6)
    return ch.reshape(-1)


def w2d(fn):
    """(H,) template applied to one axis -> flattened (H*AD,) functional."""
    cols = []
    for j in AXES:
        w = np.zeros((H, AD), np.float32)
        w[:, j] = fn
        cols.append(w.reshape(-1))
    return cols


def qr(cols):
    U, _ = np.linalg.qr(np.stack(cols, 1))
    return U.astype(np.float32)


def boxcar(a, b):
    f = np.zeros(H, np.float32)
    f[a:b] = 1.0
    return f


def build_candidates(Y, task_of, tr):
    C = {}
    C["flat_rrr5"] = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32)
    C["mh16"] = np.load(f"{RD}/pin_U_mh16.npy").astype(np.float32)
    # contiguous window displacements (temporal reallocation directly)
    segs = [(0, 12), (12, 25), (25, 37), (37, 50)]
    C["seg16"] = qr(sum((w2d(boxcar(a, b)) for a, b in segs), []))
    # dyadic Haar family per axis: full sum, half diff, quarter diffs, eighth diffs
    haar = [boxcar(0, 50)]
    for splits in ([(0, 25, 50)], [(0, 12, 25), (25, 37, 50)],
                   [(0, 6, 12), (12, 18, 25), (25, 31, 37), (37, 43, 50)]):
        for a, m, b in splits:
            haar.append(boxcar(a, m) - boxcar(m, b) * ((m - a) / max(b - m, 1)))
    C["haar32"] = qr(sum((w2d(f) for f in haar), []))
    # low-order DCT of the per-step profile
    t = (np.arange(H) + 0.5) / H
    for m, name in ((6, "dct24"), (8, "dct32")):
        fns = [np.cos(np.pi * k * t).astype(np.float32) for k in range(m)]
        C[name] = qr(sum((w2d(f) for f in fns), []))
    # mh16 + explicit terminal-velocity functionals (mean of last 6 / last 12 steps)
    tv = [boxcar(H - 6, H) / 6.0, boxcar(H - 12, H) / 12.0]
    C["mh16+tv8"] = qr([C["mh16"][:, k] for k in range(16)] + sum((w2d(f) for f in tv), []))
    # within-task PCA (center per task so between-task variance doesn't dominate)
    Yc = Y.copy()
    for tk in np.unique(task_of):
        m = (task_of == tk) & tr
        Yc[task_of == tk] -= Y[m].mean(0)
    w, V = np.linalg.eigh(np.cov(Yc[tr].T))
    C["winpca16"] = V[:, ::-1][:, :16].astype(np.float32)
    # mh16 + top-8 within-task PCA of the residual (data-driven top-up of what mh16 misses)
    P = C["mh16"] @ C["mh16"].T
    R = Yc[tr] - Yc[tr] @ P
    w, V = np.linalg.eigh(np.cov(R.T))
    C["mhres24"] = qr([C["mh16"][:, k] for k in range(16)] + [V[:, -1 - j] for j in range(8)])
    return C


def main():
    meta_rows = []
    Ys, task_of, frac, epi = [], [], [], []
    tasks_list = []
    for ti, (task, eps) in enumerate(TASKS.items()):
        tasks_list.append(task)
        for e in eps:
            d = np.load(f"{RD}/data_gate_synth/ep_{e:04d}.npz", allow_pickle=True)
            ac = d["action"].astype(np.float32)
            T = len(ac)
            for t in range(0, T, STRIDE):
                Ys.append(seg_to_Y(ac[t:]))
                task_of.append(ti)
                frac.append(t / T)
                epi.append(e)
    Y = np.stack(Ys)
    task_of, frac, epi = np.array(task_of), np.array(frac), np.array(epi)

    # feature cache rows were built in the same episode/stride order by make_u_rrr_gate_local
    z = np.load(f"{RD}/vlm_feat_gate_prefix_local.npz")
    X = z["X"].astype(np.float32)
    assert len(X) == len(Y), f"cache rows {len(X)} != chunk rows {len(Y)}"
    rng = np.random.default_rng(0)
    perm = rng.permutation(200)
    tr_eps = set(np.array(sorted({e for e in epi}))[perm[:160]].tolist())
    tr = np.array([e in tr_eps for e in epi])
    te = ~tr

    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    lam = 100.0
    A = Xb[tr].T @ Xb[tr] + lam * np.eye(Xb.shape[1], dtype=np.float32)
    Ainv_Xt = np.linalg.solve(A, Xb[tr].T)

    def cap(U, mask, ti):
        m = mask & (task_of == ti)
        Yc = Y[m] - Y[m].mean(0)
        return float(((Yc @ U) ** 2).sum() / ((Yc ** 2).sum() + 1e-9))

    li, ri = tasks_list.index("left"), tasks_list.index("right")
    mid = (frac >= 0.5) & (frac < 0.75)
    stop_m = np.array([f > 1 - H / (H / (f + 1e-9) + 1e-9) if False else False for f in frac])
    # stop rows: chunk reaches past episode end. frac alone can't tell; recompute per row:
    stop_rows = []
    k = 0
    for task, eps in TASKS.items():
        for e in eps:
            d = np.load(f"{RD}/data_gate_synth/ep_{e:04d}.npz", allow_pickle=True)
            T = len(d["action"])
            for t in range(0, T, STRIDE):
                stop_rows.append(t > T - H)
    stop_m = np.array(stop_rows)
    tail = frac > 0.7

    Cands = build_candidates(Y, task_of, tr)
    print(f"{'basis':10s} {'K':>3s}  {'capL.mid':>8s} {'capR.mid':>8s} {'capL.stop':>9s} "
          f"{'capR.stop':>9s}  {'R2all':>6s} {'R2tail':>6s} {'R2worst':>7s}")
    for name, U in Cands.items():
        Cc = Y @ U
        W = Ainv_Xt @ Cc[tr]
        P = Xb[te] @ W
        r2c = 1 - ((Cc[te] - P) ** 2).sum(0) / (((Cc[te] - Cc[te].mean(0)) ** 2).sum(0) + 1e-9)
        tte = te & tail
        Pt = Xb[tte] @ W
        r2t = 1 - ((Cc[tte] - Pt) ** 2).sum(0) / (((Cc[tte] - Cc[tte].mean(0)) ** 2).sum(0) + 1e-9)
        print(f"{name:10s} {U.shape[1]:3d}  {cap(U, mid, li):8.3f} {cap(U, mid, ri):8.3f} "
              f"{cap(U, stop_m, li):9.3f} {cap(U, stop_m, ri):9.3f}  "
              f"{r2c.mean():6.3f} {r2t.mean():6.3f} {r2c.min():7.3f}")


if __name__ == "__main__":
    main()
