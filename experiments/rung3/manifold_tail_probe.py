"""Manifold-distance instrument for the endgame failure (Denis-approved, 2026-08-13).

Question: do closed-loop commands stay cruise-like exactly when the STATE has left the demo tube?
If yes, extrapolation (covariate shift) is the enabling condition of the stop failure and no head
architecture alone fixes it; if commands are wrong even ON the tube, the head/features are at fault
where data exists.

For every logged replan (CLOG rows: pos + c_hat) match the nearest same-task demo frame
(direction-aware: heading dot > 0.3 — position alone aliases outbound/return phases), giving
  d      = distance to the demo manifold
  c*     = the demo's oracle command at the matched frame (the arm's own basis, zero-pad chunks)
Report, per arm/side: command error vs d (binned), and per chunk index: mean d, command-magnitude
ratio |c_hat|/|c*| (cruise-at-the-goal shows as ratio >> 1 late).

Built-in control: b2lam03-left STOPS successfully (goal 10/10) while b2lam03-right overshoots —
if d(tail) is small on left and large on right, the covariate-shift account fits inside one arm.
"""
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
H, AD = 50, 32
TASKS = {"left": 2, "right": 3}  # langprior task order [CFL, CFR, LEFT, RIGHT]


def normalizer():
    import openpi.transforms as T
    from openpi.shared.normalize import NormStats, load as load_ns
    ns = load_ns("/home/ubuntu/hf_bundle/gate-drone-pi0/assets/gate_nav")
    o = {}
    for k, s in ns.items():
        n = len(s.mean)
        if n >= AD:
            o[k] = s; continue
        p = AD - n
        ext = lambda a, f: None if a is None else np.concatenate(
            [np.asarray(a, np.float32), np.full(p, f, np.float32)])
        o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
    return T.Normalize(o, use_quantiles=False)


def demo_bank(task, U, nrm):
    meta = json.load(open(f"{RD}/data_gate_synth/meta.json"))
    P, Hd, C = [], [], []
    for k in sorted(meta):
        if meta[k]["task"] != task:
            continue
        d = np.load(f"{RD}/data_gate_synth/{k}.npz")
        st, ac = d["state"].astype(np.float32), d["action"].astype(np.float32)
        for t in range(0, len(st) - 5, 3):
            v = ac[t:t + 10, :3].sum(0)
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                v = np.array([0, 0, 1e-6])
            ch = np.zeros((H, AD), np.float32)
            m = min(H, len(ac) - t)
            ch[:m, :7] = ac[t:t + m]
            P.append(st[t, :3]); Hd.append(v / max(nv, 1e-6))
            C.append(nrm({"actions": ch})["actions"].reshape(-1) @ U)
    return np.array(P), np.array(Hd), np.array(C)


def analyze(name, rows, task_name, U, nrm, nch):
    """rows: (n_rollouts*nch, 3+K) ordered per rollout."""
    P, Hd, C = demo_bank(TASKS[task_name], U, nrm)
    cstd = np.linalg.norm(C.std(0))
    K = U.shape[1]
    R = rows.reshape(-1, nch, 3 + K)
    recs = []
    for r in R:
        pos = r[:, :3]
        if nch > 1:
            hd = np.diff(np.vstack([pos, pos[-1:] * 2 - pos[-2:-1]]), axis=0)
        for i in range(nch):
            if nch > 1:
                h = hd[i] / max(np.linalg.norm(hd[i]), 1e-6)
                ok = (Hd @ h) > 0.3
            else:  # single-row (interleaved parallel-client log): no heading — position-only
                ok = np.ones(len(Hd), bool)  # match; phase aliasing possible, labeled in output
            if not ok.any():
                continue
            dd = np.linalg.norm(P - pos[i], axis=1); dd[~ok] = 1e9
            j = int(dd.argmin())
            recs.append((i, float(dd[j]), float(np.linalg.norm(r[i, 3:] - C[j]) / cstd),
                         float(np.linalg.norm(r[i, 3:]) / (np.linalg.norm(C[j]) + 1e-6))))
    recs = np.array(recs)
    print(f"\n== {name} [{task_name}]  matched {len(recs)} replans")
    print("   chunk:  " + " ".join(f"{i:6d}" for i in range(nch)))
    for lab, col in (("mean d (m)   ", 1), ("|c|/|c*|     ", 3), ("cmd err /std ", 2)):
        vals = [recs[recs[:, 0] == i, col].mean() if (recs[:, 0] == i).any() else np.nan
                for i in range(nch)]
        print(f"   {lab}" + " ".join(f"{v:6.2f}" for v in vals))
    bins = [0, 0.1, 0.2, 0.35, 0.6, 10]
    print("   err-vs-d bins:", end="")
    for a, b in zip(bins[:-1], bins[1:]):
        m = (recs[:, 1] >= a) & (recs[:, 1] < b)
        print(f"  [{a}-{b}):{recs[m, 2].mean():.2f}(n={m.sum()})" if m.any() else f"  [{a}-{b}):-",
              end="")
    print()


def main():
    nrm = normalizer()
    U5 = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32)
    U16 = np.load(f"{RD}/pin_U_mh16.npy").astype(np.float32)
    Uv = np.load(f"{RD}/pin_U_vla_base_k5.npy").astype(np.float32)

    # b2lam03: OLD sequential eval — rows interleaved per trial (left 8, right 8) x 10
    cl = np.load("/home/ubuntu/ctxrun/clog_b2lam03.npy")
    R = cl.reshape(10, 2, 8, 8)  # trial, side, chunk, 3+5
    analyze("b2lam03 (stops on left, overshoots on right)", R[:, 0].reshape(-1, 8),
            "left", U5, nrm, 8)
    analyze("b2lam03", R[:, 1].reshape(-1, 8), "right", U5, nrm, 8)

    # c2: left only (right client died) — sequential, 10 rollouts x 8
    c2 = np.load("/home/ubuntu/ctxrun/clog_c2.npy")
    analyze("c2 (skips the left gate)", c2, "left", Uv, nrm, 8)

    # mh16: parallel clients -> rows interleaved arbitrarily; attribute row-wise by best
    # task-bank match, then treat rows as independent replans (no per-rollout structure)
    mh = np.load("/home/ubuntu/ctxrun/clog_mh16.npy")
    for tn in ("left", "right"):
        P, Hd, C = demo_bank(TASKS[tn], U16, nrm)
        keep = []
        Po, Ho, Co = demo_bank(TASKS["right" if tn == "left" else "left"], U16, nrm)
        for r in mh:
            d_own = np.linalg.norm(P - r[:3], axis=1).min()
            d_oth = np.linalg.norm(Po - r[:3], axis=1).min()
            if d_own < d_oth:
                keep.append(r)
        keep = np.array(keep)
        if len(keep):
            analyze(f"mh16 (thrashes at the end; {len(keep)} rows attributed)",
                    keep.reshape(len(keep), 1, -1).reshape(-1, 19)[None, :, :].reshape(-1, 19)
                    if False else keep, tn, U16, nrm, 1)


if __name__ == "__main__":
    main()
