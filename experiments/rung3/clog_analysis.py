"""How far does the SERVED command drift from the demo-consistent one, and where?

CLOG rows are [pos(3), c(K), e64(64)] per inference. For each logged inference this finds the
nearest demo state for the same task and compares the served command against the demo's own command
there, in metres of chunk displacement. It also reports how far off the demo manifold the drone was
when the command was issued — the quantity no offline metric sees, because offline evaluation only
ever queries demo states.

  python clog_analysis.py --clog <files...> --task right [--emb]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openpi.training.config as C
import openpi.transforms as T
from openpi.shared.normalize import NormStats, load as load_ns

RD = os.path.dirname(os.path.abspath(__file__))
H = 50
# data_gate_synth task map: ep0000-0049 centre-from-left, 0050-0099 centre-from-right,
# 0100-0149 LEFT, 0150-0199 RIGHT
EPS = {"center_from_left": range(0, 50), "center_from_right": range(50, 100),
       "left": range(100, 150), "right": range(150, 200)}


def _pads(d, dim):
    o = {}
    for k, s in d.items():
        n = len(s.mean)
        if n >= dim:
            o[k] = s
            continue
        p = dim - n
        ext = lambda a, f: None if a is None else np.concatenate(
            [np.asarray(a, np.float32), np.full(p, f, np.float32)])
        o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
    return o


def demo_commands(task, U, norm_dir):
    """(positions, c, displacement-in-metres) for every demo timestep of one task."""
    AD = C.get_config("pi0_gate").model.action_dim
    PS = _pads(load_ns(norm_dir), AD)
    nrm = T.Normalize(PS, use_quantiles=False)
    mean = np.asarray(PS["actions"].mean, np.float32)
    std = np.asarray(PS["actions"].std, np.float32)
    P, Cs, V = [], [], []
    for i in EPS[task]:
        f = f"{RD}/data_gate_synth/ep_{i:04d}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        st, ac = d["state"].astype(np.float32), d["action"].astype(np.float32)
        for t in range(0, len(st) - 5):
            ch = np.zeros((H, AD), np.float32)
            m = min(H, len(ac) - t)
            ch[:m, :7] = ac[t:t + m]
            P.append(st[t, :3])
            nx = st[min(t + 5, len(st) - 1), :3] - st[t, :3]
            V.append(nx / (np.linalg.norm(nx) + 1e-6))
            Cs.append((nrm({"actions": ch})["actions"].reshape(-1)) @ U)
    return np.asarray(P), np.asarray(Cs, np.float32), np.asarray(V, np.float32), mean, std


def to_metres(c, U, mean, std, AD):
    ch = (U @ np.atleast_2d(c).T).T.reshape(-1, H, AD)
    return (ch * std[None, None, :] + mean[None, None, :])[:, :, :3].sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clog", nargs="+", required=True)
    ap.add_argument("--task", required=True, choices=list(EPS))
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_gate_rrr_k5.npy")
    ap.add_argument("--norm", default="/home/dfliu/hf_bundle/gate-drone-pi0/assets/gate_nav")
    a = ap.parse_args()
    U = np.load(a.pin_u).astype(np.float32)
    K, AD = U.shape[1], C.get_config("pi0_gate").model.action_dim
    P, Cd, V, mean, std = demo_commands(a.task, U, a.norm)
    Dd = to_metres(Cd, U, mean, std, AD)
    print(f"demo reference: {len(P)} timesteps from {a.task}")
    print(f"{'file':34s} {'n':>3s} {'dist to demo mfld (m)':>22s} {'|served-demo| cmd (m)':>22s}")
    rows = []
    for f in sorted(sum([glob.glob(p) for p in a.clog], [])):
        L = np.load(f)
        pos, c = L[:, :3], L[:, 3:3 + K]
        starts = np.where(np.linalg.norm(pos - np.array([0.0, 0.0, 1.5]), axis=1) < 0.05)[0]
        neps = max(1, len(starts))
        d = np.linalg.norm(P[None, :, :] - pos[:, None, :], axis=2)
        # rollout heading at each inference, from the logged positions
        hd = np.zeros_like(pos)
        hd[:-1] = pos[1:] - pos[:-1]
        hd[-1] = hd[-2] if len(pos) > 1 else 0
        hn = hd / (np.linalg.norm(hd, axis=1, keepdims=True) + 1e-6)
        # only consider demo timesteps travelling the same way (dot > 0.3); fall back to
        # position-only when a heading has no agreeing demo timestep nearby
        agree = (hn @ V.T) > 0.3
        dd = np.where(agree, d, np.inf)
        j = np.where(np.isfinite(dd).any(1), dd.argmin(1), d.argmin(1))
        off = d[np.arange(len(pos)), j]
        Ds = to_metres(c, U, mean, std, AD)
        err = np.linalg.norm(Ds - Dd[j], axis=1)
        rows.append((off, err, pos, Ds, Dd[j]))
        print(f"{os.path.basename(f) + f' [{neps} ep]':34s} {len(pos):3d} "
              f"{off.mean():8.3f} max {off.max():7.3f} {err.mean():10.3f} max {err.max():7.3f}")
    off = np.concatenate([r[0] for r in rows]); err = np.concatenate([r[1] for r in rows])
    print(f"\npooled: {len(off)} inferences, {100 * (off > 0.30).mean():.0f}% issued more than 0.30 m "
          f"off the demo manifold")
    for lo, hi, nm in ((0.0, 0.15, "on manifold  <0.15 m"), (0.15, 0.35, "near        0.15-0.35"),
                       (0.35, 1e9, "off manifold  >0.35 m")):
        m = (off >= lo) & (off < hi)
        if m.any():
            print(f"  {nm:22s} n={m.sum():4d}  mean command error {err[m].mean():.3f} m  "
                  f"(90th pct {np.percentile(err[m], 90):.3f})")


if __name__ == "__main__":
    main()
