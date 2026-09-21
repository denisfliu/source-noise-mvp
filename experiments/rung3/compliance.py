"""Command compliance of a flown trajectory (2026-09-20).

E_comp = || (U^T a_hat - c) / sigma_c ||_rms, the deviation between the command the sketch issued at a
replan and the command realized by the 50 steps actually flown, in units of the training corpus's
per-coordinate command standard deviation.

The rollout logs positions only, so the yaw channel of every horizon band is zeroed on BOTH sides and the
score runs over the 12 translation coordinates of the 16. The sketch tracker is replayed exactly as the
server ran it (stateful, monotone), so the commanded value at replan k is the one that was served.

  python compliance.py --sketch sketch_cmpl_min4_w3.json --apc 50 --traj ~/ctxrun/traj_ws_w3_pin05_*.npy
"""
import argparse, glob, json, os
import numpy as np
from sketch_prompt import SketchPrompt

RD = os.path.dirname(os.path.abspath(__file__))
H, AD = 50, 32
NORM = "/home/dfliu/code/openpi-snmvp/assets/pi0_gate3/local/gate_nav3/norm_stats.json"


def norm_stats():
    a = json.load(open(NORM))["norm_stats"]["actions"]
    return np.asarray(a["mean"][:7], np.float32), np.asarray(a["std"][:7], np.float32)


def chunk_from_positions(P, i, amean, astd):
    """Normalized (H, AD) chunk of the H steps executed from index i (translation dims only)."""
    d = np.diff(P[i:i + H + 1, :3], axis=0)
    ch = np.zeros((H, AD), np.float32)
    n = len(d)
    ch[:n, :3] = (d - amean[:3]) / (astd[:3] + 1e-6)
    return ch


def sigma_c(U, amean, astd, demos):
    """Per-coordinate std of c = U^T a over 50-step chunks of the training corpus (translation dims)."""
    cs = []
    for P in demos:
        for i in range(0, max(1, len(P) - H), H):
            if i + H + 1 > len(P):
                break
            cs.append(chunk_from_positions(P, i, amean, astd).reshape(-1) @ U)
    return np.std(np.stack(cs), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", required=True)
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--apc", type=int, default=50)
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    U = np.load(a.pin_u).astype(np.float32)
    amean, astd = norm_stats()
    import realism as R
    d = R.load_demos()
    demos = [np.asarray(e, np.float32)[:, :3]
             for src in ("real", "synth") for grp in d[src].values() for e in grp]
    sc = sigma_c(U, amean, astd, demos)
    keep = sc > 1e-3
    rows = []
    for f in sorted(sum([glob.glob(os.path.expanduser(t)) for t in a.traj], [])):
        P = np.load(f)[:, :3].astype(np.float32)
        sk = SketchPrompt(a.sketch if os.path.isabs(a.sketch) else f"{RD}/{a.sketch}", amean, astd, U)
        trial = os.path.basename(f)
        errs = []
        for k in range(0, (len(P) - 1) // a.apc):
            i = k * a.apc
            ch_cmd, _sig, _pr, phase = sk.window(trial, P[i])
            if phase != 1 or ch_cmd is None or i + H + 1 > len(P):
                continue
            ch_cmd = np.asarray(ch_cmd, np.float32).copy()
            ch_cmd[:, 3:] = 0.0                                   # positions only, both sides
            c = ch_cmd.reshape(-1) @ U
            c_hat = chunk_from_positions(P, i, amean, astd).reshape(-1) @ U
            errs.append(np.sqrt(np.mean((((c_hat - c) / sc)[keep]) ** 2)))
        if errs:
            rows.append((os.path.basename(f), len(errs), float(np.median(errs)), float(np.percentile(errs, 90))))
    if not rows:
        print("no sketch-active replans found"); return
    med = np.median([r[2] for r in rows])
    print(f"== compliance {a.label or a.sketch}: {len(rows)} flights, sigma_c |.| {np.linalg.norm(sc[keep]):.2f}")
    for n, k, m, p in rows:
        print(f"  {n:34s} replans {k:2d}  E_comp median {m:.3f}  p90 {p:.3f}  (sigma_c units)")
    print(f"  == median over flights {med:.3f} sigma_c")


if __name__ == "__main__":
    main()
