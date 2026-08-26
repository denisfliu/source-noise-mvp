"""Two LIBERO basis questions (Denis, 2026-08-09).

1. Is the gripper eating the pin? Measure how much of the shared basis loads on the gripper
   dimension, and build a GRIPPER-FREE basis (basis rows at gripper positions zeroed, then
   re-orthonormalised) so the pin constrains arm motion only and the gripper stays fully
   learned. Writes pin_U_rrr_k5_nogrip.npy.
2. Do the suites want different subspaces? LIBERO is single-embodiment (Franka Panda); the
   analogous split is the four task suites (indices 0-9 long-horizon, 10-19 goal, 20-29
   object, 30-39 spatial). Fit a per-suite K=5 chunk-PCA basis and report variance capture
   and principal angles against the shared basis and against each other.

Variance capture is the metric here, not RRR predictability (that needs a VLM feature pass);
capture bounds what any pin in that subspace can carry.
"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RD = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero")
H, AD, K = 50, 32, 5
GRIP = 6
SUITES = {"long": (0, 10), "goal": (10, 20), "object": (20, 30), "spatial": (30, 40)}


def main():
    import openpi.shared.normalize as NZ
    from openpi import transforms as T
    from openpi.transforms import NormStats

    ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_libero_shared/physical-intelligence/libero"))

    def pads(nsd, dim):
        out = {}
        for k, s in nsd.items():
            n = np.asarray(s.mean).shape[-1]
            if n >= dim:
                out[k] = s; continue
            p = dim - n
            ext = lambda a, f: None if a is None else np.concatenate(
                [np.asarray(a, np.float32), np.full(p, f, np.float32)])
            out[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
        return out
    nrm = T.Normalize(pads(ns, AD), use_quantiles=False)
    U = np.load(f"{RD}/pin_U_rrr_k5_shared.npy").astype(np.float64)

    grip_rows = [t * AD + GRIP for t in range(H)]
    arm_rows = [t * AD + j for t in range(H) for j in range(6)]
    print(f"shared basis: {np.linalg.norm(U[grip_rows])**2 / np.linalg.norm(U)**2 * 100:.1f}% of its "
          f"squared norm sits on the GRIPPER dimension "
          f"({np.linalg.norm(U[arm_rows])**2 / np.linalg.norm(U)**2 * 100:.1f}% on the 6 arm dims)")

    # gripper-free basis: zero the gripper rows, re-orthonormalise
    Ug = U.copy(); Ug[grip_rows] = 0.0
    Q, _ = np.linalg.qr(Ug)
    Ug = Q[:, :K]
    np.save(f"{RD}/pin_U_rrr_k5_nogrip.npy", Ug.astype(np.float32))
    print(f"gripper-free basis written; residual gripper norm {np.linalg.norm(Ug[grip_rows]):.2e}")

    # chunk sample per suite
    files = sorted(glob.glob(f"{SRC}/data/chunk-*/episode_*.parquet"))
    per_suite = {k: [] for k in SUITES}
    for f in files[::3]:
        tb = pq.read_table(f, columns=["actions", "task_index"])
        ti = int(tb.column("task_index")[0].as_py())
        suite = next(s for s, (lo, hi) in SUITES.items() if lo <= ti < hi)
        ac = np.asarray(tb.column("actions").to_pylist(), np.float32)
        for t in range(0, max(len(ac) - H, 1), 16):
            ch = np.zeros((H, AD), np.float32)
            m = min(H, len(ac) - t)
            ch[:m, :ac.shape[1]] = ac[t:t + m]
            if m < H:
                ch[m:, :ac.shape[1]] = ac[-1]
            per_suite[suite].append(nrm({"actions": ch})["actions"].reshape(-1))
    per_suite = {k: np.array(v) for k, v in per_suite.items() if v}
    allY = np.concatenate(list(per_suite.values()))
    print(f"\nchunks: " + ", ".join(f"{k} {len(v)}" for k, v in per_suite.items()))

    def capture(Y, B):
        Yc = Y - Y.mean(0)
        return float((((Yc @ B) @ B.T) ** 2).sum() / ((Yc ** 2).sum() + 1e-9))

    def pca(Y, k=K):
        Yc = Y - Y.mean(0)
        _, _, Vt = np.linalg.svd(Yc[:4000], full_matrices=False)
        return Vt[:k].T

    print("\nvariance capture of the pooled chunks:")
    print(f"  shared RRR basis      {capture(allY, U) * 100:5.1f}%")
    print(f"  gripper-free basis    {capture(allY, Ug) * 100:5.1f}%  (arm dims only)")
    print(f"  pooled PCA-5 (bound)  {capture(allY, pca(allY)) * 100:5.1f}%")

    print("\nper suite: capture by the shared basis vs its own PCA-5 basis")
    bases = {}
    for s, Y in per_suite.items():
        bases[s] = pca(Y)
        print(f"  {s:8s} shared {capture(Y, U) * 100:5.1f}%   own {capture(Y, bases[s]) * 100:5.1f}%   "
              f"gripper-free {capture(Y, Ug) * 100:5.1f}%")

    def angles(A, B):
        s = np.linalg.svd(A.T @ B, compute_uv=False)
        return np.degrees(np.arccos(np.clip(s, -1, 1))).round(0)

    for s_, B in bases.items():
        np.save(f"{RD}/pin_U_suite_{s_}_k5.npy", B.astype(np.float32))
    print("\nsaved per-suite bases: " + ", ".join(f"pin_U_suite_{k}_k5.npy" for k in bases))
    print("\nprincipal angles between suite bases (0 = same subspace):")
    ks = list(bases)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            print(f"  {ks[i]:8s} vs {ks[j]:8s} {angles(bases[ks[i]], bases[ks[j]])}")
    for s in ks:
        print(f"  {s:8s} vs shared   {angles(bases[s], U)}")
    print("\nLIBERO_BASIS_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
