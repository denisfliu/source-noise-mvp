"""Variance-collapse pre-check for the source-noise pin on LIBERO.

The hard pin sets the instruction coordinate c = U^T a to a deterministic value, i.e.
zero variance on the pinned subspace U. This is safe only if the target's conditional
variance along U is small. We cannot cheaply condition on pi0's full observation, so we
decompose the variance of each mode's coordinate into a BETWEEN-task part (variation the
task instruction explains, which the pin legitimately steers) and a WITHIN-task part
(variation not explained by the language instruction). Within-task variance is a
conservative UPPER BOUND on the truly unconditional spread, because it still contains
variation explained by the per-demo initial state (part of pi0's observation). If even
this upper bound is small on the modes we pin, the pin does not collapse meaningful
target variance. Computed on raw action chunks (a linear proxy for the model's
delta/normalized action space; U is a fixed smooth basis so the variance structure
along it is preserved under the roughly-linear transforms).
"""
import glob, json, os
import numpy as np
import pandas as pd

H = 50; REAL = 7; K = 16; W = 0.5; STRIDE = 25
DATA = os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero/data")
N_EP = int(os.environ.get("SNMVP_NEP", "500"))


def grid_laplacian_dirs(h, c, w):
    d = h * c
    L = np.zeros((d, d))
    idx = lambda t, ch: t * c + ch
    for ch in range(c):
        for t in range(h - 1):
            i, j = idx(t, ch), idx(t + 1, ch)
            L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
    for t in range(h):
        for a in range(c):
            for b in range(a + 1, c):
                i, j = idx(t, a), idx(t, b)
                L[i, i] += w; L[j, j] += w; L[i, j] -= w; L[j, i] -= w
    L = 0.5 * (L + L.T)
    _, evecs = np.linalg.eigh(L)
    return evecs.T          # (d,d), smoothest first


def main():
    files = sorted(glob.glob(os.path.join(DATA, "chunk-*/*.parquet")))[:N_EP]
    dirs = grid_laplacian_dirs(H, REAL, W)          # (350,350)
    U = dirs[:K]                                    # (K,350)
    chunks, tasks = [], []
    for f in files:
        d = pd.read_parquet(f, columns=["actions", "task_index"])
        a = np.stack(d["actions"].to_numpy())       # (T,7)
        ti = int(d["task_index"].iloc[0])
        if a.shape[0] < H:
            continue
        for s in range(0, a.shape[0] - H + 1, STRIDE):
            chunks.append(a[s:s + H].reshape(-1))    # (350,) row-major t*7+c
            tasks.append(ti)
    X = np.array(chunks); tasks = np.array(tasks)
    C = X @ U.T                                      # (n_chunks, K) coordinates
    n, ntasks = len(X), len(set(tasks.tolist()))
    print(f"chunks={n} tasks={ntasks} from {len(files)} episodes; K={K}")

    # energy: fraction of total chunk variance captured by the K low modes
    Call = X @ dirs.T
    energy_frac = float((Call[:, :K].var(0).sum()) / (Call.var(0).sum() + 1e-12))
    print(f"energy in top-{K} grid-Laplacian modes = {energy_frac:.3f} of total chunk variance")

    # per-mode between-task vs within-task variance decomposition
    print(f"{'mode':>4} {'total_var':>10} {'within_frac':>11} {'between_frac':>12}")
    tot = C.var(0)
    within = np.zeros(K)
    for t in set(tasks.tolist()):
        m = tasks == t
        if m.sum() < 2:
            continue
        within += C[m].var(0) * m.sum()
    within /= n
    for k in range(K):
        wf = float(within[k] / (tot[k] + 1e-12))
        print(f"{k:>4} {tot[k]:>10.4f} {wf:>11.3f} {1 - wf:>12.3f}")
    print(f"MEAN within_frac over modes = {float(np.mean(within / (tot + 1e-12))):.3f}")
    print("VC_PRECHECK_DONE=ok")


if __name__ == "__main__":
    main()
