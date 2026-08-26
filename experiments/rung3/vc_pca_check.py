"""Is a data-fit (PCA) subspace a better instruction basis than the grid-Laplacian
for LIBERO actions, and does it transfer across TASKS (so it can be frozen and reused
few-shot)? Split tasks into source and held-out. For each K, report:
  - coverage on source (fraction of action-chunk variance the K-dim subspace captures),
  - cross-task transfer: relative reconstruction error of held-out-task chunks by the
    subspace fit on source tasks (lower = transfers across tasks),
for the grid-Laplacian basis and for PCA fit on source. Also the within/between-task
variance split of the PCA coordinate. Raw action space (linear proxy for the model's
delta/normalized space); the point is the relative comparison of the two bases.
"""
import glob, os
import numpy as np
import pandas as pd

H = 50; REAL = 7; W = 0.5; STRIDE = 25; D = H * REAL
DATA = os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero/data")
N_EP = int(os.environ.get("SNMVP_NEP", "1500"))
KS = [8, 16, 32, 64]


def grid_laplacian_dirs(h, c, w):
    d = h * c; L = np.zeros((d, d)); idx = lambda t, ch: t * c + ch
    for ch in range(c):
        for t in range(h - 1):
            i, j = idx(t, ch), idx(t + 1, ch); L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
    for t in range(h):
        for a in range(c):
            for b in range(a + 1, c):
                i, j = idx(t, a), idx(t, b); L[i, i] += w; L[j, j] += w; L[i, j] -= w; L[j, i] -= w
    L = 0.5 * (L + L.T); _, e = np.linalg.eigh(L); return e.T


def rel_err(X, U):                          # U (D,k) orthonormal cols
    P = U @ U.T
    return float(np.sqrt(((X - X @ P) ** 2).sum() / ((X ** 2).sum() + 1e-12)))


def pca(X, k):
    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False); return Vt[:k].T


def main():
    files = sorted(glob.glob(os.path.join(DATA, "chunk-*/*.parquet")))[:N_EP]
    chunks, tasks = [], []
    for f in files:
        d = pd.read_parquet(f, columns=["actions", "task_index"])
        a = np.stack(d["actions"].to_numpy())
        if a.shape[0] < H:
            continue
        ti = int(d["task_index"].iloc[0])
        for s in range(0, a.shape[0] - H + 1, STRIDE):
            chunks.append(a[s:s + H].reshape(-1)); tasks.append(ti)
    X = np.array(chunks); tasks = np.array(tasks)
    uniq = sorted(set(tasks.tolist()))
    rng = np.random.default_rng(0); rng.shuffle(uniq)
    n_src = max(1, int(0.8 * len(uniq)))
    src_t = set(uniq[:n_src]); held_t = set(uniq[n_src:])
    src = np.array([t in src_t for t in tasks]); held = ~src
    Xs, Xh = X[src], X[held]
    print(f"chunks={len(X)} tasks={len(uniq)} (src {len(src_t)} / held {len(held_t)}); "
          f"src_chunks={len(Xs)} held_chunks={len(Xh)}")

    glap = grid_laplacian_dirs(H, REAL, W)
    tot_s = (Xs - Xs.mean(0)).var(0).sum()
    print(f"{'K':>4}  {'GLAP_cov':>9} {'GLAP_transfer':>13}  {'PCA_cov':>8} {'PCA_transfer':>12}")
    for K in KS:
        Ug = glap[:K].T
        cov_g = 1 - ((Xs - Xs @ (Ug @ Ug.T)) ** 2).sum() / (((Xs - Xs.mean(0)) ** 2).sum())
        tr_g = rel_err(Xh - Xh.mean(0), Ug)
        Up = pca(Xs, K)
        cov_p = 1 - ((Xs - Xs.mean(0) - (Xs - Xs.mean(0)) @ (Up @ Up.T)) ** 2).sum() / ((Xs - Xs.mean(0)) ** 2).sum()
        tr_p = rel_err(Xh - Xh.mean(0), Up)
        print(f"{K:>4}  {cov_g:>9.3f} {tr_g:>13.3f}  {cov_p:>8.3f} {tr_p:>12.3f}")

    # within/between-task split of PCA-16 coordinate (on source tasks)
    Up = pca(Xs, 16); C = (Xs - Xs.mean(0)) @ Up
    tot = C.var(0); within = np.zeros(16); n = len(Xs)
    for t in src_t:
        m = tasks[src] == t
        if m.sum() < 2:
            continue
        within += C[m].var(0) * m.sum()
    within /= n
    print(f"PCA-16 mean within-task frac = {float(np.mean(within/(tot+1e-12))):.3f} "
          f"(grid-Lap was 0.987)")
    print("VC_PCA_CHECK_DONE=ok")


if __name__ == "__main__":
    main()
