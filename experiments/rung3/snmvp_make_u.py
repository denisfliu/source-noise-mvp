"""Build the grid-Laplacian instruction subspace U for pi0's LIBERO action chunk.

The chunk is (action_horizon=50, action_dim=32) with the 7 real LIBERO action
channels in indices 0..6 and the rest padding. U is the top-K smoothest eigenvectors
of the (time x channel) grid-graph Laplacian over the 50x7 real block (temporal path
edges weight 1, complete channel graph weight w), each embedded into the 50x32 chunk
with zeros on the padding channels and flattened in the row-major (t*32+c) convention
that matches actions.reshape(b, -1). Columns are orthonormal, so the source-noise pin
passes through exactly. Saved as (D=1600, K) float32.
"""
import os, sys
import numpy as np

H = 50; AD = 32; REAL = 7; D = H * AD
K = int(os.environ.get("SNMVP_K", "16"))
W = float(os.environ.get("SNMVP_W", "0.5"))
OUT = os.environ.get("SNMVP_OUT", os.path.expanduser("~/code/source-noise-mvp/experiments/rung3/pin_U.npy"))


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
    evals, evecs = np.linalg.eigh(L)
    return evecs.T                                  # (d,d) rows = eigenvectors, smoothest first


def main():
    dirs = grid_laplacian_dirs(H, REAL, W)[:K]      # (K, H*REAL) smoothest
    U = np.zeros((D, K))
    for k in range(K):
        full = np.zeros((H, AD))
        full[:, :REAL] = dirs[k].reshape(H, REAL)
        U[:, k] = full.reshape(D)
    # verify orthonormal columns
    err = np.abs(U.T @ U - np.eye(K)).max()
    U = U.astype(np.float32)
    np.save(OUT, U)
    print(f"saved U {U.shape} to {OUT}; orthonormality max|UtU-I|={err:.2e}")


if __name__ == "__main__":
    main()
