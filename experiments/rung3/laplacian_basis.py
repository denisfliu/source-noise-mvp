"""Laplacian orthonormal bases for the pass-through pin — principled alternatives
to the periodic Fourier basis, both still orthonormal so pass-through holds exactly.

  path (DCT-II): eigenbasis of the path-graph Laplacian on H timesteps (free/
     non-periodic endpoints). A trajectory runs start->target and does NOT wrap, so
     periodic Fourier spends modes on the endpoint mismatch; the path-Laplacian low
     modes are the smoothest functions respecting free boundaries -> the coherent
     structure packs into fewer, cleaner coefficients. Per-channel (no coupling).
  grid: eigenbasis of the (time x channel) grid-graph Laplacian — temporal edges
     (weight 1) PLUS channel edges (weight w, complete graph over the C channels).
     Its low-eigenvalue modes are smooth-in-time AND smooth-across-channels, so they
     capture CROSS-CHANNEL coordination (e.g. position-orientation co-variation) that
     a per-channel basis cannot pack into few modes — the structure 6-DOF has and the
     planar tasks did not.

Flat index convention matches chunk.reshape(H*C): node (t,c) -> t*C + c.
Selection: among the basis's direction vectors, take the top-k by the SAME coherence
objective used for Fourier ((e^T Sb e)/(e^T Sw e)), so basis FAMILY is the only
difference across arms. Any subset of an orthonormal set is orthonormal -> pass-through.
"""
import numpy as np


def dct_dirs(H, C):
    """Per-channel DCT-II basis vectors (path-graph Laplacian eigenvectors),
    returned as (n_dir, D) with orthonormal rows. D = H*C, node (t,c)->t*C+c."""
    n = np.arange(H)
    dirs = []
    for c in range(C):
        for k in range(H):
            f = np.cos(np.pi * k * (2 * n + 1) / (2 * H))
            v = np.zeros((H, C)); v[:, c] = f
            v = v.reshape(H * C)
            dirs.append(v / np.linalg.norm(v))
    return np.array(dirs)


def grid_laplacian_dirs(H, C, w=0.5):
    """Eigenvectors of the (time x channel) grid-graph Laplacian, (n_dir=D, D),
    orthonormal rows, sorted smoothest-first. Temporal path edges weight 1;
    complete channel graph weight w."""
    D = H * C
    L = np.zeros((D, D))

    def idx(t, c):
        return t * C + c

    # temporal path edges (per channel)
    for c in range(C):
        for t in range(H - 1):
            i, j = idx(t, c), idx(t + 1, c)
            L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
    # channel edges (complete graph over channels, per timestep)
    for t in range(H):
        for a in range(C):
            for b in range(a + 1, C):
                i, j = idx(t, a), idx(t, b)
                L[i, i] += w; L[j, j] += w; L[i, j] -= w; L[j, i] -= w
    L = 0.5 * (L + L.T)
    evals, evecs = np.linalg.eigh(L)                # ascending (smoothest first)
    return evecs.T                                  # (D, D) rows = eigenvectors


def _select(dirs, Sb, Sw, k):
    sc = np.array([(e @ Sb @ e) / (e @ Sw @ e) for e in dirs])
    return dirs[np.argsort(-sc)[:k]].T              # (D,k), orthonormal


def basis_dct(Sb, Sw, k, H, C):
    return _select(dct_dirs(H, C), Sb, Sw, k)


def basis_gridlap(Sb, Sw, k, H, C, w=0.5):
    return _select(grid_laplacian_dirs(H, C, w), Sb, Sw, k)
