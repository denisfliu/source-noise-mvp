"""Shared helpers for the PCA source-noise pin at inference.

The pin sets the U-subspace coordinate of the flow source noise to a commanded value
c and leaves the orthogonal complement Gaussian: noise = (I - U U^T) g + U c, in the
model's normalized action space (flattened H*action_dim). A separate prior maps the
observation's proprioceptive state to c, since c must be supplied online where the
action is not available. The prior here is ridge regression on the normalized state,
which the gate check found predicts the leading (high-variance) PCA coordinates well.
"""
import numpy as np


def load_U(path):
    return np.load(path).astype(np.float32)                      # (D, K), orthonormal cols


def build_pca_noise(c, U, rng, H, ad):
    """c: (K,) or (b,K); U: (D=H*ad, K). Returns noise (b,H,ad) [or (H,ad) if c is 1-D]
    whose projection onto U equals c and whose complement is standard Gaussian."""
    single = (np.ndim(c) == 1)
    C = np.atleast_2d(c).astype(np.float32)                      # (b,K)
    b = C.shape[0]; D = U.shape[0]
    g = rng.standard_normal((b, D)).astype(np.float32)
    pinned = g - (g @ U) @ U.T + C @ U.T                         # (b,D)
    out = pinned.reshape(b, H, ad)
    return out[0] if single else out


def fit_state_prior(S, C, lam=1e-2):
    """Ridge regression c ~ state. S:(n,ds), C:(n,K). Returns (W,b) with c = S@W + b."""
    n = S.shape[0]
    Sa = np.concatenate([S, np.ones((n, 1))], 1).astype(np.float64)
    A = Sa.T @ Sa + lam * np.eye(Sa.shape[1])
    Wfull = np.linalg.solve(A, Sa.T @ C.astype(np.float64))      # (ds+1, K)
    return Wfull[:-1].astype(np.float32), Wfull[-1].astype(np.float32)


def apply_prior(W, b, S):
    return (S.astype(np.float32) @ W) + b                        # (n,K)
