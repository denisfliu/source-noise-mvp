"""Generalize the FIXED Fourier factoring to a LEARNED orthonormal basis.

The pass-through pin works for ANY orthonormal transform T of the action chunk
(orthonormal T preserves both the linear flow path and the isotropic Gaussian
source, so clamping a T-subspace gives zero velocity there -> the output carries
the command exactly). Fourier is one such T; it wins only when the cross-demo-
coherent, low-variance-across-realization directions happen to align with low
Fourier modes. On richer (multi-obstacle) tasks that energy spreads across bins
and the Fourier pin degrades. This tests whether a data-driven orthonormal basis
recovers a clean low-dimensional pass-through subspace where Fourier cannot.

Controlled comparison — the ONLY thing that differs across arms is the basis U
(same general-subspace pin, same prior, same executor):
  A  scratch (no pin)
  F  Fourier basis, top-k directions by the coherence objective
  L  learned orthonormal basis: unrestricted top-k maximizers of that objective
     (generalized eigenvectors of (Sigma_between, Sigma_within), orthonormalized)
  R  random orthonormal basis (control)

Objective (the linear rendering of the phase-coherence criterion): a direction e
scores S(e) = (e^T Sigma_b e)/(e^T Sigma_w e) — high between-scene variance
(carries scene-specific instruction) and low within-scene variance (agrees across
demos of the same scene = coherent). L is the unrestricted argmax subspace; F is
the best Fourier-restricted subspace under the same S.

Point-robot, canonical frame (reach fixed to +x so the basis question is isolated
from reach rotation, which is an identical orthogonal factor for all arms).
"""
import json, os, sys
import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.misc.optimizers import adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_dataset_hard as mb            # noqa: E402

H = mb.H
D = 2 * H
ACT_SCALE = mb.ACT_SCALE
OBS_DIM = mb.OBS_DIM
HID = 128
EULER_STEPS = 20
N_SCENES = 100
N_HELD = 20
N_DEMOS = 6
K = 8                                   # pinned-subspace dimension (same for F/L/R)
ITERS_EXEC = 2500
ITERS_PRIOR = 2000
SEEDS = [0, 1, 2]
N_OBST = [1, 2, 3]
TOL = mb.SUCCESS_TOL


# --------------------------------- data ------------------------------------

def gen_data(n_obst, n_scenes, n_demos, seed):
    rng = np.random.default_rng(seed)
    chunks = np.zeros((n_scenes, n_demos, H, 2))
    obsv = np.zeros((n_scenes, OBS_DIM))
    for si in range(n_scenes):
        sc = mb.make_scene(rng, n_obst)
        sc["angle"] = 0.0                                # canonical: reach along +x
        sc["target"] = sc["radius"] * np.array([1.0, 0.0])
        for di in range(n_demos):
            chunks[si, di] = mb.make_demo(sc, rng)       # (H,2) canonical, *ACT_SCALE
        obsv[si] = mb.scene_obs(sc)
    return chunks, obsv


def success(chunk_flat, obs):
    """chunk_flat (D,) scaled canonical deltas; obs (OBS_DIM,). reach + clear."""
    pos = np.cumsum(chunk_flat.reshape(H, 2) / ACT_SCALE, axis=0)     # (H,2)
    target = obs[:2]
    if np.linalg.norm(pos[-1] - target) >= TOL:
        return 0.0
    for j in range(mb.MAX_OBST):
        cx, cy, r = obs[2 + 3 * j: 2 + 3 * j + 3]
        if r <= 0:
            continue
        if (np.linalg.norm(pos - np.array([cx, cy]), axis=1) <= r).any():
            return 0.0
    return 1.0


def batch_success(chunks_flat, obs):
    return float(np.mean([success(chunks_flat[i], obs[i]) for i in range(len(obs))]))


# ------------------------------- bases -------------------------------------

def covariances(chunks_tr):
    """Sigma_b (between-scene) and Sigma_w (within-scene), both (D,D)."""
    S, N = chunks_tr.shape[:2]
    X = chunks_tr.reshape(S, N, D)
    scene_mean = X.mean(axis=1)                          # (S,D)
    Sb = np.cov(scene_mean, rowvar=False)                # (D,D)
    Sw = np.zeros((D, D))
    for s in range(S):
        Xc = X[s] - scene_mean[s]
        Sw += Xc.T @ Xc / max(N - 1, 1)
    Sw /= S
    Sw += 1e-4 * np.eye(D)                               # ridge (numerical)
    return Sb, Sw


def fourier_basis():
    """Orthonormal Fourier direction vectors in R^D (per channel x/y, per freq)."""
    t = np.arange(H)
    vecs = []
    for ch in (0, 1):
        for w in range(H // 2 + 1):
            for kind in ("cos", "sin"):
                if w == 0 and kind == "sin":
                    continue
                if w == H // 2 and kind == "sin":
                    continue
                wf = np.cos(2 * np.pi * w * t / H) if kind == "cos" else np.sin(2 * np.pi * w * t / H)
                v = np.zeros((H, 2)); v[:, ch] = wf
                v = v.reshape(D)
                vecs.append(v / np.linalg.norm(v))
    return np.array(vecs)                                # (n_dir, D), orthonormal


def score(e, Sb, Sw):
    return float((e @ Sb @ e) / (e @ Sw @ e))


def basis_fourier(Sb, Sw, k):
    F = fourier_basis()
    sc = np.array([score(e, Sb, Sw) for e in F])
    top = F[np.argsort(-sc)[:k]]                          # already orthonormal
    return top.T                                          # (D,k)


def basis_learned(Sb, Sw, k):
    """Top-k generalized eigenvectors of (Sb, Sw), then orthonormalize (QR)."""
    L = np.linalg.cholesky(Sw)
    Linv = np.linalg.inv(L)
    M = Linv @ Sb @ Linv.T
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)                              # ascending
    order = np.argsort(-w)[:k]
    G = Linv.T @ V[:, order]                              # (D,k) generalized eigvecs
    Q, _ = np.linalg.qr(G)                                # orthonormal span
    return Q[:, :k]


def basis_random(k, seed):
    rng = np.random.default_rng(3000 + seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, k)))
    return Q[:, :k]


# ------------------------------ executor -----------------------------------

def init_exec(rng):
    dims = [D + 1 + OBS_DIM, HID, HID, HID, D]
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def fwd(params, xt, t, obs):
    h = anp.concatenate([xt, t.reshape(-1, 1), obs], axis=1)
    for w, b in params[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = params[-1]
    return h @ w + b


def pin_source(eps, U, c):
    """Replace the U-projection of eps with commands c (B,k). eps,out (B,D)."""
    return eps + (c - eps @ U) @ U.T


def train_exec(obs_flat, X_flat, U, seed, iters):
    n = obs_flat.shape[0]

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, n, size=256)
        a = X_flat[idx]
        obs = obs_flat[idx]
        eps = rng.normal(size=(256, D))
        if U is not None:
            eps = pin_source(eps, U, a @ U)              # teacher-forced command = a@U
        t = rng.uniform(0, 1, size=256)
        xt = t[:, None] * eps + (1 - t[:, None]) * a
        v = eps - a
        return anp.mean((fwd(params, xt, t, obs) - v) ** 2)

    params = init_exec(np.random.default_rng(seed))
    return adam(grad(loss), params, num_iters=iters, step_size=1e-3)


def rollout(params, obs, U, c, seed):
    rng = np.random.default_rng(500 + seed)
    n = obs.shape[0]
    eps = rng.normal(size=(n, D))
    if U is not None and c is not None:
        eps = pin_source(eps, U, c)
    x = eps
    dt = 1.0 / EULER_STEPS
    for k in range(EULER_STEPS):
        t = np.full(n, 1.0 - k * dt)
        x = x - dt * np.asarray(fwd(params, x, t, obs))
    return x


# -------------------------------- prior ------------------------------------

def train_prior(obs_tr, c_tr, seed, iters):
    rng = np.random.default_rng(seed)
    M = obs_tr.shape[0]
    dims = [OBS_DIM, 64, 64, K]
    params = [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
              for a, b in zip(dims[:-1], dims[1:])]

    def net(p, x):
        h = x
        for w, b in p[:-1]:
            h = anp.maximum(0.0, h @ w + b)
        w, b = p[-1]
        return h @ w + b

    def loss(p, it):
        r = np.random.default_rng(it)
        idx = r.integers(0, M, size=min(128, M))
        return anp.mean((net(p, obs_tr[idx]) - c_tr[idx]) ** 2)

    params = adam(grad(loss), params, num_iters=iters, step_size=1e-3)
    return lambda x: np.asarray(net(params, x))


# --------------------------------- run -------------------------------------

def run_level(n_obst):
    out = {}
    for seed in SEEDS:
        ch, obs = gen_data(n_obst, N_SCENES, N_DEMOS, seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N_SCENES)
        tr, he = perm[N_HELD:], perm[:N_HELD]
        X = ch.reshape(N_SCENES, N_DEMOS, D)
        Sb, Sw = covariances(ch[tr])
        bases = {"F": basis_fourier(Sb, Sw, K), "L": basis_learned(Sb, Sw, K),
                 "R": basis_random(K, seed)}

        # flattened training pairs (obs repeated per demo)
        obs_tr = np.repeat(obs[tr], N_DEMOS, axis=0)
        X_tr = X[tr].reshape(-1, D)
        # per-scene instruction command = scene-mean projected onto U
        scene_mean_tr = X[tr].mean(axis=1)               # (|tr|,D)

        he_obs = obs[he]
        row = {}
        # demo ceiling on held-out scenes
        ceil = batch_success(X[he].reshape(-1, D), np.repeat(he_obs, N_DEMOS, axis=0))

        # A: scratch
        pA = train_exec(obs_tr, X_tr, None, seed, ITERS_EXEC)
        chA = rollout(pA, he_obs, None, None, seed)
        row["A"] = batch_success(chA, he_obs)

        for name, U in bases.items():
            c_tr = scene_mean_tr @ U                      # (|tr|,K) commands
            prior = train_prior(obs_tr, np.repeat(c_tr, N_DEMOS, axis=0), seed + 10, ITERS_PRIOR)
            p = train_exec(obs_tr, X_tr, U, seed, ITERS_EXEC)
            c_he = prior(he_obs)
            chE = rollout(p, he_obs, U, c_he, seed)
            row[name] = batch_success(chE, he_obs)
        row["ceil"] = ceil
        out[f"s{seed}"] = {k: round(v, 3) for k, v in row.items()}
        print(f"n_obst={n_obst} seed{seed}: {out[f's{seed}']}", flush=True)

    pooled = {k: round(float(np.mean([out[f's{s}'][k] for s in SEEDS])), 3)
              for k in ["A", "F", "L", "R", "ceil"]}
    print(f"n_obst={n_obst} POOLED: {pooled}", flush=True)
    return {"per_seed": out, "pooled": pooled}


def main():
    results = {}
    for n in N_OBST:
        results[f"n{n}"] = run_level(n)
    summary = {f"n{n}": results[f"n{n}"]["pooled"] for n in N_OBST}
    out = {"config": {"K": K, "N_SCENES": N_SCENES, "N_DEMOS": N_DEMOS,
                      "ITERS_EXEC": ITERS_EXEC, "H": H, "D": D},
           "results": results, "summary": summary}
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, "learned_basis_result.json"), "w"), indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2))
    print("LEARNED_BASIS_DONE=ok")


if __name__ == "__main__":
    main()
