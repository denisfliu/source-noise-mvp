"""Where does a LEARNED orthonormal basis beat the FIXED Fourier basis for the
pass-through pin, and does the Fourier factoring extend to a new action axis?

Generalizes learned_basis_toy.py to C channels. Same controlled comparison (only
the basis U differs across arms): A scratch, F Fourier (top-k by the coherence
objective), L learned (unrestricted top-k generalized eigvecs of (Sb,Sw)), R
random orthonormal.

Two tasks:
  waypoint2d (C=2, NON-FOURIER): reach the target passing through m LOCALIZED,
    DIAGONAL waypoints — narrow-in-time offsets (energy spread across many Fourier
    bins) pointing in a scene-specific (x,y) direction (cross-channel). This is
    Fourier's worst case: per-channel Fourier needs many bins x 2 channels per
    waypoint, so a fixed-k Fourier subspace cannot pack them; a learned basis puts
    one localized-diagonal atom per waypoint. Prediction: L > F.
  reach3d (C=3, SMOOTH): reach the target avoiding a sphere with a smooth
    over/around detour. Extends the Fourier factoring to a 3rd action axis.
    Prediction: F stays strong, L ~ F (Fourier scales to the new axis).

Point robot, canonical frame (reach along +x), positions start at origin.
"""
import json, os, sys
import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.misc.optimizers import adam

H = 20
ACT_SCALE = 5.0
HID = 128
EULER_STEPS = 20
N_SCENES = 120
N_HELD = 40
N_DEMOS = 6
ITERS_EXEC = int(os.environ.get("SNMVP_ITERS", "4000"))
ITERS_PRIOR = 2500
SEEDS = [0, 1, 2]

TASK = os.environ.get("SNMVP_TASK", "waypoint2d")
K = int(os.environ.get("SNMVP_K", "6"))
M_WP = int(os.environ.get("SNMVP_MWP", "3"))         # waypoints (waypoint2d)
N_OBST = int(os.environ.get("SNMVP_NOBST", "1"))     # spheres (reach3d)
TOL = 0.15
TOL_WP = 0.18


# ------------------------------- tasks -------------------------------------

def _endpoint_vanish(s, s_o, w):
    raw = np.exp(-((s - s_o) ** 2) / (2 * w ** 2))
    g = raw - ((1 - s) * raw[0] + s * raw[-1])
    pk = g[np.argmin(np.abs(s - s_o))]
    return g / (pk if abs(pk) > 1e-6 else 1.0)          # peaks ~1 at s_o, 0 at ends


def _wp_atoms():
    """FIXED (shared across all scenes) localized-diagonal atoms: position s_j and
    a diagonal (non-axis-aligned) direction. Only the amplitude varies per scene,
    so the instruction is LOW-RANK (rank = M_WP) but non-Fourier (localized in time
    + cross-channel), which is the learned basis's genuine best case."""
    s_all = [0.30, 0.50, 0.70, 0.40]
    th_all = np.radians([35.0, 95.0, 150.0, 65.0])
    return np.array(s_all[:M_WP]), th_all[:M_WP]


def gen_waypoint2d(n_scenes, n_demos, seed):
    """C=2. Shared localized-diagonal atoms, scene-varying signed amplitude.
    obs=[target(2), wp_xy(2)*M_WP]."""
    C = 2
    rng = np.random.default_rng(seed)
    s_j, th_j = _wp_atoms()
    dirs = np.stack([np.cos(th_j), np.sin(th_j)], axis=1)    # (M_WP,2) fixed directions
    chunks = np.zeros((n_scenes, n_demos, H, C))
    obs = np.zeros((n_scenes, 2 + 2 * M_WP))
    meta = []
    s = np.linspace(0, 1, H + 1)
    for si in range(n_scenes):
        rad = rng.uniform(1.0, 1.7)
        target = np.array([rad, 0.0])
        A = rng.uniform(-0.6, 0.6, M_WP) * rad              # SIGNED amplitude (scene-varying)
        wp = np.array([s_j[j] * target + A[j] * dirs[j] for j in range(M_WP)])
        obs[si] = np.concatenate([target, wp.reshape(-1)])
        meta.append({"target": target, "wp": wp})
        for di in range(n_demos):
            pos = np.outer(s, target)                       # straight line origin->target
            for j in range(M_WP):
                g = _endpoint_vanish(s, s_j[j], 0.05)       # LOCALIZED (narrow) bump
                amp = A[j] * (1 + rng.normal(0, 0.06))
                d = dirs[j] + rng.normal(0, 0.02, 2)
                pos += np.outer(g, amp * d)
            pos[:, 1] += 0.02 * np.sin(np.pi * rng.integers(4, 8) * s) * np.sin(np.pi * s)  # style
            pos[0] = 0.0; pos[-1] = target
            chunks[si, di] = np.diff(pos, axis=0) * ACT_SCALE
    return chunks, obs, meta, C


def success_waypoint2d(chunk_flat, ob, C):
    pos = np.cumsum(chunk_flat.reshape(H, C) / ACT_SCALE, axis=0)
    target = ob[:2]
    if np.linalg.norm(pos[-1] - target) >= TOL:
        return 0.0
    for j in range(M_WP):
        wp = ob[2 + 2 * j: 2 + 2 * j + 2]
        if np.min(np.linalg.norm(pos - wp, axis=1)) >= TOL_WP:
            return 0.0
    return 1.0


def gen_reach3d(n_scenes, n_demos, seed):
    """C=3. Sphere obstacle, smooth over/around detour in a scene-chosen (y,z)
    direction. obs=[target(3), sphere_center(3), r(1)]."""
    C = 3
    rng = np.random.default_rng(seed)
    chunks = np.zeros((n_scenes, n_demos, H, C))
    obs = np.zeros((n_scenes, 3 + 4 * N_OBST))
    meta = []
    for si in range(n_scenes):
        rad = rng.uniform(1.0, 1.7)
        target = np.array([rad, 0.0, 0.0])
        cols = [target]
        obst = []
        for _ in range(N_OBST):
            s_o = rng.uniform(0.4, 0.6)
            r = rng.uniform(0.18, 0.28)
            psi = rng.uniform(0, np.pi)                     # detour plane (y,z), coherent/scene
            perp = np.array([0.0, np.cos(psi), np.sin(psi)])
            center = s_o * target                           # sphere centered on the line
            obst.append({"s_o": s_o, "r": r, "perp": perp, "center": center})
            cols += [center, [r]]
        obs[si] = np.concatenate([np.atleast_1d(c).ravel() for c in cols])
        meta.append({"target": target, "obst": obst})
        s = np.linspace(0, 1, H + 1)
        for di in range(n_demos):
            pos = np.outer(s, target)
            for o in obst:
                g = _endpoint_vanish(s, o["s_o"], 0.14)     # SMOOTH broad bump
                amp = (o["r"] + 0.18) * (1 + rng.normal(0, 0.05))
                pos += np.outer(g, amp * o["perp"])
            pos[0] = 0.0; pos[-1] = target
            chunks[si, di] = np.diff(pos, axis=0) * ACT_SCALE
    return chunks, obs, meta, C


def success_reach3d(chunk_flat, ob, C):
    pos = np.cumsum(chunk_flat.reshape(H, C) / ACT_SCALE, axis=0)
    target = ob[:3]
    if np.linalg.norm(pos[-1] - target) >= TOL:
        return 0.0
    for j in range(N_OBST):
        base = 3 + 4 * j
        center = ob[base:base + 3]; r = ob[base + 3]
        if (np.linalg.norm(pos - center, axis=1) <= r).any():
            return 0.0
    return 1.0


TASKS = {"waypoint2d": (gen_waypoint2d, success_waypoint2d),
         "reach3d": (gen_reach3d, success_reach3d)}


# ------------------------------- bases -------------------------------------

def covariances(chunks_tr, D):
    S, N = chunks_tr.shape[:2]
    X = chunks_tr.reshape(S, N, D)
    scene_mean = X.mean(axis=1)
    Sb = np.cov(scene_mean, rowvar=False)
    Sw = np.zeros((D, D))
    for s in range(S):
        Xc = X[s] - scene_mean[s]
        Sw += Xc.T @ Xc / max(N - 1, 1)
    Sw = Sw / S + 1e-4 * np.eye(D)
    return Sb, Sw


def fourier_basis(C):
    t = np.arange(H)
    vecs = []
    for ch in range(C):
        for w in range(H // 2 + 1):
            for kind in ("cos", "sin"):
                if (w == 0 or w == H // 2) and kind == "sin":
                    continue
                wf = np.cos(2 * np.pi * w * t / H) if kind == "cos" else np.sin(2 * np.pi * w * t / H)
                v = np.zeros((H, C)); v[:, ch] = wf
                v = v.reshape(H * C)
                vecs.append(v / np.linalg.norm(v))
    return np.array(vecs)


def basis_fourier(Sb, Sw, k, C):
    F = fourier_basis(C)
    sc = np.array([(e @ Sb @ e) / (e @ Sw @ e) for e in F])
    return F[np.argsort(-sc)[:k]].T


def basis_learned(Sb, Sw, k):
    L = np.linalg.cholesky(Sw)
    Linv = np.linalg.inv(L)
    M = Linv @ Sb @ Linv.T
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    G = Linv.T @ V[:, np.argsort(-w)[:k]]
    Q, _ = np.linalg.qr(G)
    return Q[:, :k]


def basis_random(k, D, seed):
    rng = np.random.default_rng(3000 + seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, k)))
    return Q[:, :k]


# ------------------------------ executor -----------------------------------

def init_exec(rng, D, obs_dim):
    dims = [D + 1 + obs_dim, HID, HID, HID, D]
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def fwd(params, xt, t, obs):
    h = anp.concatenate([xt, t.reshape(-1, 1), obs], axis=1)
    for w, b in params[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = params[-1]
    return h @ w + b


def pin_source(eps, U, c):
    return eps + (c - eps @ U) @ U.T


def train_exec(obs_flat, X_flat, U, seed, iters, D, obs_dim):
    n = obs_flat.shape[0]

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, n, size=256)
        a = X_flat[idx]; obs = obs_flat[idx]
        eps = rng.normal(size=(256, D))
        if U is not None:
            eps = pin_source(eps, U, a @ U)
        t = rng.uniform(0, 1, size=256)
        xt = t[:, None] * eps + (1 - t[:, None]) * a
        return anp.mean((fwd(params, xt, t, obs) - (eps - a)) ** 2)

    params = init_exec(np.random.default_rng(seed), D, obs_dim)
    return adam(grad(loss), params, num_iters=iters, step_size=1e-3)


def rollout(params, obs, U, c, seed, D):
    rng = np.random.default_rng(500 + seed)
    eps = rng.normal(size=(obs.shape[0], D))
    if U is not None and c is not None:
        eps = pin_source(eps, U, c)
    x = eps; dt = 1.0 / EULER_STEPS
    for k in range(EULER_STEPS):
        t = np.full(obs.shape[0], 1.0 - k * dt)
        x = x - dt * np.asarray(fwd(params, x, t, obs))
    return x


def train_prior(obs_tr, c_tr, seed, iters, obs_dim, k):
    rng = np.random.default_rng(seed)
    M = obs_tr.shape[0]
    dims = [obs_dim, 64, 64, k]
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

def main():
    gen, succ_fn = TASKS[TASK]
    out = {}
    for seed in SEEDS:
        ch, obs, meta, C = gen(N_SCENES, N_DEMOS, seed)
        D = H * C; obs_dim = obs.shape[1]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N_SCENES)
        tr, he = perm[N_HELD:], perm[:N_HELD]
        X = ch.reshape(N_SCENES, N_DEMOS, D)
        Sb, Sw = covariances(ch[tr], D)
        bases = {"F": basis_fourier(Sb, Sw, K, C), "L": basis_learned(Sb, Sw, K),
                 "R": basis_random(K, D, seed)}
        obs_tr = np.repeat(obs[tr], N_DEMOS, axis=0)
        X_tr = X[tr].reshape(-1, D)
        scene_mean_tr = X[tr].mean(axis=1)
        he_obs = obs[he]

        def bs(chunks_flat, obss):
            return float(np.mean([succ_fn(chunks_flat[i], obss[i], C) for i in range(len(obss))]))

        row = {}
        row["ceil"] = bs(X[he].reshape(-1, D), np.repeat(he_obs, N_DEMOS, axis=0))
        pA = train_exec(obs_tr, X_tr, None, seed, ITERS_EXEC, D, obs_dim)
        row["A"] = bs(rollout(pA, he_obs, None, None, seed, D), he_obs)
        for name, U in bases.items():
            c_tr = scene_mean_tr @ U
            prior = train_prior(obs_tr, np.repeat(c_tr, N_DEMOS, axis=0), seed + 10,
                                ITERS_PRIOR, obs_dim, K)
            p = train_exec(obs_tr, X_tr, U, seed, ITERS_EXEC, D, obs_dim)
            row[name] = bs(rollout(p, he_obs, U, prior(he_obs), seed, D), he_obs)
        out[f"s{seed}"] = {k: round(v, 3) for k, v in row.items()}
        print(f"{TASK} K={K} seed{seed}: {out[f's{seed}']}", flush=True)

    pooled = {k: round(float(np.mean([out[f's{s}'][k] for s in SEEDS])), 3)
              for k in ["A", "F", "L", "R", "ceil"]}
    res = {"task": TASK, "K": K, "M_WP": M_WP, "N_OBST": N_OBST,
           "per_seed": out, "pooled": pooled}
    here = os.path.dirname(os.path.abspath(__file__))
    tag = f"{TASK}_K{K}"
    json.dump(res, open(os.path.join(here, f"basis_lab_{tag}.json"), "w"), indent=2)
    print(f"{TASK} K={K} POOLED: {pooled}", flush=True)
    print("BASIS_LAB_DONE=ok")


if __name__ == "__main__":
    main()
