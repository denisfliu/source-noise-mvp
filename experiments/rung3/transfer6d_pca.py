"""#1: PCA-pin vs grid-Laplacian-pin CROSS-EMBODIMENT TRANSFER SUCCESS.

Extends transfer6d.py with a data-fit PCA basis estimated on the set-A arms (the
E2b diagnostic winner) as another frozen pinned subspace, to test whether its
offline reconstruction advantage over the fixed grid-Laplacian converts into
transfer SUCCESS + steerability on a held-out arm.

Bases (all orthonormal -> pass-through holds; all frozen on set-A, prior on set-A,
executor relearned on held-out arm B):
  S        scratch flow on B (no pin)
  GLAP     grid-Laplacian pin (cross-arm coherence selection)
  PCA      top-K principal directions of pooled set-A chunks (variance-fit)
  F        per-channel Fourier pin
  Rand     random orthonormal pin
  *_or     commanded by B's OWN true c (oracle upper bound), for GLAP and PCA
Plus cross-arm c-invariance per basis, and a steering-slope probe on B: sweep the
command along the empirical side-direction and regress realized detour vs command.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
os.environ.setdefault("SNMVP_DS", "c1")
import basis_lab as BL                  # noqa: E402
import laplacian_basis as LB            # noqa: E402
import structure_test_pose6d_hard as HD  # noqa: E402

H, C, D = HD.H, HD.C, HD.D
BL.H = H
BL.HID = 256
K = int(os.environ.get("SNMVP_K", "12"))
ITERS = int(os.environ.get("SNMVP_ITERS", "10000"))
ITERS_PRIOR = 3000
GLAP_W = 0.5
SET_A = os.environ.get("SNMVP_SETA", "Panda,IIWA,UR5e").split(",")
HELDOUT = os.environ.get("SNMVP_HELD", "Jaco").split(",")
N_HELD = 40
SEEDS = [0, 1, 2]
DATA_DIR = os.path.join(HERE, "data_pose6d_hard")
OUT = os.environ.get("SNMVP_OUT", "transfer6d_pca_result.json")


def load(arm):
    d = np.load(os.path.join(DATA_DIR, f"{arm}.npz"))
    return d["chunks"].astype(float), d["obs"], d["success"]


def pca_setA(Xc, arms, k):
    """Top-k principal directions of the pooled set-A chunks (scaled space).
    Orthonormal columns (D,k) -> pass-through pin still exact. Fit on set-A only
    (held-out arm never seen), so this is a fair transfer basis."""
    Xf = np.concatenate([Xc[a].reshape(-1, D) for a in arms], axis=0)
    _, _, Vt = np.linalg.svd(Xf - Xf.mean(axis=0), full_matrices=False)
    return Vt[:k].T


def main():
    ch = {}; obs = None; succ = {}
    for arm in SET_A + HELDOUT:
        c, o, s = load(arm)
        ch[arm] = c; succ[arm] = s
        obs = o if obs is None else obs
    S, N = ch[SET_A[0]].shape[:2]
    scale = 1.0 / np.mean([np.abs(ch[a]).mean() for a in ch])
    Xc = {a: (ch[a] * scale).reshape(S, N, D) for a in ch}
    tgt, obst, r, aa = HD.scene_targets(obs)
    obs_dim = obs.shape[1]
    print(f"arms set_A={SET_A} held={HELDOUT}; S={S} N={N} D={D} K={K} scale={scale:.1f}")
    for a in ch:
        print(f"  {a}: demo ceiling {succ[a].mean():.3f}")

    bmean = {a: Xc[a].mean(axis=1) for a in ch}
    Xarms = np.stack([bmean[a] for a in SET_A], axis=1)
    Sb, Sw = BL.covariances(Xarms, D)
    invariant = Xarms.mean(axis=1)
    gensep = float(np.linalg.eigvalsh(np.linalg.solve(Sw, Sb)).max())
    bases = {"GLAP": LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W),
             "PCA": pca_setA(Xc, SET_A, K),
             "F": BL.basis_fourier(Sb, Sw, K, C),
             "Rand": BL.basis_random(K, D, 0)}
    print(f"top_gen_eig={gensep:.1f}", flush=True)

    def bs(cf, idx):
        return float(np.mean([HD.success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                              for i in range(len(idx))]))

    def steer_slope(p, U, ob_he, he):
        """Pass-through steering probe on the held-out arm (executor relearned for
        this basis). Steer command along the dominant instruction direction d (top
        PC of the set-A coordinate), regress the REALIZED coordinate projection
        (U^T a)·d against the commanded magnitude. Slope ~1 = pass-through survives
        the relearned executor for this basis. Also report the realized lateral
        detour slope (behavior actually moves)."""
        cA = invariant[he] @ U                                   # (n,K)
        _, _, Vt = np.linalg.svd(cA - cA.mean(0), full_matrices=False)
        d = Vt[0]                                                # unit instruction dir in c-space
        base = cA.mean(0)
        mags = np.array([-2., -1., 0., 1., 2.])
        real_c, real_lat = [], []
        for m in mags:
            cmd = np.repeat((base + m * d)[None], len(he), axis=0)
            cf = BL.rollout(p, ob_he, U, cmd, 0, D)              # (n,D)
            real_c.append(float(((cf @ U) @ d).mean()))          # realized coordinate along d
            real_lat.append(float(cf.reshape(len(he), H, C)[:, :, 1].mean()))  # lateral detour
        A = np.vstack([mags, np.ones_like(mags)]).T
        pt = float(np.linalg.lstsq(A, np.array(real_c), rcond=None)[0][0])
        beh = float(np.linalg.lstsq(A, np.array(real_lat), rcond=None)[0][0])
        return {"passthrough": round(pt, 3), "lateral": round(beh, 3)}

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S)
        he = perm[:N_HELD]; tr = perm[N_HELD:]
        priors = {nm: BL.train_prior(obs[tr], invariant[tr] @ U, seed + 10, ITERS_PRIOR, obs_dim, K)
                  for nm, U in bases.items()}
        row = {}
        for nm, U in bases.items():
            cA = invariant[he] @ U
            row[f"cinv_{nm}"] = {b: round(float(np.linalg.norm(bmean[b][he] @ U - cA, axis=1).mean()
                                / (np.linalg.norm(cA, axis=1).mean() + 1e-9)), 3) for b in HELDOUT}
        for b in HELDOUT:
            ob_tr = np.repeat(obs[tr], N, axis=0); X_tr = Xc[b][tr].reshape(-1, D)
            pS = BL.train_exec(ob_tr, X_tr, None, seed, ITERS, D, obs_dim)
            row[f"{b}_S"] = bs(BL.rollout(pS, obs[he], None, None, seed, D), he)
            for nm, U in bases.items():
                p = BL.train_exec(ob_tr, X_tr, U, seed, ITERS, D, obs_dim)
                cmd = priors[nm](obs[he])
                row[f"{b}_{nm}"] = bs(BL.rollout(p, obs[he], U, cmd, seed, D), he)
                if nm in ("GLAP", "PCA"):
                    c_or = bmean[b][he] @ U
                    row[f"{b}_{nm}_or"] = bs(BL.rollout(p, obs[he], U, c_or, seed, D), he)
                    st = steer_slope(p, U, obs[he], he)
                    row[f"{b}_{nm}_steerPT"] = st["passthrough"]
                    row[f"{b}_{nm}_steerLAT"] = st["lateral"]
        results[f"s{seed}"] = row
        flat = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
        print(f"seed{seed}: {json.dumps(flat)}", flush=True)

    scalar_keys = [k for k in results["s0"] if isinstance(results["s0"][k], float)]
    pooled = {k: round(float(np.mean([results[f's{s}'][k] for s in SEEDS])), 3) for k in scalar_keys}
    cinv = {nm: {b: round(float(np.mean([results[f's{s}'][f'cinv_{nm}'][b] for s in SEEDS])), 3)
                 for b in HELDOUT} for nm in bases}
    out = {"config": {"K": K, "ITERS": ITERS, "SET_A": SET_A, "HELDOUT": HELDOUT,
                      "top_gen_eig": round(gensep, 1)},
           "ceilings": {a: round(float(succ[a].mean()), 3) for a in ch},
           "pooled_success": pooled, "pooled_cinvariance": cinv, "per_seed": results}
    json.dump(out, open(os.path.join(HERE, OUT), "w"), indent=2)
    print("POOLED:", json.dumps(pooled, indent=2))
    print("POOLED_CINV:", json.dumps(cinv, indent=2))
    print("TRANSFER6D_PCA_DONE=ok")


if __name__ == "__main__":
    main()
