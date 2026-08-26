"""6-DOF CROSS-EMBODIMENT TRANSFER of the task-space instruction subspace.

The sharpest generality test: freeze the pinned subspace on set-A arms, transfer
it to a held-out arm with different kinematics/workspace, relearn ONLY the flow
executor (the body-specific complement) on the held-out arm, and measure transfer.

Same coupled 6-DOF task collected on every arm (scenes seeded identically ->
paired scenes across arms, so the cross-arm covariance Sigma_body is well-defined).
Basis = grid-Laplacian (DOF-agnostic cross-channel), selected by the CROSS-ARM
coherence ratio (Sigma_scene / Sigma_body over set-A arms) — the directions that
carry task info while being arm-invariant. Prior maps scene -> the set-A invariant.

Arms on held-out body B (held-out SCENES):
  S        scratch flow on B (no pin)
  GLAP     grid-Laplacian pin (frozen on set-A) + set-A prior, relearn executor on B
  GLAP_or  same, commanded by B's OWN true c (oracle upper bound)
  F        per-channel Fourier pin (same selection) + prior (predict fragile)
  Rand     random orthonormal pin + prior (mixing control)
Plus cross-arm c-invariance of each basis (||c_B - c_setA|| / ||c_setA||).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
os.environ.setdefault("SNMVP_DS", "c1")
import basis_lab as BL                  # noqa: E402
import laplacian_basis as LB            # noqa: E402
import structure_test_pose6d_hard as HD  # noqa: E402  (scene_targets, success, consts)

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


def load(arm):
    d = np.load(os.path.join(DATA_DIR, f"{arm}.npz"))
    return d["chunks"].astype(float), d["obs"], d["success"]


def main():
    # load all arms; scenes are paired (same seed) so obs identical across arms
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

    # cross-arm covariances on SET_A (arms on axis 1, using per-arm scene means)
    bmean = {a: Xc[a].mean(axis=1) for a in ch}                 # (S,D)
    Xarms = np.stack([bmean[a] for a in SET_A], axis=1)         # (S,|A|,D)
    Sb, Sw = BL.covariances(Xarms, D)                           # Sigma_scene, Sigma_body
    invariant = Xarms.mean(axis=1)                              # (S,D) set-A shared invariant
    gensep = float(np.linalg.eigvalsh(np.linalg.solve(Sw, Sb)).max())
    bases = {"GLAP": LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W),
             "F": BL.basis_fourier(Sb, Sw, K, C),
             "Rand": BL.basis_random(K, D, 0)}
    print(f"top_gen_eig={gensep:.1f}", flush=True)

    def bs(cf, idx):
        return float(np.mean([HD.success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                              for i in range(len(idx))]))

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S)
        he = perm[:N_HELD]; tr = perm[N_HELD:]
        # set-A priors (scene -> invariant coordinate), one per basis
        priors = {nm: BL.train_prior(obs[tr], invariant[tr] @ U, seed + 10, ITERS_PRIOR, obs_dim, K)
                  for nm, U in bases.items()}
        row = {}
        # cross-arm c-invariance on held-out scenes (per basis): how far each
        # held-out arm's own c is from the set-A invariant c
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
                if nm == "GLAP":
                    c_or = bmean[b][he] @ U
                    row[f"{b}_GLAP_or"] = bs(BL.rollout(p, obs[he], U, c_or, seed, D), he)
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
    json.dump(out, open(os.path.join(HERE, "transfer6d_result.json"), "w"), indent=2)
    print("POOLED_SUCCESS:", json.dumps(pooled, indent=2))
    print("POOLED_CINV:", json.dumps(cinv, indent=2))
    print("TRANSFER6D_DONE=ok")


if __name__ == "__main__":
    main()
