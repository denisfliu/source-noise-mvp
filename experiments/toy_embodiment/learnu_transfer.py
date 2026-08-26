"""Learn the orthonormal basis U for CROSS-EMBODIMENT TRANSFER (not single-body
success), and test whether a learned U transfers better than fixed Fourier to
held-out bodies with different workspaces.

Objective (body = nuisance): on set-A bodies, U = top-k generalized eigenvectors
of (Sigma_scene, Sigma_body), orthonormalized —
  Sigma_scene = between-scene covariance of the across-body mean chunk (instruction
                signal; want high),
  Sigma_body  = mean over scenes of the across-body covariance at fixed scene
                (body-caused variation; want low).
So U's coordinates c=Uᵀa are maximally scene-informative and body-invariant; the
body-specific realization (incl. coupling) is pushed into the complement.

Transfer: freeze U on set A; train a scene->c prior on the set-A invariant; on a
held-out body B relearn ONLY the flow executor (complement) with U pinned, command
from the prior. Compare held-out-scene success on B:
  S        scratch flow on B (no pin)
  Ulearn   learned-U pin + set-A prior
  F        Fourier-U (best bins by the SAME Sigma_scene/Sigma_body ratio) + prior
  Urand    random orthonormal U + prior
  Ulearn_o learned-U pin + B's OWN true c (oracle upper bound)
Also reports c-invariance across bodies (||c_B - c_setA|| / ||c_setA|| on held-out
scenes) per basis — the direct diagnostic of whether U is body-invariant.

Everything in the CANONICAL task frame (reach -> +x); generated chunks rotated back
to world for the mb_dataset obstacle-detour success check.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
import basis_lab as BL                  # noqa: E402
import mb_dataset as mb                 # noqa: E402
import dataset as tf                    # noqa: E402
from embodiments import make_bodies     # noqa: E402

H = mb.H                                 # 20
C = 2
D = H * C
BL.H = H
BL.HID = 128
ACT = mb.ACT_SCALE
K = int(os.environ.get("SNMVP_K", "6"))
ITERS = int(os.environ.get("SNMVP_ITERS", "3000"))
ITERS_PRIOR = 2500
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "200"))
N_DEMOS = 6
N_HELD = 60
SET_A = ["arm2", "arm3", "arm4", "arm5"]         # arm family (reach 1.8-2.5)
HELDOUT = ["point", "point_drag", "arm_short"]   # point = drone analog (unconstrained)
SEEDS = [0, 1, 2]


def canon_obs(scenes):
    return np.array([[s["radius"], s["s_o"], s["lateral"], s["obst_r"]] for s in scenes])


def main():
    bodies = make_bodies()
    use = {b: bodies[b] for b in SET_A + HELDOUT}
    rng = np.random.default_rng(0)
    scenes, _, angles, chunks = mb.make_dataset(use, N_SCENES, N_DEMOS, rng)
    obs_c = canon_obs(scenes)                       # (S,4) canonical scene descriptor
    obs_dim = obs_c.shape[1]
    S = N_SCENES
    # canonical chunks per body: (S,N,H,2)
    canon = {b: tf.to_canonical(chunks[b], angles[:, None]) for b in use}
    bmean = {b: canon[b].mean(axis=1).reshape(S, D) for b in use}    # body-mean over demos

    # cross-body covariances on SET_A (bodies on axis 1)
    Xbodies = np.stack([bmean[b] for b in SET_A], axis=1)            # (S,|A|,D)
    Sb, Sw = BL.covariances(Xbodies, D)             # Sigma_scene, Sigma_body
    invariant = Xbodies.mean(axis=1)                # (S,D) set-A shared invariant
    gensep = float(np.linalg.eigvalsh(np.linalg.solve(Sw, Sb)).max())  # top gen-eigval
    print(f"D={D} K={K} set_A={SET_A} heldout={HELDOUT} top_gen_eig={gensep:.2f}", flush=True)

    def succ_world(gen_canon_scaled, idx):
        """gen (n,D) canonical scaled -> rotate to world -> mb success."""
        g = gen_canon_scaled.reshape(-1, H, C)
        world = tf.to_canonical(g, -angles[idx])                    # canonical->world (1D angles: g is 3D)
        return float(np.mean([mb.success(scenes[idx[i]], world[i]) for i in range(len(idx))]))

    results = {}
    for seed in SEEDS:
        r = np.random.default_rng(seed)
        perm = r.permutation(S)
        he = perm[:N_HELD]; tr = perm[N_HELD:]
        bases = {"Ulearn": BL.basis_learned(Sb, Sw, K),
                 "F": BL.basis_fourier(Sb, Sw, K, C),
                 "Urand": BL.basis_random(K, D, seed)}
        # scene->c priors (set-A invariant), one per basis
        priors = {}
        for nm, U in bases.items():
            c_tr = invariant[tr] @ U
            priors[nm] = BL.train_prior(obs_c[tr], c_tr, seed + 10, ITERS_PRIOR, obs_dim, K)

        row = {}
        # c-invariance diagnostic (held-out scenes): how far is each body's own c
        # from the set-A invariant c, per basis (lower = more body-invariant)
        for nm, U in bases.items():
            cA = invariant[he] @ U
            inv_err = {}
            for b in HELDOUT:
                cB = bmean[b][he] @ U
                inv_err[b] = float(np.linalg.norm(cB - cA, axis=1).mean()
                                   / (np.linalg.norm(cA, axis=1).mean() + 1e-9))
            row[f"cinv_{nm}"] = {b: round(v, 3) for b, v in inv_err.items()}

        for b in HELDOUT:
            ob_tr = np.repeat(obs_c[tr], N_DEMOS, axis=0)
            X_tr = canon[b][tr].reshape(-1, D) * 1.0        # already *ACT
            # scratch on B
            pS = BL.train_exec(ob_tr, X_tr, None, seed, ITERS, D, obs_dim)
            row[f"{b}_S"] = succ_world(BL.rollout(pS, obs_c[he], None, None, seed, D), he)
            for nm, U in bases.items():
                p = BL.train_exec(ob_tr, X_tr, U, seed, ITERS, D, obs_dim)
                cmd = priors[nm](obs_c[he])
                row[f"{b}_{nm}"] = succ_world(BL.rollout(p, obs_c[he], U, cmd, seed, D), he)
                if nm == "Ulearn":
                    c_or = bmean[b][he] @ U             # B's own true c
                    row[f"{b}_Ulearn_o"] = succ_world(BL.rollout(p, obs_c[he], U, c_or, seed, D), he)
        results[f"s{seed}"] = row
        flat = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
        print(f"seed{seed}: {json.dumps(flat)}", flush=True)

    # pool scalar keys
    scalar_keys = [k for k in results["s0"] if isinstance(results["s0"][k], float)]
    pooled = {k: round(float(np.mean([results[f's{s}'][k] for s in SEEDS])), 3) for k in scalar_keys}
    # pool c-invariance
    cinv = {}
    for nm in ["Ulearn", "F", "Urand"]:
        cinv[nm] = {b: round(float(np.mean([results[f's{s}'][f'cinv_{nm}'][b] for s in SEEDS])), 3)
                    for b in HELDOUT}
    out = {"config": {"K": K, "ITERS": ITERS, "N_SCENES": N_SCENES, "SET_A": SET_A,
                      "HELDOUT": HELDOUT, "top_gen_eig": round(gensep, 2)},
           "pooled_success": pooled, "pooled_cinvariance": cinv, "per_seed": results}
    json.dump(out, open(os.path.join(HERE, "learnu_transfer_result.json"), "w"), indent=2)
    print("POOLED_SUCCESS:", json.dumps(pooled, indent=2))
    print("POOLED_CINV:", json.dumps(cinv, indent=2))
    print("LEARNU_TRANSFER_DONE=ok")


if __name__ == "__main__":
    main()
