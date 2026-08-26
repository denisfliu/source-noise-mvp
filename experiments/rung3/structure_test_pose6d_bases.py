"""6-DOF basis comparison in the BOTTLENECKED regime (the only regime where the
pin helps — see pose6d_diag: at high capacity the task is obs-solvable and the pin
hurts). Data bottleneck (few training scenes) at moderate capacity. Compares the
orthonormal basis FAMILY for the pass-through pin:
  A     scratch (no pin)
  F     Fourier (periodic Laplacian eigenbasis)
  DCT   path-graph Laplacian eigenbasis (free endpoints; DCT-II), per-channel
  GLAP  (time x channel) grid-graph Laplacian eigenbasis (cross-channel coupling)
  R     random orthonormal (control)
Each pinned basis is run with PRIOR (deployable) and ORACLE (true held-out coeff)
commands; per-basis prior-prediction error reported. All bases select their top-K
directions by the SAME coherence objective (Sb/Sw), so basis family is the only
difference. Prediction: boundary/cross-channel-respecting Laplacian bases pack the
structure into fewer, more prior-predictable coefficients -> beat Fourier here.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import basis_lab as BL                  # noqa: E402
import laplacian_basis as LB            # noqa: E402
import structure_test_pose6d as ST      # noqa: E402

BL.H = ST.H
H, C, D = ST.H, ST.C, ST.D
K = int(os.environ.get("SNMVP_K", "10"))
HID = int(os.environ.get("SNMVP_HID", "256"))
ITERS = int(os.environ.get("SNMVP_ITERS", "12000"))
ITERS_PRIOR = 3000
N_TRAIN = int(os.environ.get("SNMVP_NTRAIN", "40"))    # DATA bottleneck
N_HELD = 30
GLAP_W = float(os.environ.get("SNMVP_GLAPW", "0.5"))
SEEDS = [0, 1, 2]


def main():
    BL.HID = HID
    d = np.load(ST.DATA)
    chunks, obs, succ = d["chunks"].astype(float), d["obs"], d["success"]
    S, N = chunks.shape[:2]
    scale = 1.0 / np.abs(chunks).mean()
    ch_s = chunks * scale
    tgt, obst, r, aa = ST.canon_pos_from_obs(obs)
    X = ch_s.reshape(S, N, D)
    obs_dim = obs.shape[1]
    print(f"{ST.ARM}: demo ceiling {succ.mean():.3f}; HID={HID} ITERS={ITERS} "
          f"N_TRAIN={N_TRAIN} K={K} GLAP_W={GLAP_W}")

    def bs(cf, idx):
        return float(np.mean([ST.success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                              for i in range(len(idx))]))

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S)
        he = perm[:N_HELD]; tr = perm[N_HELD:N_HELD + N_TRAIN]        # data bottleneck
        Sb, Sw = BL.covariances(ch_s[tr], D)
        bases = {"F": BL.basis_fourier(Sb, Sw, K, C),
                 "DCT": LB.basis_dct(Sb, Sw, K, H, C),
                 "GLAP": LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W),
                 "R": BL.basis_random(K, D, seed)}
        obs_tr = np.repeat(obs[tr], N, axis=0); X_tr = X[tr].reshape(-1, D)
        scene_mean_tr = X[tr].mean(axis=1); scene_mean_he = X[he].mean(axis=1)
        he_obs = obs[he]

        row = {"ceil": float(succ[he].mean())}
        pA = BL.train_exec(obs_tr, X_tr, None, seed, ITERS, D, obs_dim)
        row["A"] = bs(BL.rollout(pA, he_obs, None, None, seed, D), he)
        for name, U in bases.items():
            c_tr = scene_mean_tr @ U
            c_or = scene_mean_he @ U
            prior = BL.train_prior(obs_tr, np.repeat(c_tr, N, axis=0), seed + 10,
                                   ITERS_PRIOR, obs_dim, K)
            p = BL.train_exec(obs_tr, X_tr, U, seed, ITERS, D, obs_dim)
            c_pred = prior(he_obs)
            row[name] = bs(BL.rollout(p, he_obs, U, c_pred, seed, D), he)
            row[f"{name}_or"] = bs(BL.rollout(p, he_obs, U, c_or, seed, D), he)
            row[f"{name}_perr"] = round(float(np.linalg.norm(c_pred - c_or, axis=1).mean()
                                              / (np.linalg.norm(c_or, axis=1).mean() + 1e-9)), 3)
        results[f"s{seed}"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
        print(f"seed{seed}: {results[f's{seed}']}", flush=True)

    keys = list(results["s0"].keys())
    pooled = {k: round(float(np.mean([results[f's{s}'][k] for s in SEEDS])), 3) for k in keys}
    out = {"config": {"HID": HID, "ITERS": ITERS, "N_TRAIN": N_TRAIN, "K": K, "GLAP_W": GLAP_W},
           "per_seed": results, "pooled": pooled}
    json.dump(out, open(os.path.join(HERE, "pose6d_bases_result.json"), "w"), indent=2)
    print("POOLED:", json.dumps(pooled, indent=2))
    print("POSE6D_BASES_DONE=ok")


if __name__ == "__main__":
    main()
