"""Decisive mechanistic control for the grid-Laplacian win: is CROSS-CHANNEL
coupling the mechanism? Sweep the channel-edge weight w of the (time x channel)
grid Laplacian. At w=0 there are no channel edges -> per-channel path-Laplacian
(≈DCT), which the 3-seed comparison showed does NOT beat scratch. If the pin
benefit appears only for w>0 and tracks w, cross-channel coupling is confirmed as
the source of the 6-DOF advantage. Baseline A (scratch, w-independent) trained
once per seed. Tighter eval (N_HELD=50) + more seeds than the screen.
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
K = 10
HID = 256
BL.HID = HID
ITERS = 10000
ITERS_PRIOR = 3000
N_TRAIN = 30
N_HELD = 50
WS = [0.0, 0.25, 0.5, 1.0, 2.0]
SEEDS = [0, 1, 2, 3]


def main():
    d = np.load(ST.DATA)
    chunks, obs, succ = d["chunks"].astype(float), d["obs"], d["success"]
    S, N = chunks.shape[:2]
    scale = 1.0 / np.abs(chunks).mean()
    ch_s = chunks * scale
    tgt, obst, r, aa = ST.canon_pos_from_obs(obs)
    X = ch_s.reshape(S, N, D)
    obs_dim = obs.shape[1]
    print(f"{ST.ARM}: ceiling {succ.mean():.3f}; HID={HID} ITERS={ITERS} "
          f"N_TRAIN={N_TRAIN} N_HELD={N_HELD} K={K} seeds={SEEDS}")

    def bs(cf, idx):
        return float(np.mean([ST.success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                              for i in range(len(idx))]))

    per_seed = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S)
        he = perm[:N_HELD]; tr = perm[N_HELD:N_HELD + N_TRAIN]
        Sb, Sw = BL.covariances(ch_s[tr], D)
        obs_tr = np.repeat(obs[tr], N, axis=0); X_tr = X[tr].reshape(-1, D)
        scene_mean_tr = X[tr].mean(axis=1); scene_mean_he = X[he].mean(axis=1)
        he_obs = obs[he]
        row = {"ceil": float(succ[he].mean())}
        pA = BL.train_exec(obs_tr, X_tr, None, seed, ITERS, D, obs_dim)
        row["A"] = bs(BL.rollout(pA, he_obs, None, None, seed, D), he)
        for w in WS:
            U = LB.basis_gridlap(Sb, Sw, K, H, C, w)
            c_tr = scene_mean_tr @ U; c_or = scene_mean_he @ U
            prior = BL.train_prior(obs_tr, np.repeat(c_tr, N, axis=0), seed + 10,
                                   ITERS_PRIOR, obs_dim, K)
            p = BL.train_exec(obs_tr, X_tr, U, seed, ITERS, D, obs_dim)
            row[f"w{w}"] = bs(BL.rollout(p, he_obs, U, prior(he_obs), seed, D), he)
            row[f"w{w}_or"] = bs(BL.rollout(p, he_obs, U, c_or, seed, D), he)
        per_seed[f"s{seed}"] = {k: round(v, 3) for k, v in row.items()}
        print(f"seed{seed}: {per_seed[f's{seed}']}", flush=True)

    keys = list(per_seed["s0"].keys())
    pooled = {k: round(float(np.mean([per_seed[f's{s}'][k] for s in SEEDS])), 3) for k in keys}
    out = {"config": {"HID": HID, "ITERS": ITERS, "N_TRAIN": N_TRAIN, "N_HELD": N_HELD,
                      "K": K, "WS": WS, "seeds": SEEDS}, "per_seed": per_seed, "pooled": pooled}
    json.dump(out, open(os.path.join(HERE, "glap_sweep_result.json"), "w"), indent=2)
    print("POOLED:", json.dumps(pooled, indent=2))
    print("GLAP_SWEEP_DONE=ok")


if __name__ == "__main__":
    main()
