"""Basis comparison on the STRONGLY-BOTTLENECKED coupled 6-DOF task. Tests whether
the cross-channel grid-Laplacian pin wins LARGE where per-channel bases and scratch
fail, and (via the c0 dataset) whether that advantage vanishes without coupling.

Arms: A scratch; F Fourier; DCT path-Laplacian; GLAP grid-Laplacian(w); R random.
Each pinned basis: PRIOR + ORACLE command; per-basis prior-err. Data bottleneck
(N_TRAIN small). Run once per dataset (SNMVP_DS in {c1,c0}).

Success (offline geometric): cumsum dpos_canonical -> reach (rad,0,0) within TOL_POS
and clear obstacle disk at (s_o*rad, lateral, 0), radius r; integrated world dori ->
reach target bank axis-angle within TOL_ROT. Target aa recomputed from scene
(bank_axisangle) since orientation is not in obs.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import basis_lab as BL                  # noqa: E402
import laplacian_basis as LB            # noqa: E402

ARM = os.environ.get("SNMVP_ARM", "Panda")
DS = os.environ.get("SNMVP_DS", "c1")
COUPLE = 1.0 if DS == "c1" else 0.0
DATA = os.path.join(HERE, "data_pose6d_hard", f"{ARM}_{DS}.npz")
H, C = 32, 6
D = H * C
BL.H = H
K = int(os.environ.get("SNMVP_K", "10"))
HID = int(os.environ.get("SNMVP_HID", "256"))
BL.HID = HID
ITERS = int(os.environ.get("SNMVP_ITERS", "12000"))
ITERS_PRIOR = 3000
N_TRAIN = int(os.environ.get("SNMVP_NTRAIN", "25"))
N_HELD = 50
GLAP_W = float(os.environ.get("SNMVP_GLAPW", "0.5"))
OVERCLEAR = 0.12
SEEDS = [0, 1, 2, 3]
TOL_POS = 0.03
TOL_ROT = 0.20


def scene_targets(obs):
    """obs rows [rad, s_o, r, lateral]. Return canonical target pos, obstacle
    center, radius, and target bank |axis-angle| magnitude (signed by side)."""
    rad, s_o, r, lat = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    tgt = np.stack([rad, np.zeros_like(rad), np.zeros_like(rad)], axis=1)
    obst = np.stack([s_o * rad, lat, np.zeros_like(rad)], axis=1)
    side = np.sign(lat)                                   # collector: lateral = side*|..|
    ang = COUPLE * (-side) * np.clip(3.0 * (r + OVERCLEAR), 0.0, 0.7)   # bank, canonical +x
    aa = np.stack([ang, np.zeros_like(ang), np.zeros_like(ang)], axis=1)
    return tgt, obst, r, aa


def success(chunk_flat, tgt, obst, r, aa_tgt, scale):
    ch = chunk_flat.reshape(H, C) / scale
    pos = np.cumsum(ch[:, :3], axis=0)
    if np.linalg.norm(pos[-1] - tgt) >= TOL_POS:
        return 0.0
    if (np.linalg.norm(pos - obst, axis=1) <= r).any():
        return 0.0
    aa_final = ch[:, 3:].sum(axis=0)
    if np.linalg.norm(aa_final - aa_tgt) >= TOL_ROT:
        return 0.0
    return 1.0


def main():
    d = np.load(DATA)
    chunks, obs, succ = d["chunks"].astype(float), d["obs"], d["success"]
    S, N = chunks.shape[:2]
    scale = 1.0 / np.abs(chunks).mean()
    ch_s = chunks * scale
    X = ch_s.reshape(S, N, D)
    tgt, obst, r, aa = scene_targets(obs)
    obs_dim = obs.shape[1]
    print(f"{ARM} DS={DS} COUPLE={COUPLE}: ceiling {succ.mean():.3f}; HID={HID} "
          f"ITERS={ITERS} N_TRAIN={N_TRAIN} K={K} GLAP_W={GLAP_W}")

    def bs(cf, idx):
        return float(np.mean([success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                              for i in range(len(idx))]))

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S)
        he = perm[:N_HELD]; tr = perm[N_HELD:N_HELD + N_TRAIN]
        Sb, Sw = BL.covariances(ch_s[tr], D)
        bases = {"F": BL.basis_fourier(Sb, Sw, K, C),
                 "DCT": LB.basis_dct(Sb, Sw, K, H, C),
                 "GLAP": LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W),
                 "R": BL.basis_random(K, D, seed)}
        obs_tr = np.repeat(obs[tr], N, axis=0); X_tr = X[tr].reshape(-1, D)
        smean_tr = X[tr].mean(axis=1); smean_he = X[he].mean(axis=1)
        he_obs = obs[he]
        row = {"ceil": float(succ[he].mean())}
        pA = BL.train_exec(obs_tr, X_tr, None, seed, ITERS, D, obs_dim)
        row["A"] = bs(BL.rollout(pA, he_obs, None, None, seed, D), he)
        for name, U in bases.items():
            c_tr = smean_tr @ U; c_or = smean_he @ U
            prior = BL.train_prior(obs_tr, np.repeat(c_tr, N, axis=0), seed + 10,
                                   ITERS_PRIOR, obs_dim, K)
            p = BL.train_exec(obs_tr, X_tr, U, seed, ITERS, D, obs_dim)
            row[name] = bs(BL.rollout(p, he_obs, U, prior(he_obs), seed, D), he)
            row[f"{name}_or"] = bs(BL.rollout(p, he_obs, U, c_or, seed, D), he)
        results[f"s{seed}"] = {k: round(v, 3) for k, v in row.items()}
        print(f"seed{seed}: {results[f's{seed}']}", flush=True)

    keys = list(results["s0"].keys())
    pooled = {k: round(float(np.mean([results[f's{s}'][k] for s in SEEDS])), 3) for k in keys}
    out = {"dataset": DS, "COUPLE": COUPLE, "config": {"HID": HID, "ITERS": ITERS,
           "N_TRAIN": N_TRAIN, "K": K, "GLAP_W": GLAP_W}, "per_seed": results, "pooled": pooled}
    json.dump(out, open(os.path.join(HERE, f"pose6d_hard_{DS}_result.json"), "w"), indent=2)
    print("POOLED:", json.dumps(pooled, indent=2))
    print(f"STRUCTURE_TEST_POSE6D_HARD_{DS}_DONE=ok")


if __name__ == "__main__":
    main()
