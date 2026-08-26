"""Rung 3 Step A: does the FOURIER factoring scale to a real 6-DOF embodiment?

Reuses the validated C-channel machinery from basis_lab.py (Fourier basis over C
channels, projection pass-through pin c=Uᵀa, scene→command prior, flow executor).
Data = collect_pose6d.py 6-channel achieved pose-delta chunks [dpos_canon(3),
dori_world(3)] on the real Panda. Coherence discovery over demos = the (Σ_b, Σ_w)
between-scene/within-scene split (Σ_w over demos of a scene = the coherence
criterion). Arms:
  A  scratch flow (no pin)
  F  Fourier top-k directions by the coherence objective, pinned + prior command
  R  random orthonormal pin + its prior (control)
Success (offline geometric, this step): cumsum dpos → reach target position within
TOL_POS and clear the obstacle sphere; orientation delta integrated → reach target
axis-angle within TOL_ROT. Step B (separate) replays in sim for execution success.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import basis_lab as BL                  # noqa: E402

ARM = os.environ.get("SNMVP_ARM", "Panda")
DATA = os.path.join(HERE, "data_pose6d", f"{ARM}.npz")
H = 32
C = 6
D = H * C
BL.H = H                                # fourier_basis builds H-length modes
K = int(os.environ.get("SNMVP_K", "10"))
N_HELD = 20
ITERS = int(os.environ.get("SNMVP_ITERS", "6000"))
ITERS_PRIOR = 3000
SEEDS = [0, 1, 2]
TOL_POS = 0.03
TOL_ROT = 0.20


def canon_pos_from_obs(obs):
    """obs rows = [rad, s_o, r, cos_psi, sin_psi, aa(3)]. Canonical target is
    (rad,0,0); obstacle canonical center = (s_o*rad, 0, 0); detour perp is
    (0,cos_psi,sin_psi) — but clearance is checked against the sphere center only."""
    rad, s_o, r = obs[:, 0], obs[:, 1], obs[:, 2]
    tgt = np.stack([rad, np.zeros_like(rad), np.zeros_like(rad)], axis=1)      # (M,3)
    obst = np.stack([s_o * rad, np.zeros_like(rad), np.zeros_like(rad)], axis=1)
    aa = obs[:, 5:8]
    return tgt, obst, r, aa


def success(chunk_flat, tgt, obst, r, aa_tgt, scale):
    """chunk_flat (D,) scaled canonical [dpos(3),dori(3)] per step. Geometric."""
    ch = chunk_flat.reshape(H, C) / scale
    dpos, dori = ch[:, :3], ch[:, 3:]
    pos = np.cumsum(dpos, axis=0)                      # (H,3) canonical, start at ~origin
    if np.linalg.norm(pos[-1] - tgt) >= TOL_POS:
        return 0.0
    if (np.linalg.norm(pos - obst, axis=1) <= r).any():
        return 0.0
    # orientation: integrate world axis-angle deltas (small-angle sum is adequate
    # at these magnitudes; the demo target rotation is <0.6 rad total)
    aa_final = dori.sum(axis=0)
    if np.linalg.norm(aa_final - aa_tgt) >= TOL_ROT:
        return 0.0
    return 1.0


def main():
    d = np.load(DATA)
    chunks, obs, succ = d["chunks"].astype(float), d["obs"], d["success"]
    S, N = chunks.shape[:2]
    print(f"{ARM}: {S} scenes x {N} demos; demo pose6d ceiling {succ.mean():.3f}")
    scale = 1.0 / np.abs(chunks).mean()
    ch_s = chunks * scale
    tgt, obst, r, aa = canon_pos_from_obs(obs)
    print(f"ACT_SCALE={scale:.1f} mean|dpos|={np.abs(chunks[...,:3]).mean():.5f} "
          f"mean|dori|={np.abs(chunks[...,3:]).mean():.5f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(S)
    tr, he = perm[N_HELD:], perm[:N_HELD]
    X = ch_s.reshape(S, N, D)

    Sb, Sw = BL.covariances(ch_s[tr], D)
    bases = {"F": BL.basis_fourier(Sb, Sw, K, C), "R": BL.basis_random(K, D, 0)}

    obs_tr = np.repeat(obs[tr], N, axis=0)
    X_tr = X[tr].reshape(-1, D)
    scene_mean_tr = X[tr].mean(axis=1)
    scene_mean_he = X[he].mean(axis=1)              # held-out TRUE scene mean (oracle)
    he_obs = obs[he]
    obs_dim = obs.shape[1]

    def bs(chunks_flat, idx):
        return float(np.mean([success(chunks_flat[i], tgt[idx[i]], obst[idx[i]],
                                      r[idx[i]], aa[idx[i]], scale)
                              for i in range(len(idx))]))

    results = {}
    for seed in SEEDS:
        row = {}
        # demo ceiling on held-out (first demo of each held-out scene)
        row["ceil"] = float(succ[he].mean())
        pA = BL.train_exec(obs_tr, X_tr, None, seed, ITERS, D, obs_dim)
        chA = BL.rollout(pA, he_obs, None, None, seed, D)
        row["A"] = bs(chA, he)
        for name, U in bases.items():
            c_tr = scene_mean_tr @ U
            c_or = scene_mean_he @ U                 # oracle command (true held-out coeff)
            prior = BL.train_prior(obs_tr, np.repeat(c_tr, N, axis=0), seed + 10,
                                   ITERS_PRIOR, obs_dim, K)
            p = BL.train_exec(obs_tr, X_tr, U, seed, ITERS, D, obs_dim)
            c_pred = prior(he_obs)
            chE = BL.rollout(p, he_obs, U, c_pred, seed, D)
            row[name] = bs(chE, he)
            chO = BL.rollout(p, he_obs, U, c_or, seed, D)
            row[f"{name}_oracle"] = bs(chO, he)
            row[f"{name}_prior_err"] = round(float(np.linalg.norm(c_pred - c_or, axis=1).mean()), 3)
        results[f"s{seed}"] = {k: round(v, 3) for k, v in row.items()}
        print(f"seed{seed}: {results[f's{seed}']}", flush=True)

    keys = ["A", "F", "R", "F_oracle", "R_oracle", "F_prior_err", "R_prior_err", "ceil"]
    pooled = {k: round(float(np.mean([results[f's{s}'][k] for s in SEEDS])), 3)
              for k in keys}
    verdict = {"pooled": pooled, "F_minus_A": round(pooled["F"] - pooled["A"], 3),
               "F_minus_R": round(pooled["F"] - pooled["R"], 3),
               "Foracle_minus_F": round(pooled["F_oracle"] - pooled["F"], 3),
               "Foracle_minus_Roracle": round(pooled["F_oracle"] - pooled["R_oracle"], 3),
               "K": K, "C": C,
               "task": "6-DOF pose-reach around obstacle (offline geometric)"}
    out = {"per_seed": results, "verdict": verdict}
    json.dump(out, open(os.path.join(HERE, "pose6d_result.json"), "w"), indent=2)
    print("VERDICT:", json.dumps(verdict, indent=2))
    print("STRUCTURE_TEST_POSE6D_DONE=ok")


if __name__ == "__main__":
    main()
