"""Variable-DOF cross-embodiment transfer. Embodiments have different action
dimensions: six-channel pose embodiments (position + orientation, from
data_pose6d_hard) and three-channel position embodiments (from data_pos3). The
pinned instruction lives in the shared three-channel end-effector position space
and is embedded into each embodiment's full action space, so the same instruction
coordinate applies regardless of the embodiment's action dimension.

Shared subspace U0 (grid-Laplacian on the H-by-3 position trajectory, selected by
cross-embodiment coherence over set-A) is frozen; a set-A prior maps the scene to
its coordinate; each embodiment's executor is trained in its own action space with
U0 embedded on the position channels; only the executor is relearned on the
held-out embodiment. Success is position reach-and-clear, common to all embodiments.

Set-A and held-out are given as name:C entries via SNMVP_SETA and SNMVP_HELD,
where C is 6 (pose) or 3 (position).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import basis_lab as BL                  # noqa: E402
import laplacian_basis as LB            # noqa: E402

H = 32
BL.H = H; BL.HID = 256
DP = 3; DPOS = H * DP                    # shared position space dimension
K = int(os.environ.get("SNMVP_K", "10"))
GLAP_W = 0.5
ITERS = int(os.environ.get("SNMVP_ITERS", "10000"))
ITERS_PRIOR = 3000
N_HELD = 40
TOL_POS = 0.03
OVERCLEAR = 0.12
SEEDS = [0, 1, 2]
# entries "name:C"
SET_A = [e.split(":") for e in os.environ.get("SNMVP_SETA", "Panda:6,IIWA:6,UR5e:6").split(",")]
HELD = [e.split(":") for e in os.environ.get("SNMVP_HELD", "Panda:3").split(",")]


def load(name, C):
    C = int(C)
    d = "data_pose6d_hard" if C == 6 else "data_pos3"
    z = np.load(os.path.join(HERE, d, f"{name}.npz"))
    return z["chunks"].astype(float), z["obs"], int(C)


def embed(U0, C):
    """Embed shared position subspace U0 (DPOS,k) into a full action space of C
    channels, placing it on the position channels (0,1,2)."""
    k = U0.shape[1]
    Uf = np.zeros((H * C, k))
    Ur = U0.reshape(H, DP, k)
    full = np.zeros((H, C, k)); full[:, :DP, :] = Ur
    return full.reshape(H * C, k)


def pos_success(chunk_full_scaled, C, scale, tgt, obst, r):
    ch = chunk_full_scaled.reshape(H, C) / scale
    pos = np.cumsum(ch[:, :DP], axis=0)
    if np.linalg.norm(pos[-1] - tgt) >= TOL_POS:
        return 0.0
    if (np.linalg.norm(pos - obst, axis=1) <= r).any():
        return 0.0
    return 1.0


def main():
    embods = {n: load(n, C) for n, C in (SET_A + HELD)}
    names_A = [n for n, C in SET_A]; names_H = [n for n, C in HELD]
    obs = embods[names_A[0]][1]
    S, N = embods[names_A[0]][0].shape[:2]
    # global position scale (common across embodiments so the shared coordinate matches)
    scale = 1.0 / np.mean([np.abs(embods[n][0][..., :DP]).mean() for n in embods])
    rad, s_o, r, lat = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    tgt = np.stack([rad, np.zeros_like(rad), np.zeros_like(rad)], axis=1)
    obst = np.stack([s_o * rad, lat, np.zeros_like(rad)], axis=1)
    obs_dim = obs.shape[1]
    print(f"set_A={SET_A} held={HELD}; S={S} N={N} K={K} scale={scale:.1f}")
    for n in embods:
        print(f"  {n}: C={embods[n][2]} demo pos-success {embods[n][2] and float(np.load(os.path.join(HERE, ('data_pose6d_hard' if embods[n][2]==6 else 'data_pos3'), n+'.npz'))['success'].mean()):.3f}")

    # per-embodiment scaled full chunks and shared-position mean
    full_s = {n: (embods[n][0] * scale).reshape(S, N, H * embods[n][2]) for n in embods}
    posmean = {n: (embods[n][0][..., :DP] * scale).mean(axis=1).reshape(S, DPOS) for n in embods}

    def bs(cf, C, idx):
        return float(np.mean([pos_success(cf[i], C, scale, tgt[idx[i]], obst[idx[i]], r[idx[i]])
                              for i in range(len(idx))]))

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S); he = perm[:N_HELD]; tr = perm[N_HELD:]
        # shared subspace from set-A position, cross-embodiment coherence
        Xpos = np.stack([posmean[n] for n in names_A], axis=1)          # (S,|A|,DPOS)
        Sb, Sw = BL.covariances(Xpos, DPOS)
        U0 = LB.basis_gridlap(Sb, Sw, K, H, DP, GLAP_W)                 # (DPOS,K)
        invariant = Xpos.mean(axis=1)                                  # (S,DPOS)
        prior = BL.train_prior(obs[tr], invariant[tr] @ U0, seed + 10, ITERS_PRIOR, obs_dim, K)

        row = {}
        for n, C in HELD:
            C = int(C); D = H * C; Uf = embed(U0, C)
            ob_tr = np.repeat(obs[tr], N, axis=0); X_tr = full_s[n][tr].reshape(-1, D)
            pS = BL.train_exec(ob_tr, X_tr, None, seed, ITERS, D, obs_dim)
            row[f"{n}_S"] = bs(BL.rollout(pS, obs[he], None, None, seed, D), C, he)
            pG = BL.train_exec(ob_tr, X_tr, Uf, seed, ITERS, D, obs_dim)
            row[f"{n}_GLAP"] = bs(BL.rollout(pG, obs[he], Uf, prior(obs[he]), seed, D), C, he)
            c_or = posmean[n][he] @ U0
            row[f"{n}_GLAP_or"] = bs(BL.rollout(pG, obs[he], Uf, c_or, seed, D), C, he)
        results[f"s{seed}"] = {k: round(v, 3) for k, v in row.items()}
        print(f"seed{seed}: {results[f's{seed}']}", flush=True)

    keys = list(results["s0"].keys())
    pooled = {k: round(float(np.mean([results[f's{s}'][k] for s in SEEDS])), 3) for k in keys}
    out = {"config": {"SET_A": SET_A, "HELD": HELD, "K": K, "ITERS": ITERS},
           "pooled": pooled, "per_seed": results}
    tag = "_".join(f"{n}{C}" for n, C in HELD)
    json.dump(out, open(os.path.join(HERE, f"vardof_{tag}_result.json"), "w"), indent=2)
    print("POOLED:", json.dumps(pooled, indent=2))
    print("VARDOF_DONE=ok")


if __name__ == "__main__":
    main()
