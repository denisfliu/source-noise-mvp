"""Closed-loop execution, stage 1 (autograd env): train scratch and grid-Laplacian
policies on the coupled 6-DOF data, generate action chunks for held-out scenes, and
save them for execution in robosuite (stage 2). Also records the offline geometric
success of the same chunks, so stage 2 can compare closed-loop against offline.

Saves cle_chunks.npz: gen_S, gen_GLAP (N_HELD,H,6) canonical pose-delta chunks
(unscaled), obs (N_HELD,4)=[rad,s_o,r,lateral], off_S, off_GLAP (N_HELD,) offline
geometric success, scale.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
os.environ.setdefault("SNMVP_DS", "c1")
import basis_lab as BL                  # noqa: E402
import laplacian_basis as LB            # noqa: E402
import structure_test_pose6d_hard as HD  # noqa: E402

H, C, D = HD.H, HD.C, HD.D
BL.H = H; BL.HID = 256
K = 12; GLAP_W = 0.5
ITERS = int(os.environ.get("SNMVP_ITERS", "12000"))
ITERS_PRIOR = 3000
N_HELD = 40
SEED = 0


def main():
    d = np.load(HD.DATA)
    chunks, obs = d["chunks"].astype(float), d["obs"]
    S, N = chunks.shape[:2]
    scale = 1.0 / np.abs(chunks).mean()
    X = (chunks * scale).reshape(S, N, D)
    tgt, obst, r, aa = HD.scene_targets(obs)
    obs_dim = obs.shape[1]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(S)
    he = perm[:N_HELD]; tr = perm[N_HELD:]
    Sb, Sw = BL.covariances((chunks * scale)[tr], D)
    U = LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W)
    obs_tr = np.repeat(obs[tr], N, axis=0); X_tr = X[tr].reshape(-1, D)
    smean_tr = X[tr].mean(axis=1)
    prior = BL.train_prior(obs_tr, np.repeat(smean_tr @ U, N, axis=0), SEED + 10,
                           ITERS_PRIOR, obs_dim, K)

    pS = BL.train_exec(obs_tr, X_tr, None, SEED, ITERS, D, obs_dim)
    pG = BL.train_exec(obs_tr, X_tr, U, SEED, ITERS, D, obs_dim)
    genS = BL.rollout(pS, obs[he], None, None, SEED, D)
    genG = BL.rollout(pG, obs[he], U, prior(obs[he]), SEED, D)

    def offline(cf, idx):
        return np.array([HD.success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                         for i in range(len(idx))])
    off_S = offline(genS, he); off_G = offline(genG, he)
    print(f"offline geometric: scratch {off_S.mean():.3f}  GLAP {off_G.mean():.3f}", flush=True)

    np.savez(os.path.join(HERE, "cle_chunks.npz"),
             gen_S=(genS / scale).reshape(N_HELD, H, C),
             gen_GLAP=(genG / scale).reshape(N_HELD, H, C),
             obs=obs[he], off_S=off_S, off_GLAP=off_G, scale=scale)
    print("GEN_CLE_DONE=ok")


if __name__ == "__main__":
    main()
