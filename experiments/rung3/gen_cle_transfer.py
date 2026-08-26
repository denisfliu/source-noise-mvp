"""Closed-loop execution for the cross-embodiment transfer, stage 1 (autograd env).
Freeze the grid-Laplacian subspace on the set-A arms, train the scene-to-coordinate
prior on set-A, relearn only the flow executor on the held-out arm, and generate
action chunks for held-out scenes. Also trains a scratch policy on the held-out arm
for comparison. Saves cle_chunks_<HELD>.npz for execution in robosuite (stage 2).

Mirrors transfer6d.py for the training path, then generates and saves chunks in the
same format as gen_cle.py.
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
ITERS = int(os.environ.get("SNMVP_ITERS", "10000"))
ITERS_PRIOR = 3000
N_HELD = 40
SEED = 0
SET_A = os.environ.get("SNMVP_SETA", "Panda,IIWA,UR5e").split(",")
HELD = os.environ.get("SNMVP_HELD", "Jaco")
DATA_DIR = os.path.join(HERE, "data_pose6d_hard")


def load(arm):
    return np.load(os.path.join(DATA_DIR, f"{arm}.npz"))["chunks"].astype(float)


def main():
    arms = SET_A + [HELD]
    ch = {a: load(a) for a in arms}
    obs = np.load(os.path.join(DATA_DIR, f"{HELD}.npz"))["obs"]     # scenes paired across arms
    S, N = ch[HELD].shape[:2]
    scale = 1.0 / np.mean([np.abs(ch[a]).mean() for a in ch])
    Xc = {a: (ch[a] * scale).reshape(S, N, D) for a in ch}
    tgt, obst, r, aa = HD.scene_targets(obs)
    obs_dim = obs.shape[1]

    bmean = {a: Xc[a].mean(axis=1) for a in ch}
    Xarms = np.stack([bmean[a] for a in SET_A], axis=1)
    Sb, Sw = BL.covariances(Xarms, D)
    invariant = Xarms.mean(axis=1)
    U = LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(S); he = perm[:N_HELD]; tr = perm[N_HELD:]
    prior = BL.train_prior(obs[tr], invariant[tr] @ U, SEED + 10, ITERS_PRIOR, obs_dim, K)

    ob_tr = np.repeat(obs[tr], N, axis=0); X_tr = Xc[HELD][tr].reshape(-1, D)
    pS = BL.train_exec(ob_tr, X_tr, None, SEED, ITERS, D, obs_dim)
    pG = BL.train_exec(ob_tr, X_tr, U, SEED, ITERS, D, obs_dim)
    genS = BL.rollout(pS, obs[he], None, None, SEED, D)
    genG = BL.rollout(pG, obs[he], U, prior(obs[he]), SEED, D)

    def offline(cf):
        return np.array([HD.success(cf[i], tgt[he[i]], obst[he[i]], r[he[i]], aa[he[i]], scale)
                         for i in range(len(he))])
    off_S, off_G = offline(genS), offline(genG)
    print(f"{HELD}: offline geometric scratch {off_S.mean():.3f}  GLAP-transfer {off_G.mean():.3f}", flush=True)

    np.savez(os.path.join(HERE, f"cle_chunks_{HELD}.npz"),
             gen_S=(genS / scale).reshape(N_HELD, H, C),
             gen_GLAP=(genG / scale).reshape(N_HELD, H, C),
             obs=obs[he], off_S=off_S, off_GLAP=off_G, scale=scale)
    print("GEN_CLE_TRANSFER_DONE=ok")


if __name__ == "__main__":
    main()
