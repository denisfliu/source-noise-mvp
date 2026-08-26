"""Diagnose the weak 6-DOF result: is the pin BINDING, and is the executor the
bottleneck? Single seed. Reports (1) pin-binding error ||gen@U - c|| (≈0 if
pass-through works), (2) A/F/F_oracle at increasing executor capacity+data."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import basis_lab as BL                  # noqa: E402
import structure_test_pose6d as ST      # noqa: E402

BL.H = ST.H
H, C, D, K = ST.H, ST.C, ST.D, ST.K
d = np.load(ST.DATA)
chunks, obs, succ = d["chunks"].astype(float), d["obs"], d["success"]
S, N = chunks.shape[:2]
scale = 1.0 / np.abs(chunks).mean()
ch_s = chunks * scale
tgt, obst, r, aa = ST.canon_pos_from_obs(obs)
rng = np.random.default_rng(0)
perm = rng.permutation(S); tr, he = perm[ST.N_HELD:], perm[:ST.N_HELD]
X = ch_s.reshape(S, N, D)
Sb, Sw = BL.covariances(ch_s[tr], D)
U = BL.basis_fourier(Sb, Sw, K, C)
obs_tr = np.repeat(obs[tr], N, axis=0); X_tr = X[tr].reshape(-1, D)
scene_mean_tr = X[tr].mean(axis=1); scene_mean_he = X[he].mean(axis=1)
he_obs = obs[he]; obs_dim = obs.shape[1]
c_or = scene_mean_he @ U

def bs(cf, idx):
    return float(np.mean([ST.success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                          for i in range(len(idx))]))

print(f"demo ceiling held-out {succ[he].mean():.3f}; |c_or| mean {np.linalg.norm(c_or,axis=1).mean():.2f}")
for HID, ITERS in [(128, 6000), (256, 15000), (384, 25000)]:
    BL.HID = HID
    pA = BL.train_exec(obs_tr, X_tr, None, 0, ITERS, D, obs_dim)
    A = bs(BL.rollout(pA, he_obs, None, None, 0, D), he)
    pF = BL.train_exec(obs_tr, X_tr, U, 0, ITERS, D, obs_dim)
    genF = BL.rollout(pF, he_obs, U, c_or, 0, D)
    bind = float(np.linalg.norm(genF @ U - c_or, axis=1).mean())    # pass-through error
    Fo = bs(genF, he)
    print(f"HID={HID} ITERS={ITERS}: A={A:.3f} F_oracle={Fo:.3f} pin_bind_err={bind:.4f}", flush=True)
print("POSE6D_DIAG_DONE=ok")
