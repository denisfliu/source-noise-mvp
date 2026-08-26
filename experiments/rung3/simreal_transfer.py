"""Sim-to-real transfer with a fixed action interface. The embodiments are dynamics
variants of one arm (Section: same OSC_POSE six-channel action, different controller
gain, damping, and latency). The grid-Laplacian instruction subspace is frozen across
the simulated variants (SET_A) and transferred to a held-out variant (HELD) that
stands in for the physical system; only the flow executor (the realization) is
relearned on the held-out variant, over a sweep of how many held-out scenes are used.

SET_A and HELD are comma-separated variant file names (in data_dyn). Reports, per
number of held-out training scenes, scratch / grid-Laplacian transfer / oracle
success, and the cross-variant coordinate invariance of the frozen subspace.
"""
import json, os, sys
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
ITERS = int(os.environ.get("SNMVP_ITERS", "9000"))
ITERS_PRIOR = 3000
N_HELD = 40
NTRAIN_SWEEP = [10, 25, 50]
SEEDS = [0, 1, 2]
SET_A = os.environ.get("SNMVP_SETA", "sim1,sim2,sim3").split(",")
HELD = os.environ.get("SNMVP_HELD", "real").split(",")
DATA_DIR = os.path.join(HERE, "data_dyn")


def load(v):
    z = np.load(os.path.join(DATA_DIR, f"{v}.npz"))
    return z["chunks"].astype(float), z["obs"], z["success"]


def main():
    variants = SET_A + HELD
    ch = {}; succ = {}; obs = None
    for v in variants:
        c, o, s = load(v); ch[v] = c; succ[v] = s; obs = o if obs is None else obs
    S, N = ch[SET_A[0]].shape[:2]
    scale = 1.0 / np.mean([np.abs(ch[v]).mean() for v in ch])
    Xc = {v: (ch[v] * scale).reshape(S, N, D) for v in ch}
    tgt, obst, r, aa = HD.scene_targets(obs)
    obs_dim = obs.shape[1]
    print(f"set_A={SET_A} held={HELD}; ceilings " +
          " ".join(f"{v}:{succ[v].mean():.3f}" for v in ch), flush=True)

    bmean = {v: Xc[v].mean(axis=1) for v in ch}
    Xsim = np.stack([bmean[v] for v in SET_A], axis=1)
    Sb, Sw = BL.covariances(Xsim, D)
    invariant = Xsim.mean(axis=1)
    U = LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W)
    gensep = float(np.linalg.eigvalsh(np.linalg.solve(Sw, Sb)).max())
    hb = HELD[0]
    cinv = float(np.mean([np.linalg.norm(bmean[hb] @ U - invariant @ U, axis=1).mean()
                          / (np.linalg.norm(invariant @ U, axis=1).mean() + 1e-9)]))
    print(f"top_gen_eig={gensep:.1f}  held-out c-invariance={cinv:.3f}", flush=True)

    def bs(cf, idx):
        return float(np.mean([HD.success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                              for i in range(len(idx))]))

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S); he = perm[:N_HELD]; pool = perm[N_HELD:]
        prior = BL.train_prior(obs[pool], invariant[pool] @ U, seed + 10, ITERS_PRIOR, obs_dim, K)
        for ntr in NTRAIN_SWEEP:
            tr = pool[:ntr]
            ob_tr = np.repeat(obs[tr], N, axis=0); X_tr = Xc[hb][tr].reshape(-1, D)
            pS = BL.train_exec(ob_tr, X_tr, None, seed, ITERS, D, obs_dim)
            pG = BL.train_exec(ob_tr, X_tr, U, seed, ITERS, D, obs_dim)
            row = results.setdefault(f"n{ntr}", {"S": [], "GLAP": [], "GLAP_or": []})
            row["S"].append(bs(BL.rollout(pS, obs[he], None, None, seed, D), he))
            row["GLAP"].append(bs(BL.rollout(pG, obs[he], U, prior(obs[he]), seed, D), he))
            row["GLAP_or"].append(bs(BL.rollout(pG, obs[he], U, bmean[hb][he] @ U, seed, D), he))
        print(f"seed{seed} done", flush=True)

    pooled = {n: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for n, d in results.items()}
    out = {"config": {"SET_A": SET_A, "HELD": HELD, "K": K, "top_gen_eig": round(gensep, 1),
                      "held_c_invariance": round(cinv, 3)},
           "ceilings": {v: round(float(succ[v].mean()), 3) for v in ch}, "pooled": pooled}
    json.dump(out, open(os.path.join(HERE, "simreal_result.json"), "w"), indent=2)
    print("POOLED:", json.dumps(pooled, indent=2))
    print("SIMREAL_DONE=ok")


if __name__ == "__main__":
    main()
