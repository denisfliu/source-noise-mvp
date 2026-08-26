"""Variable-DOF transfer of the instruction subspace across a controller / action-
dimension change, with the four complement strategies. Set-A = the hard slalom under
OSC_POSE (6-ch) on Panda/IIWA/UR5e; held-out = the same task under OSC_POSITION (3-ch)
on UR5e. The shared instruction lives in the 3-channel end-effector position space;
set-A provides the subspace and the scene->coordinate prior from its position channels,
and only the held-out executor (3-ch) is relearned.

  S      scratch 3-ch executor, no pin
  ACH    pin the grid-Laplacian coherent ACHIEVED-position subspace fit on set-A pose
         bodies (their position coordinate is OSC_POSE-specific)
  PLAN   pin the deterministic PLANNED position path (identical across controllers)
  DECONV planned path + per-channel FIR realization identified on the held-out body
  *_or   commanded by the held-out body's OWN coordinate (oracle upper bound)

Success = position reach within tolerance and clearance of both slalom obstacles.
Reports pooled success over seeds and a held-out-scene sweep, the cross-controller
coordinate invariance of the ACH and PLAN pins, and a trajectory-match diagnostic
(rel error of each output vs the held-out body's own achieved demos).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import basis_lab as BL                  # noqa: E402
import laplacian_basis as LB            # noqa: E402

H = 32; DP = 3; D = H * DP
BL.H = H; BL.HID = 256
K = int(os.environ.get("SNMVP_K", "10"))
GLAP_W = 0.5
ITERS = int(os.environ.get("SNMVP_ITERS", "9000"))
ITERS_PRIOR = 3000
N_HELD = 40
NTRAIN_SWEEP = [int(x) for x in os.environ.get("SNMVP_NTR", "10,25,50").split(",")]
SEEDS = [0, 1, 2]
L = 6; OVERCLEAR = 0.085; BUMP_W = 0.08
TOL_POS = 0.03
DATA = os.path.join(HERE, "data_vardof_slalom")
SET_A = os.environ.get("SNMVP_SETA", "pose_Panda,pose_IIWA,pose_UR5e").split(",")
HELD = os.environ.get("SNMVP_HELD", "pos_UR5e")


def load(name):
    z = np.load(os.path.join(DATA, f"{name}.npz"))
    return z["chunks"].astype(float), z["obs"], z["success"]


def planned_dpos(obs_row):
    """Deterministic canonical planned position-delta chunk (H,3) for the slalom
    (matches collect_vardof_slalom.plan_canonical_xy without the style wiggle)."""
    rad, s1, s2, d1, d2, r1, r2 = obs_row
    s = np.linspace(0, 1, H + 1)
    p = np.clip(s / 0.72, 0.0, 1.0)
    prog = rad * (3 * p ** 2 - 2 * p ** 3)
    q = prog / rad

    def bump(q, s_o, amp, w=BUMP_W):
        raw = np.exp(-((q - s_o) ** 2) / (2 * w ** 2))
        ramp = (1 - q) * raw[0] + q * raw[-1]
        peak = 1.0 - ((1 - s_o) * raw[0] + s_o * raw[-1])
        return amp * (raw - ramp) / max(peak, 1e-6)

    a1 = -1.0 * (d1 + r1 + OVERCLEAR); a2 = +1.0 * (d2 + r2 + OVERCLEAR)
    lat = bump(q, s1, a1) + bump(q, s2, a2)
    prog[-1] = rad; lat[0] = 0.0; lat[-1] = 0.0
    pos = np.stack([prog, lat, np.zeros(H + 1)], axis=1)
    return np.diff(pos, axis=0)                              # (H,3)


def scene_geom(obs):
    rad, s1, s2, d1, d2, r1, r2 = [obs[:, i] for i in range(7)]
    tgt = np.stack([rad, 0 * rad, 0 * rad], 1)
    o1 = np.stack([s1 * rad, d1, 0 * rad], 1)
    o2 = np.stack([s2 * rad, -d2, 0 * rad], 1)
    return tgt, o1, o2, r1, r2


def fir_fit(P_list, A_list):
    hs = np.zeros((DP, L))
    for c in range(DP):
        rows, targ = [], []
        for P, A in zip(P_list, A_list):
            for n in range(H):
                rows.append([P[n - k, c] if n - k >= 0 else 0.0 for k in range(L)])
                targ.append(A[n, c])
        hs[c] = np.linalg.lstsq(np.array(rows), np.array(targ), rcond=None)[0]
    return hs


def fir_apply(P, hs):
    out = np.zeros((H, DP))
    for c in range(DP):
        for n in range(H):
            out[n, c] = sum(hs[c, k] * (P[n - k, c] if n - k >= 0 else 0.0) for k in range(L))
    return out


def main():
    dA = {v: load(v) for v in SET_A}
    chH, obs, succH = load(HELD)
    S, N = chH.shape[:2]
    # global position scale over all bodies
    allpos = [dA[v][0][..., :DP] for v in SET_A] + [chH[..., :DP]]
    scale = 1.0 / np.mean([np.abs(x).mean() for x in allpos])
    posA = {v: (dA[v][0][..., :DP] * scale).reshape(S, N, D) for v in SET_A}      # achieved pos (set-A)
    XcH = (chH[..., :DP] * scale).reshape(S, N, D)                                # held-out achieved pos
    tgt, o1, o2, r1, r2 = scene_geom(obs)
    obs_dim = obs.shape[1]
    plan_raw = np.stack([planned_dpos(obs[i]) for i in range(S)])                 # (S,H,3) raw
    plan_s = (plan_raw * scale).reshape(S, D)                                     # scaled, flat

    print(f"set_A={SET_A} held={HELD}; S={S} N={N} K={K} scale={scale:.1f} "
          f"ceilings " + " ".join(f"{v}:{dA[v][2].mean():.3f}" for v in SET_A) +
          f" {HELD}:{succH.mean():.3f}", flush=True)

    # ACH subspace: grid-Laplacian coherence over set-A achieved-position (cross pose-body)
    bmeanA = {v: posA[v].mean(1) for v in SET_A}
    Xbody = np.stack([bmeanA[v] for v in SET_A], 1)
    Sb, Sw = BL.covariances(Xbody, D)
    ach_inv = Xbody.mean(1)
    U_ach = LB.basis_gridlap(Sb, Sw, K, H, DP, GLAP_W)
    # PLAN subspace: grid-Laplacian on planned-path scene variance
    U_plan = LB.basis_gridlap(np.cov(plan_s, rowvar=False), np.eye(D), K, H, DP, GLAP_W)
    heldmean = XcH.mean(1)

    def inv_gap(U, coord):
        cA = coord @ U
        return round(float(np.linalg.norm(heldmean @ U - cA, axis=1).mean()
                           / (np.linalg.norm(cA, axis=1).mean() + 1e-9)), 3)
    ci_ach = inv_gap(U_ach, ach_inv)                 # cross-controller invariance of achieved coord
    gap_plan = inv_gap(U_plan, plan_s)               # achieved-vs-planned gap for the held-out body
    print(f"ACH cross-controller c-invariance={ci_ach}  |  PLAN achieved-vs-planned gap={gap_plan}",
          flush=True)

    def succ(cf, idx):
        out = []
        for m, i in enumerate(idx):
            p = np.cumsum(cf[m].reshape(H, DP) / scale, 0)
            ok = (np.linalg.norm(p[-1] - tgt[i]) < TOL_POS and
                  (np.linalg.norm(p - o1[i], axis=1) > r1[i]).all() and
                  (np.linalg.norm(p - o2[i], axis=1) > r2[i]).all())
            out.append(float(ok))
        return float(np.mean(out))

    def succ_raw(a_list, idx):
        out = []
        for m, i in enumerate(idx):
            p = np.cumsum(a_list[m], 0)
            ok = (np.linalg.norm(p[-1] - tgt[i]) < TOL_POS and
                  (np.linalg.norm(p - o1[i], axis=1) > r1[i]).all() and
                  (np.linalg.norm(p - o2[i], axis=1) > r2[i]).all())
            out.append(float(ok))
        return float(np.mean(out))

    ref = {i: chH[i, :, :, :DP].mean(0) for i in range(S)}   # held-out achieved demo mean (H,3) raw

    def traj_s(cf, idx):
        return round(float(np.mean([np.sqrt(((cf[m].reshape(H, DP) / scale - ref[idx[m]]) ** 2).sum()
                    / ((ref[idx[m]] ** 2).sum() + 1e-12)) for m in range(len(idx))])), 3)

    def traj_r(a_list, idx):
        return round(float(np.mean([np.sqrt(((a_list[m] - ref[idx[m]]) ** 2).sum()
                    / ((ref[idx[m]] ** 2).sum() + 1e-12)) for m in range(len(idx))])), 3)

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S); he = perm[:N_HELD]; pool = perm[N_HELD:]
        prior_ach = BL.train_prior(obs[pool], ach_inv[pool] @ U_ach, seed + 10, ITERS_PRIOR, obs_dim, K)
        prior_plan = BL.train_prior(obs[pool], plan_s[pool] @ U_plan, seed + 20, ITERS_PRIOR, obs_dim, K)
        for ntr in NTRAIN_SWEEP:
            tr = pool[:ntr]
            ob_tr = np.repeat(obs[tr], N, 0); X_tr = XcH[tr].reshape(-1, D)
            pS = BL.train_exec(ob_tr, X_tr, None, seed, ITERS, D, obs_dim)
            pA = BL.train_exec(ob_tr, X_tr, U_ach, seed, ITERS, D, obs_dim)
            pP = BL.train_exec(ob_tr, X_tr, U_plan, seed, ITERS, D, obs_dim)
            hs = fir_fit([plan_raw[i] for i in tr for _ in range(N)],
                         [chH[i, j, :, :DP] for i in tr for j in range(N)])
            a_dec = [fir_apply(plan_raw[i], hs) for i in he]
            cfS = BL.rollout(pS, obs[he], None, None, seed, D)
            cfA = BL.rollout(pA, obs[he], U_ach, prior_ach(obs[he]), seed, D)
            cfP = BL.rollout(pP, obs[he], U_plan, prior_plan(obs[he]), seed, D)
            cfAor = BL.rollout(pA, obs[he], U_ach, heldmean[he] @ U_ach, seed, D)
            cfPor = BL.rollout(pP, obs[he], U_plan, heldmean[he] @ U_plan, seed, D)
            row = results.setdefault(f"n{ntr}", {k: [] for k in
                ["S", "ACH", "PLAN", "DECONV", "ACH_or", "PLAN_or",
                 "S_traj", "ACH_traj", "PLAN_traj", "DECONV_traj", "fir_err"]})
            row["S"].append(succ(cfS, he)); row["S_traj"].append(traj_s(cfS, he))
            row["ACH"].append(succ(cfA, he)); row["ACH_traj"].append(traj_s(cfA, he))
            row["PLAN"].append(succ(cfP, he)); row["PLAN_traj"].append(traj_s(cfP, he))
            row["ACH_or"].append(succ(cfAor, he)); row["PLAN_or"].append(succ(cfPor, he))
            row["DECONV"].append(succ_raw(a_dec, he)); row["DECONV_traj"].append(traj_r(a_dec, he))
            row["fir_err"].append(traj_r(a_dec, he))
        print(f"seed{seed} done", flush=True)

    pooled = {n: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for n, d in results.items()}
    out = {"config": {"SET_A": SET_A, "HELD": HELD, "K": K, "ITERS": ITERS,
                      "cinv_ACH": ci_ach, "gap_PLAN": gap_plan},
           "ceilings": {**{v: round(float(dA[v][2].mean()), 3) for v in SET_A},
                        HELD: round(float(succH.mean()), 3)},
           "pooled": pooled}
    json.dump(out, open(os.path.join(HERE, os.environ.get("SNMVP_OUT", "vardof_complement_result.json")), "w"),
              indent=2)
    print("POOLED:", json.dumps(pooled, indent=2))
    print("VARDOF_COMPLEMENT_DONE=ok")


if __name__ == "__main__":
    main()
