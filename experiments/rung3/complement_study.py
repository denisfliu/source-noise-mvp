"""Complement / realization study: does the factoring strategy work when the pinned
instruction is the PLANNED task path (body-invariant) rather than the ACHIEVED
trajectory (realization-contaminated), and is a structured realization model
(deconvolution) needed over a black-box relearned executor?

Controlled four-way comparison on the sim-to-real dynamics data (data_dyn: one arm,
fixed OSC_POSE six-channel interface, dynamics varied by controller gain / damping /
latency; the task is the bottlenecked coupled 6-DOF detour). Set-A = the simulated
variants; held-out = 'real'. Everything shares the same held-out data budget (ntr
scenes) and the same offline geometric success metric.

  S      scratch executor on the held-out variant (no pin)
  ACH    grid-Laplacian pinned on the cross-variant coherent ACHIEVED subspace
         (the current transfer strategy) + set-A prior + relearned executor
  PLAN   grid-Laplacian pinned on the PLANNED path (deterministic from the scene,
         identical across variants) + prior + relearned executor. The instruction
         is trivially body-invariant; the executor is the realization complement.
  DECONV planned path + a per-channel FIR realization identified on the same ntr
         held-out scenes (system-ID / deconvolution complement); no learned executor.

Reports pooled success over seeds and a held-out-scene sweep, plus the cross-variant
coordinate invariance of each pin (achieved vs planned) and the FIR reconstruction
error, so the mechanism is visible.
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
NTRAIN_SWEEP = [int(x) for x in os.environ.get("SNMVP_NTR", "10,25,50").split(",")]
SEEDS = [0, 1, 2]
L = 6                                    # FIR filter length
OVERCLEAR = 0.12; COUPLE = 1.0; BUMP_W = 0.16
SET_A = os.environ.get("SNMVP_SETA", "sim1,sim2,sim3").split(",")
HELD = os.environ.get("SNMVP_HELD", "real").split(",")
DATA_DIR = os.path.join(HERE, "data_dyn")


def load(v):
    z = np.load(os.path.join(DATA_DIR, f"{v}.npz"))
    return z["chunks"].astype(float), z["obs"], z["success"]


def planned_canonical(obs_row):
    """Deterministic planned canonical pose-delta chunk (H,C) from the scene recipe
    (matches collect_dyn / simreal_deconv)."""
    rad, s_o, r, lat = obs_row
    s = np.linspace(0, 1, H + 1)
    p = np.clip(s / 0.72, 0.0, 1.0)
    prog = rad * (3 * p ** 2 - 2 * p ** 3)
    q = prog / rad
    side = -np.sign(lat)
    amp = r + OVERCLEAR
    raw = np.exp(-((q - s_o) ** 2) / (2 * BUMP_W ** 2))
    ramp = (1 - q) * raw[0] + q * raw[-1]
    peak = 1.0 - ((1 - s_o) * raw[0] + s_o * raw[-1])
    bump = side * amp * (raw - ramp) / max(peak, 1e-6)
    pos = np.stack([prog, bump, np.zeros(H + 1)], axis=1)
    pos[0] = 0.0; pos[-1] = np.array([rad, 0.0, 0.0])
    dpos = np.diff(pos, axis=0)
    bank = COUPLE * (-np.sign(lat)) * np.clip(3.0 * (r + OVERCLEAR), 0.0, 0.7)
    dori = np.zeros((H, 3)); dori[:, 0] = bank * np.diff(p)
    return np.concatenate([dpos, dori], axis=1)


def fir_fit(planned_list, achieved_list):
    hs = np.zeros((C, L))
    for c in range(C):
        rows, targ = [], []
        for P, A in zip(planned_list, achieved_list):
            for n in range(H):
                rows.append([P[n - k, c] if n - k >= 0 else 0.0 for k in range(L)])
                targ.append(A[n, c])
        hs[c] = np.linalg.lstsq(np.array(rows), np.array(targ), rcond=None)[0]
    return hs


def fir_apply(planned, hs):
    out = np.zeros((H, C))
    for c in range(C):
        for n in range(H):
            out[n, c] = sum(hs[c, k] * (planned[n - k, c] if n - k >= 0 else 0.0)
                            for k in range(L))
    return out


def scene_var_basis(P_scaled, k):
    """grid-Laplacian modes ranked by planned-path scene variance (no body axis for
    the plan, so selection is by e^T Sigma_scene e; Sw=I)."""
    Sb = np.cov(P_scaled.reshape(P_scaled.shape[0], D), rowvar=False)
    return LB.basis_gridlap(Sb, np.eye(D), k, H, C, GLAP_W)


def main():
    variants = SET_A + HELD
    ch = {}; succ = {}; obs = None
    for v in variants:
        c, o, s = load(v); ch[v] = c; succ[v] = s; obs = o if obs is None else obs
    S, N = ch[SET_A[0]].shape[:2]
    hb = HELD[0]
    scale = 1.0 / np.mean([np.abs(ch[v]).mean() for v in ch])
    Xc = {v: (ch[v] * scale).reshape(S, N, D) for v in ch}
    tgt, obst, r, aa = HD.scene_targets(obs)
    obs_dim = obs.shape[1]

    # planned paths (raw for FIR, scaled for the pin/executor space)
    plan_raw = np.stack([planned_canonical(obs[i]) for i in range(S)])       # (S,H,C)
    plan_scaled = (plan_raw * scale).reshape(S, D)                           # (S,D)

    print(f"set_A={SET_A} held={hb}; ceilings " +
          " ".join(f"{v}:{succ[v].mean():.3f}" for v in ch) + f"; scale={scale:.1f}", flush=True)

    # ACH pin: cross-variant coherent achieved subspace (= simreal_transfer's U)
    bmean = {v: Xc[v].mean(axis=1) for v in ch}
    Xsim = np.stack([bmean[v] for v in SET_A], axis=1)
    Sb, Sw = BL.covariances(Xsim, D)
    ach_inv = Xsim.mean(axis=1)                                             # (S,D)
    U_ach = LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W)
    # PLAN pin: grid-Laplacian on planned scene-variance
    U_plan = scene_var_basis(plan_scaled, K)

    def gap(U, coord):
        cA = coord @ U
        return round(float(np.linalg.norm(bmean[hb] @ U - cA, axis=1).mean()
                           / (np.linalg.norm(cA, axis=1).mean() + 1e-9)), 3)
    # ACH: cross-variant invariance of the pinned ACHIEVED coordinate (set-A vs held-out).
    # PLAN: the planned coordinate is body-invariant by construction (invariance = 0);
    #       this number is the achieved-vs-planned realization gap the complement must cover.
    ci_ach = gap(U_ach, ach_inv)
    realization_gap_plan = gap(U_plan, plan_scaled)
    print(f"ACH cross-variant c-invariance={ci_ach}  |  "
          f"PLAN achieved-vs-planned realization gap={realization_gap_plan}", flush=True)

    def bs(cf, idx):
        return float(np.mean([HD.success(cf[i], tgt[idx[i]], obst[idx[i]], r[idx[i]], aa[idx[i]], scale)
                              for i in range(len(idx))]))

    def score_raw(a_hat, i):
        return HD.success(a_hat.reshape(-1), tgt[i], obst[i], r[i], aa[i], scale=1.0)

    # stricter diagnostic: how far each produced trajectory is from the held-out body's
    # OWN achieved demos (per-scene demo mean), in raw units. A method can "succeed"
    # under the lenient geometric metric while sitting off the body's realized manifold
    # (e.g. deconvolution's lightly-filtered plan); this makes that gap quantitative.
    ach_ref = {i: ch[hb][i].mean(axis=0) for i in range(S)}          # (H,C) raw per scene

    def traj_err_scaled(cf, idx):
        return float(np.mean([np.sqrt(((cf[m].reshape(H, C) / scale - ach_ref[idx[m]]) ** 2).sum()
                    / ((ach_ref[idx[m]] ** 2).sum() + 1e-12)) for m in range(len(idx))]))

    def traj_err_raw(a_list, idx):
        return float(np.mean([np.sqrt(((a_list[m] - ach_ref[idx[m]]) ** 2).sum()
                    / ((ach_ref[idx[m]] ** 2).sum() + 1e-12)) for m in range(len(idx))]))

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(S); he = perm[:N_HELD]; pool = perm[N_HELD:]
        prior_ach = BL.train_prior(obs[pool], ach_inv[pool] @ U_ach, seed + 10, ITERS_PRIOR, obs_dim, K)
        prior_plan = BL.train_prior(obs[pool], plan_scaled[pool] @ U_plan, seed + 20, ITERS_PRIOR, obs_dim, K)
        for ntr in NTRAIN_SWEEP:
            tr = pool[:ntr]
            ob_tr = np.repeat(obs[tr], N, axis=0); X_tr = Xc[hb][tr].reshape(-1, D)
            pS = BL.train_exec(ob_tr, X_tr, None, seed, ITERS, D, obs_dim)
            pA = BL.train_exec(ob_tr, X_tr, U_ach, seed, ITERS, D, obs_dim)
            pP = BL.train_exec(ob_tr, X_tr, U_plan, seed, ITERS, D, obs_dim)
            # deconvolution: FIR fit on the same ntr scenes (all demos), reconstruct held-out
            hs = fir_fit([plan_raw[i] for i in tr for _ in range(N)],
                         [ch[hb][i, j] for i in tr for j in range(N)])
            a_dec = [fir_apply(plan_raw[i], hs) for i in he]
            fir_err = float(np.mean([np.sqrt(((a_dec[m] - ch[hb][he[m], 0]) ** 2).sum()
                            / (((ch[hb][he[m], 0]) ** 2).sum() + 1e-12)) for m in range(len(he))]))
            cfS = BL.rollout(pS, obs[he], None, None, seed, D)
            cfA = BL.rollout(pA, obs[he], U_ach, prior_ach(obs[he]), seed, D)
            cfP = BL.rollout(pP, obs[he], U_plan, prior_plan(obs[he]), seed, D)
            row = results.setdefault(f"n{ntr}",
                {"S": [], "ACH": [], "PLAN": [], "DECONV": [], "fir_err": [],
                 "S_traj": [], "ACH_traj": [], "PLAN_traj": [], "DECONV_traj": []})
            row["S"].append(bs(cfS, he));  row["S_traj"].append(traj_err_scaled(cfS, he))
            row["ACH"].append(bs(cfA, he)); row["ACH_traj"].append(traj_err_scaled(cfA, he))
            row["PLAN"].append(bs(cfP, he)); row["PLAN_traj"].append(traj_err_scaled(cfP, he))
            row["DECONV"].append(float(np.mean([score_raw(a_dec[m], he[m]) for m in range(len(he))])))
            row["DECONV_traj"].append(traj_err_raw(a_dec, he))
            row["fir_err"].append(fir_err)
        print(f"seed{seed} done", flush=True)

    pooled = {n: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for n, d in results.items()}
    out = {"config": {"SET_A": SET_A, "HELD": HELD, "K": K, "L": L, "ITERS": ITERS,
                      "cinv_ACH": ci_ach, "realization_gap_PLAN": realization_gap_plan},
           "ceilings": {v: round(float(succ[v].mean()), 3) for v in ch}, "pooled": pooled}
    json.dump(out, open(os.path.join(HERE, os.environ.get("SNMVP_OUT", "complement_result.json")), "w"), indent=2)
    print("POOLED:", json.dumps(pooled, indent=2))
    print("COMPLEMENT_DONE=ok")


if __name__ == "__main__":
    main()
