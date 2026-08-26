"""General deconvolution evaluator for the ablation. For a held-out body, recompute
the deterministic planned reference from the scene, identify a per-channel
finite-impulse-response filter from a few of the body's demonstrations, reconstruct
the body's action chunk by applying that filter to the planned reference, and score
it. Works for three-channel (position) and six-channel (pose) actions, so the same
method covers the cross-arm, sim-to-real, and variable-dimension cases. Reported over
a sweep of how many demonstrations the identification uses, with the filter
reconstruction error alongside the success.

Env: SNMVP_DATA (data directory), SNMVP_BODY (npz name), SNMVP_C (3 or 6),
SNMVP_TAG (output label).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = 32
OVERCLEAR = 0.12
COUPLE = 1.0
BUMP_W = 0.16
L = 6
N_HELD = 40
N_ID_SWEEP = [2, 5, 10, 25]
SEEDS = [0, 1, 2]
TOL_POS = 0.03
TOL_ROT = 0.20
DATA = os.environ["SNMVP_DATA"]
BODY = os.environ["SNMVP_BODY"]
C = int(os.environ["SNMVP_C"])
TAG = os.environ.get("SNMVP_TAG", f"{BODY}_C{C}")


def planned_canonical(obs_row):
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
    if C == 3:
        return dpos
    bank = COUPLE * (-np.sign(lat)) * np.clip(3.0 * (r + OVERCLEAR), 0.0, 0.7)
    dori = np.zeros((H, 3)); dori[:, 0] = bank * np.diff(p)
    return np.concatenate([dpos, dori], axis=1)


def success(chunk, obs_row):
    rad, s_o, r, lat = obs_row
    pos = np.cumsum(chunk[:, :3], axis=0)
    if np.linalg.norm(pos[-1] - np.array([rad, 0.0, 0.0])) >= TOL_POS:
        return 0.0
    if (np.linalg.norm(pos - np.array([s_o * rad, lat, 0.0]), axis=1) <= r).any():
        return 0.0
    if C == 6:
        aa = chunk[:, 3:].sum(axis=0)
        bank = COUPLE * (-np.sign(lat)) * np.clip(3.0 * (r + OVERCLEAR), 0.0, 0.7)
        if np.linalg.norm(aa - np.array([bank, 0.0, 0.0])) >= TOL_ROT:
            return 0.0
    return 1.0


def fir_fit(P_list, A_list):
    hs = np.zeros((C, L))
    for c in range(C):
        rows, targ = [], []
        for P, A in zip(P_list, A_list):
            for n in range(H):
                rows.append([P[n - k, c] if n - k >= 0 else 0.0 for k in range(L)])
                targ.append(A[n, c])
        hs[c] = np.linalg.lstsq(np.array(rows), np.array(targ), rcond=None)[0]
    return hs


def fir_apply(P, hs):
    out = np.zeros((H, C))
    for c in range(C):
        for n in range(H):
            out[n, c] = sum(hs[c, k] * (P[n - k, c] if n - k >= 0 else 0.0) for k in range(L))
    return out


def main():
    z = np.load(os.path.join(HERE, DATA, f"{BODY}.npz"))
    ch, obs, succ = z["chunks"].astype(float), z["obs"], z["success"]
    S, N = ch.shape[:2]
    plan = np.stack([planned_canonical(obs[i]) for i in range(S)])
    res = {}
    for nid in N_ID_SWEEP:
        ss, ee = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(S); he = perm[:N_HELD]; pool = perm[N_HELD:]
            hs = fir_fit([plan[i] for i in pool[:nid]], [ch[i, 0] for i in pool[:nid]])
            s_he, e_he = [], []
            for i in he:
                a_hat = fir_apply(plan[i], hs)
                s_he.append(success(a_hat, obs[i]))
                e_he.append(np.sqrt(((a_hat - ch[i, 0]) ** 2).sum() / (((ch[i, 0]) ** 2).sum() + 1e-12)))
            ss.append(float(np.mean(s_he))); ee.append(float(np.mean(e_he)))
        res[f"nid{nid}"] = {"deconv_success": round(float(np.mean(ss)), 3),
                            "recon_rel_err": round(float(np.mean(ee)), 3)}
        print(f"{TAG} n_id={nid}: success={res[f'nid{nid}']['deconv_success']} "
              f"recon_err={res[f'nid{nid}']['recon_rel_err']}", flush=True)
    out = {"tag": TAG, "body": BODY, "C": C, "ceiling": round(float(succ.mean()), 3), "deconv": res}
    json.dump(out, open(os.path.join(HERE, f"deconv_{TAG}.json"), "w"), indent=2)
    print(f"DECONV_EVAL_DONE={TAG}")


if __name__ == "__main__":
    main()
