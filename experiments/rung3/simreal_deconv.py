"""System identification and deconvolution on the sim-to-real setup, compared with
the additive grid-Laplacian transfer from simreal_transfer.py.

The planned reference for a scene is deterministic under the collection recipe, so
it is recomputed in the canonical frame from the scene descriptor. For the held-out
("real") dynamics variant, a per-channel finite-impulse-response filter mapping the
planned trajectory to the achieved trajectory is identified by least squares from a
few held-out demonstrations, and the held-out achieved trajectory for a new scene is
reconstructed by applying that filter to the planned reference. Success is scored by
the same offline geometric measure used elsewhere. Reported over a sweep of how many
held-out demonstrations the identification uses, alongside the reconstruction error
of the filter, so the linear-time-invariant fit to the robosuite dynamics is visible.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
os.environ.setdefault("SNMVP_DS", "c1")
import structure_test_pose6d_hard as HD  # noqa: E402

H, C = HD.H, HD.C
OVERCLEAR = 0.12
COUPLE = 1.0
BUMP_W = 0.16
L = 6                                    # filter length
N_HELD = 40
N_ID_SWEEP = [2, 5, 10, 25]
SEEDS = [0, 1, 2]
REAL = "real"
DATA_DIR = os.path.join(HERE, "data_dyn")


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
    bank = COUPLE * (-np.sign(lat)) * np.clip(3.0 * (r + OVERCLEAR), 0.0, 0.7)
    dori = np.zeros((H, 3)); dori[:, 0] = bank * np.diff(p)
    return np.concatenate([dpos, dori], axis=1)          # (H, C)


def fir_fit(planned_list, achieved_list):
    hs = np.zeros((C, L))
    for c in range(C):
        rows, targ = [], []
        for P, A in zip(planned_list, achieved_list):
            for n in range(H):
                rows.append([P[n - k, c] if n - k >= 0 else 0.0 for k in range(L)])
                targ.append(A[n, c])
        h, _, _, _ = np.linalg.lstsq(np.array(rows), np.array(targ), rcond=None)
        hs[c] = h
    return hs


def fir_apply(planned, hs):
    out = np.zeros((H, C))
    for c in range(C):
        for n in range(H):
            out[n, c] = sum(hs[c, k] * (planned[n - k, c] if n - k >= 0 else 0.0)
                            for k in range(L))
    return out


def main():
    z = np.load(os.path.join(DATA_DIR, f"{REAL}.npz"))
    ch, obs, succ = z["chunks"].astype(float), z["obs"], z["success"]
    S, N = ch.shape[:2]
    tgt, obst, r, aa = HD.scene_targets(obs)
    plan = np.stack([planned_canonical(obs[i]) for i in range(S)])        # (S,H,C)
    print(f"real ceiling {succ.mean():.3f}; S={S} N={N} L={L}", flush=True)

    def score(chunk_hc, i):
        return HD.success(chunk_hc.reshape(-1), tgt[i], obst[i], r[i], aa[i], scale=1.0)

    results = {}
    for nid in N_ID_SWEEP:
        succs, recon = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(S); he = perm[:N_HELD]; pool = perm[N_HELD:]
            idsc = pool[:nid]
            hs = fir_fit([plan[i] for i in idsc], [ch[i, 0] for i in idsc])
            s_he, e_he = [], []
            for i in he:
                a_hat = fir_apply(plan[i], hs)
                s_he.append(score(a_hat, i))
                e_he.append(np.sqrt(((a_hat - ch[i, 0]) ** 2).sum() / ((ch[i, 0]) ** 2).sum()))
            succs.append(float(np.mean(s_he))); recon.append(float(np.mean(e_he)))
        results[f"nid{nid}"] = {"deconv_success": round(float(np.mean(succs)), 3),
                                "recon_rel_err": round(float(np.mean(recon)), 3)}
        print(f"n_id={nid}: deconv_success={results[f'nid{nid}']['deconv_success']} "
              f"recon_rel_err={results[f'nid{nid}']['recon_rel_err']}", flush=True)

    # baselines from the additive transfer run, if present
    base = {}
    bp = os.path.join(HERE, "simreal_result.json")
    if os.path.exists(bp):
        bj = json.load(open(bp))["pooled"]
        base = {k: {"scratch": v["S"], "GLAP_additive": v["GLAP"]} for k, v in bj.items()}
    out = {"real_ceiling": round(float(succ.mean()), 3), "deconv": results,
           "additive_baseline": base}
    json.dump(out, open(os.path.join(HERE, "simreal_deconv_result.json"), "w"), indent=2)
    print("BASELINE(additive):", json.dumps(base))
    print("SIMREAL_DECONV_DONE=ok")


if __name__ == "__main__":
    main()
