"""Steerability + interpretability of the pass-through pin at 6-DOF (the primary
goals, demonstrated directly at high DOF with a FIXED grid-Laplacian basis).

Uses the coupled 6-DOF data (Panda_c1). Trains ONE flow executor with a K-dim
grid-Laplacian pin. Then:

INTERPRETABILITY: project demo chunks onto U -> coordinates c; correlate each
coordinate with named scene quantities (endpoint radius, obstacle side/lateral,
target bank angle). Report which coordinates are the interpretable handles.

STEERABILITY: for a fixed scene, sweep the COMMANDED value of a chosen coordinate
across its demo range; generate; measure
  (a) pass-through fidelity: realized c_k (= Uᵀ of generated chunk) vs commanded
      c_k -> slope ~1, residual ~0 (the identity guarantee, at D=192);
  (b) behavioral effect: does the generated 6-DOF trajectory change as intended
      (detour side = sign of max lateral excursion; bank = sign/size of summed
      orientation delta) as the commanded coordinate is swept.

Unconditional demonstration: does not depend on the policy being bottlenecked —
setting c controls the output by pass-through regardless.
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
BL.H = H
BL.HID = 256
K = 10
ITERS = int(os.environ.get("SNMVP_ITERS", "12000"))
GLAP_W = 0.5
SEED = 0


def traj_features(chunk_flat, scale):
    """From a scaled canonical chunk -> (detour_side, bank_x). detour_side = sign
    of the peak lateral(y) position excursion; bank_x = summed orientation-delta
    x-component (the banking axis)."""
    ch = chunk_flat.reshape(H, C) / scale
    pos = np.cumsum(ch[:, :3], axis=0)
    lat = pos[:, 1]
    detour = lat[np.argmax(np.abs(lat))]
    bank_x = ch[:, 3:].sum(axis=0)[0]
    return detour, bank_x


def main():
    d = np.load(HD.DATA)
    chunks, obs = d["chunks"].astype(float), d["obs"]
    S, N = chunks.shape[:2]
    scale = 1.0 / np.abs(chunks).mean()
    ch_s = chunks * scale
    X = ch_s.reshape(S, N, D)
    obs_dim = obs.shape[1]
    print(f"6-DOF steer demo: S={S} N={N} D={D} K={K} scale={scale:.1f}")

    # basis: grid-Laplacian, selected by cross-demo coherence (Sb/Sw)
    Sb, Sw = BL.covariances(ch_s, D)
    U = LB.basis_gridlap(Sb, Sw, K, H, C, GLAP_W)         # (D,K)

    # ---- INTERPRETABILITY: correlate pinned coordinates with named quantities ----
    bmean = X.mean(axis=1)                                # (S,D) per-scene mean chunk
    c = bmean @ U                                         # (S,K) pinned coordinates
    rad, s_o, r, lat = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    side = np.sign(lat)
    bank = HD.COUPLE * (-side) * np.clip(3.0 * (r + HD.OVERCLEAR), 0.0, 0.7)
    named = {"radius(endpoint)": rad, "lateral(side)": lat, "bank_angle": bank, "obst_r": r}
    corr = {}
    for k in range(K):
        ck = c[:, k]
        corr[k] = {nm: round(float(np.corrcoef(ck, v)[0, 1]), 2) for nm, v in named.items()}
    # pick the coordinate most correlated with side and with endpoint
    side_k = max(range(K), key=lambda k: abs(corr[k]["lateral(side)"]))
    end_k = max(range(K), key=lambda k: abs(corr[k]["radius(endpoint)"]))
    print("coordinate <-> named-quantity correlations:")
    for k in range(K):
        print(f"  c[{k}]: {corr[k]}")
    print(f"side-steering coordinate = c[{side_k}] (corr {corr[side_k]['lateral(side)']}); "
          f"endpoint coordinate = c[{end_k}] (corr {corr[end_k]['radius(endpoint)']})", flush=True)

    # the steering handle is a DIRECTION in the pinned subspace (side is encoded
    # redundantly across coordinates), not a single coordinate. Empirical side
    # direction = difference of mean-c between the two detour sides.
    d_side = c[side > 0].mean(0) - c[side < 0].mean(0)                    # (K,)
    d_hat = d_side / np.linalg.norm(d_side)
    t_demo = (c - c.mean(0)) @ d_hat                                      # demo projection
    print(f"side-direction |d_side|={np.linalg.norm(d_side):.2f}; "
          f"demo projection range [{t_demo.min():.1f}, {t_demo.max():.1f}]")

    # ---- train ONE executor with the U pin ----
    obs_flat = np.repeat(obs, N, axis=0); X_flat = X.reshape(-1, D)
    p = BL.train_exec(obs_flat, X_flat, U, SEED, ITERS, D, obs_dim)

    # ---- STEERABILITY: fix a scene, sweep the command ALONG the side direction ----
    mid = np.argmin(np.abs(rad - np.median(rad)))
    base = obs[mid][None, :]
    c_ref = c.mean(0)                                                     # neutral point in c-space
    lo, hi = np.percentile(t_demo, 8), np.percentile(t_demo, 92)
    sweep = np.linspace(lo, hi, 9)
    realized_t, detours, banks, resid = [], [], [], []
    for t in sweep:
        cmd = np.tile(c_ref + t * d_hat, (12, 1))                        # command in c-space
        gen = BL.rollout(p, np.tile(base, (12, 1)), U, cmd, SEED, D)       # (12,D)
        proj = (gen @ U) @ d_hat                                          # realized projection
        realized_t.append(float(proj.mean()))
        resid.append(float(np.mean(np.abs(proj - t))))
        feats = [traj_features(gen[i], scale) for i in range(12)]
        detours.append(float(np.mean([f[0] for f in feats])))
        banks.append(float(np.mean([f[1] for f in feats])))
    cmd_range = hi - lo
    slope = float(np.polyfit(sweep, realized_t, 1)[0])                    # pass-through, want ~1
    detour_slope = float(np.polyfit(sweep, detours, 1)[0])
    bank_slope = float(np.polyfit(sweep, banks, 1)[0])
    side_flips = detours[0] * detours[-1] < 0                             # detour flips sign across sweep
    print(f"STEERABILITY along side-direction (K-dim), fixed scene:")
    print(f"  pass-through: realized vs commanded slope = {slope:.3f} (want ~1), "
          f"mean|residual| = {np.mean(resid):.3f} over command range {cmd_range:.1f}")
    print(f"  behavioral: detour slope = {detour_slope:.4f}, bank slope = {bank_slope:.4f}, "
          f"detour flips sign across sweep = {side_flips}")
    print(f"  detours: {[round(x,3) for x in detours]}")
    print(f"  banks:   {[round(x,3) for x in banks]}")
    out = {"corr": corr, "side_k": side_k, "end_k": end_k,
           "passthrough_slope": round(slope, 3),
           "passthrough_resid": round(float(np.mean(resid)), 3),
           "command_range": round(cmd_range, 2),
           "detour_slope": round(detour_slope, 4), "bank_slope": round(bank_slope, 4),
           "detour_flips_sign": bool(side_flips),
           "detours": [round(x, 3) for x in detours],
           "banks": [round(x, 3) for x in banks]}
    json.dump(out, open(os.path.join(HERE, "steer6d_result.json"), "w"), indent=2)
    print("STEER6D_DONE=ok")


if __name__ == "__main__":
    main()
