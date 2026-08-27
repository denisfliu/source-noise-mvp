"""Actionable sim-to-real distance in PIN SPACE (2026-08-26, Denis's question: does real
data predict similar pins to sim data?). Matched-state pairing: for sampled real frames,
find the nearest synth frame (position + yaw), then decompose the gap in cstd units:

  behavior gap    |c_oracle_real - c_oracle_synth|   how differently the datasets FLY from
                                                     the same state (planner artifact ->
                                                     fix in the course generator)
  prediction gap  |c_head(real) - c_head(synth)|     perception-side domain shift, judged
                                                     against the in-domain baseline
  baseline        |c_head - c_oracle| per domain     the head's own error floor
  detectability   sigma* real vs synth               does the trust dial KNOW it is
                                                     off-domain? (online gap detector)

  SNMVP_HEAD=1 ... python pin_gap_probe.py --ckpt <ck> --pin-u <U>
"""
import argparse
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import joint_head
from sigma_phase_probe import gmm_params

H, AD = 50, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--pairs", type=int, default=80)
    ap.add_argument("--max-dist", type=float, default=0.35, help="match gate: position (m)")
    ap.add_argument("--max-dyaw", type=float, default=0.6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt,
                                      norm_stats=_nz.load(a.norm))
    U = np.load(a.pin_u).astype(np.float32)
    K = U.shape[1]
    NS = json.load(open(os.path.expanduser(
        "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
    amean, astd = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)

    def chunk_c(ac, t):
        ch = np.zeros((H, AD), np.float32)
        m = min(H, len(ac) - t)
        ch[:m, :7] = (ac[t:t + m] - amean) / (astd + 1e-6)
        return ch.reshape(-1) @ U

    # index all synth frames (states + episode/frame refs); subsample for tractability
    synth = []
    for e in range(0, 200, 2):
        d = np.load(f"{RD}/data_gate_synth3/ep_{e:04d}.npz", allow_pickle=True)
        st = d["state"].astype(np.float32)
        for t in range(0, len(st) - H - 1, 6):
            synth.append((e, t, st[t]))
    S = np.stack([s for _, _, s in synth])
    print(f"synth index: {len(synth)} frames", flush=True)

    rng = np.random.default_rng(0)
    Cs = []
    for i in rng.permutation(len(synth))[:400]:
        e, t, _ = synth[i]
        d = np.load(f"{RD}/data_gate_synth3/ep_{e:04d}.npz", allow_pickle=True)
        Cs.append(chunk_c(d["action"].astype(np.float32), t))
    cstd = np.std(np.stack(Cs), axis=0)
    cstd_n = float(np.linalg.norm(cstd))

    PROMPT = "fly through the gate and hover over the stuffed animal"
    rows = []
    tried = 0
    for e in rng.permutation(100):
        if len(rows) >= a.pairs:
            break
        dr = np.load(f"{RD}/data_gate_real/ep_{int(e):04d}.npz", allow_pickle=True)
        str_, acr = dr["state"].astype(np.float32), dr["action"].astype(np.float32)
        T = len(str_)
        for frac in (0.2, 0.5, 0.8):
            if len(rows) >= a.pairs:
                break
            t = int(frac * (T - H - 1))
            s = str_[t]
            dp = np.linalg.norm(S[:, :3] - s[:3], axis=1)
            dy = np.abs(np.angle(np.exp(1j * (S[:, 3] - s[3]))))
            cand = np.where((dp < a.max_dist) & (dy < a.max_dyaw))[0]
            tried += 1
            if not len(cand):
                continue
            j = cand[np.argmin(dp[cand] + 0.3 * dy[cand])]
            es, ts, ss = synth[j]
            ds = np.load(f"{RD}/data_gate_synth3/ep_{es:04d}.npz", allow_pickle=True)
            obs_r = {"observation/image": r224(dr["image"][t]),
                     "observation/wrist_image": r224(dr["wrist"][t]),
                     "observation/state": s, "prompt": PROMPT}
            obs_s = {"observation/image": r224(ds["image"][ts]),
                     "observation/wrist_image": r224(ds["wrist"][ts]),
                     "observation/state": ds["state"].astype(np.float32)[ts], "prompt": PROMPT}
            w, mu, sig = gmm_params(policy, [obs_r, obs_s])
            ch = mu[np.arange(2), w.argmax(1)]
            sg = np.linalg.norm(sig[np.arange(2), w.argmax(1)], axis=1)
            rows.append({
                "frac": frac, "dist": float(dp[j]), "dyaw": float(dy[j]),
                "co_r": chunk_c(acr, t), "co_s": chunk_c(ds["action"].astype(np.float32), ts),
                "ch_r": ch[0], "ch_s": ch[1], "sg_r": float(sg[0]), "sg_s": float(sg[1]),
            })
            if len(rows) % 20 == 0:
                print(f"  {len(rows)} pairs...", flush=True)

    print(f"\npairs={len(rows)} (match attempts {tried}); match dist median "
          f"{np.median([r['dist'] for r in rows]):.2f} m, dyaw "
          f"{np.median([r['dyaw'] for r in rows]):.2f} rad; cstd={cstd_n:.1f}")
    def L(v):
        return float(np.linalg.norm(v))
    beh = [L(r["co_r"] - r["co_s"]) / cstd_n for r in rows]
    prd = [L(r["ch_r"] - r["ch_s"]) / cstd_n for r in rows]
    b_r = [L(r["ch_r"] - r["co_r"]) / cstd_n for r in rows]
    b_s = [L(r["ch_s"] - r["co_s"]) / cstd_n for r in rows]
    print(f"\nGAPS (/cstd, median [iqr]):")
    for name, v in [("behavior  |c_oracle_r - c_oracle_s|", beh),
                    ("prediction |c_head_r - c_head_s|", prd),
                    ("baseline real  |c_head - c_oracle|", b_r),
                    ("baseline synth |c_head - c_oracle|", b_s)]:
        q = np.percentile(v, [25, 50, 75])
        print(f"  {name:38s} {q[1]:.3f} [{q[0]:.3f}-{q[2]:.3f}]")
    print(f"\nsigma* (trust dial): real median {np.median([r['sg_r'] for r in rows]):.2f}  "
          f"synth median {np.median([r['sg_s'] for r in rows]):.2f}  "
          f"(elevated-on-real = the dial detects the domain)")
    print("\nper-dim |c_head_r - c_head_s| median (/cstd_dim):")
    D = np.stack([np.abs(r["ch_r"] - r["ch_s"]) for r in rows])
    for k in range(K):
        print(f"  dim {k:2d}: {np.median(D[:, k]) / cstd[k]:.3f}")
    print("\nby phase (behavior / prediction gap):")
    for frac in (0.2, 0.5, 0.8):
        sel = [i for i, r in enumerate(rows) if r["frac"] == frac]
        if sel:
            print(f"  frac {frac}: n={len(sel)}  beh {np.median([beh[i] for i in sel]):.3f}  "
                  f"pred {np.median([prd[i] for i in sel]):.3f}")
    if a.out:
        np.savez(a.out, **{k: np.stack([np.atleast_1d(r[k]) for r in rows])
                           for k in ("co_r", "co_s", "ch_r", "ch_s")},
                 sg=np.array([[r["sg_r"], r["sg_s"]] for r in rows]), cstd=cstd)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
