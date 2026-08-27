"""Synthetic pins served on real observations (2026-08-27, Denis: 'use synthetic pins in
real directly — what happens at the right gate?'). For real anchors, find the matched synth
state (position+yaw nearest, as in pin_gap_probe) and serve the REAL obs with the SYNTH
state's oracle c at sigma=0. Controls on the same anchors: real-oracle pin (ceiling) and
head pin (status quo). Metrics: heading error vs real continuation, chunk speed, and for
pre-right-gate anchors the aperture crossing s-coordinate + min post distance.

  SNMVP_HEAD=1 ... python synthpin_in_real.py --ckpt <ck> --out <npz>
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
from real_angle_fix import classify

H, AD = 50, 32
PROMPTS = {"left": "go through the gate on the left and hover over the stuffed animal",
           "right": "go through the gate on the right and hover over the stuffed animal"}
GA, GB = np.array([0.195, -1.348]), np.array([0.924, -0.952])   # right gate posts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import gate_ctx_common as gc
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    cfg = C.get_config("pi0_gate")
    policy = PC.create_trained_policy(cfg, a.ckpt,
                                      norm_stats=gc.pad_norm_stats(_nz.load(a.norm),
                                                                   cfg.model.action_dim))
    U = np.load(a.pin_u).astype(np.float32)
    NS = json.load(open(os.path.expanduser(
        "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
    amean, astd = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(0)

    def chunk_c(ac, t):
        ch = np.zeros((H, AD), np.float32)
        m = min(H, len(ac) - t)
        ch[:m, :7] = (ac[t:t + m] - amean) / (astd + 1e-6)
        return ch.reshape(-1) @ U

    # synth frame index (states + oracle c refs), right/left episodes of synth3
    synth = []
    for e in range(100, 200, 2):     # synth L eps 100-149, R eps 150-199
        d = np.load(f"{RD}/data_gate_synth3/ep_{e:04d}.npz", allow_pickle=True)
        st, ac = d["state"].astype(np.float32), d["action"].astype(np.float32)
        for t in range(0, len(st) - H - 1, 5):
            synth.append((st[t], chunk_c(ac, t)))
    SS = np.stack([s for s, _ in synth])
    print(f"synth index {len(synth)}", flush=True)

    def gen(obs, c):
        g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
        return np.asarray(policy.infer(obs, noise=noise, snmvp_sigma=0.0)["actions"], np.float32)[:H]

    rows = []
    counts = {"left": 0, "right": 0}
    for e in range(100):
        if counts["right"] >= 8 and counts["left"] >= 4:
            break
        d = np.load(f"{RD}/data_gate_real/ep_{e:04d}.npz", allow_pickle=True)
        st, ac = d["state"].astype(np.float32), d["action"].astype(np.float32)
        side = classify(st)
        if side is None:
            continue
        cap_n = 8 if side == "right" else 4
        if counts[side] >= cap_n:
            continue
        counts[side] += 1
        for t in range(10, len(st) - 30, 35):
            s0 = st[t]
            dp = np.linalg.norm(SS[:, :3] - s0[:3], axis=1)
            dy = np.abs(np.angle(np.exp(1j * (SS[:, 3] - s0[3]))))
            cand = np.where((dp < 0.35) & (dy < 0.6))[0]
            if not len(cand):
                continue
            j = int(cand[np.argmin(dp[cand] + 0.3 * dy[cand])])
            obs = {"observation/image": r224(d["image"][t]),
                   "observation/wrist_image": r224(d["wrist"][t]),
                   "observation/state": s0, "prompt": PROMPTS[side]}
            w, mu, _ = gmm_params(policy, [obs])
            c_head = mu[0, int(w[0].argmax())]
            rec = {"side": side, "e": e, "t": t, "frac": t / len(st),
                   "anchor": s0[:3].copy(), "match_d": float(dp[j])}
            m = min(25, len(st) - 1 - t)
            rec["hr"] = float(np.arctan2(*(st[t + m, :2] - st[t, :2])[::-1]))
            for tag, c in [("synthpin", synth[j][1]), ("realpin", chunk_c(ac, t)),
                           ("headpin", c_head)]:
                acts = gen(obs, c.astype(np.float32))
                traj = s0[:3] + np.cumsum(acts[:, :3], axis=0)
                rec[f"{tag}_traj"] = np.concatenate([s0[:3][None], traj])
                rec[f"{tag}_h"] = float(np.arctan2(*np.sum(acts[:m, :2], axis=0)[::-1]))
                rec[f"{tag}_len"] = float(np.sum(np.linalg.norm(acts[:, :3], axis=1)))
            rows.append(rec)
        print(f"ep{e:03d} [{side}] done ({len(rows)} rows)", flush=True)

    np.savez(a.out, meta=json.dumps([{k: v for k, v in r.items()
                                      if not isinstance(v, np.ndarray)} for r in rows]),
             **{f"arr_{i}_{k}": v for i, r in enumerate(rows)
                for k, v in r.items() if isinstance(v, np.ndarray)})
    print(f"saved -> {a.out}  ({len(rows)} anchors)")


if __name__ == "__main__":
    main()
