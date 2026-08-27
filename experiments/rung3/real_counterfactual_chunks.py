"""Counterfactual + danger + zero-confidence pass over the real anchors (2026-08-27):
same real episodes/anchors as real_pred_chunks, three additions:
  1. CENTER-prompt fans (CFL on left eps, CFR on right): what does the pin give when the
     task is counterfactually swapped on real observations? (task-binding probe, visual)
  2. pin error per anchor |c_head - c_oracle|/cstd (side prompt AND center prompt vs the
     real continuation), phase-resolved — 'how far off are the right gate pins?'
  3. at the top-decile sigma* anchors (side prompt), a MAX-DISTRUST fan (sigma_serve=cap):
     what zero confidence looks like when acted on.

  SNMVP_HEAD=1 ... python real_counterfactual_chunks.py --ckpt <ck> --pin-u <U> --out <npz>
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
from real_pred_chunks import classify, PROMPTS, H, AD

CENTER_PROMPTS = {"left": "go through the center gate from the left and hover over the stuffed animal",
                  "right": "go through the center gate from the right and hover over the stuffed animal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--sigma-map", default=f"{RD}/sigma_map_gmsig3.json")
    ap.add_argument("--eps-per-side", type=int, default=8)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt,
                                      norm_stats=_nz.load(a.norm))
    U = np.load(a.pin_u).astype(np.float32)
    sm = json.load(open(a.sigma_map))
    xs, ys, cap = np.asarray(sm["sig_star"]), np.asarray(sm["sig_serve"]), float(sm["cap"])
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

    def gen(obs, c, sig_serve):
        g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
        res = policy.infer(obs, noise=noise, snmvp_sigma=sig_serve)
        return np.asarray(res["actions"], np.float32)[:H, :3]

    rows = {s: [] for s in ("left", "right")}   # per-anchor dicts
    counts = {"left": 0, "right": 0}
    for e in range(100):
        if all(counts[s] >= a.eps_per_side for s in counts):
            break
        d = np.load(f"{RD}/data_gate_real/ep_{e:04d}.npz", allow_pickle=True)
        st, ac = d["state"].astype(np.float32), d["action"].astype(np.float32)
        side = classify(st)
        if side is None or counts[side] >= a.eps_per_side:
            continue
        counts[side] += 1
        for t in range(10, len(st) - 5, a.stride):
            base = {"observation/image": r224(d["image"][t]),
                    "observation/wrist_image": r224(d["wrist"][t]),
                    "observation/state": st[t]}
            rec = {"e": e, "t": t, "frac": t / len(st), "anchor": st[t, :3].copy(),
                   "c_oracle": chunk_c(ac, t)}
            for tag, prompt in [("side", PROMPTS[side]), ("ctr", CENTER_PROMPTS[side])]:
                obs = dict(base, prompt=prompt)
                w, mu, sig = gmm_params(policy, [obs])
                j = int(w[0].argmax())
                c = mu[0, j]
                sstar = float(np.linalg.norm(sig[0, j]))
                sig_serve = float(np.clip(np.interp(sstar, xs, ys), 0.0, cap))
                act = gen(obs, c, sig_serve)
                rec[f"{tag}_c"] = c
                rec[f"{tag}_sig"] = sstar
                rec[f"{tag}_traj"] = np.concatenate([st[t, :3][None],
                                                     st[t, :3] + np.cumsum(act, axis=0)])
            rows[side].append(rec)
        print(f"ep{e:03d} [{side}] done", flush=True)

    # max-distrust fans at the top-decile side-prompt sigma* anchors
    for side in rows:
        sigs = np.array([r["side_sig"] for r in rows[side]])
        thr = np.quantile(sigs, 0.9)
        for r in rows[side]:
            if r["side_sig"] >= thr:
                d = np.load(f"{RD}/data_gate_real/ep_{r['e']:04d}.npz", allow_pickle=True)
                st = d["state"].astype(np.float32)
                obs = {"observation/image": r224(d["image"][r["t"]]),
                       "observation/wrist_image": r224(d["wrist"][r["t"]]),
                       "observation/state": st[r["t"]], "prompt": PROMPTS[side]}
                act = gen(obs, r["side_c"], cap)
                r["distrust_traj"] = np.concatenate([r["anchor"][None],
                                                     r["anchor"] + np.cumsum(act, axis=0)])

    np.savez(a.out, meta=json.dumps({s: len(rows[s]) for s in rows}),
             **{f"{s}_{i}_{k}": v for s in rows for i, r in enumerate(rows[s])
                for k, v in r.items() if isinstance(v, np.ndarray)},
             **{f"{s}_{i}_{k}": np.float32(v) for s in rows for i, r in enumerate(rows[s])
                for k, v in r.items() if isinstance(v, float)})
    print(f"saved -> {a.out}  left={len(rows['left'])} right={len(rows['right'])} anchors")


if __name__ == "__main__":
    main()
