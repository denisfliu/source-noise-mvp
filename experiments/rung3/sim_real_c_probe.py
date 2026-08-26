"""Sim-vs-real c consistency at matched states (Denis, 2026-08-12): if we predict c on a REAL
frame, is it the same c the head predicts at the SAME SPOT in sim?

For each sampled real frame (data_gate_real), find the nearest same-task synth frame
(data_gate_synth) by position with a direction-consistency requirement (heading dot > 0.3 —
position alone aliases outbound/return phases; clog_analysis lesson). Run the checkpoint's own
command head on BOTH observations and compare:

  cos(c_real, c_sim), ||c_real - c_sim|| in units of std(oracle c)   <- representation consistency
  R2(c_real vs oracle_real), R2(c_sim vs oracle_sim)                 <- accuracy on each domain

Oracle c uses the zero-pad chunk convention (what the joint arms train on). Matches farther than
MAXD are dropped (reported). Usage:
  SNMVP_PIN_U=... python sim_real_c_probe.py --ckpt <joint ckpt> [--n-eps 30 --stride 12]
"""
import argparse
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
H, MAXD = 50, 0.5


def heading(ac, t, k=10):
    v = ac[t:t + k, :3].sum(0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def load_eps(d):
    meta = json.load(open(f"{RD}/{d}/meta.json"))
    eps = []
    for k in sorted(meta):
        z = np.load(f"{RD}/{d}/{k}.npz", allow_pickle=True)
        eps.append({"task": meta[k]["task"], "lang": meta[k]["lang"],
                    "image": z["image"], "wrist": z["wrist"],
                    "state": z["state"].astype(np.float32), "action": z["action"].astype(np.float32)})
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_gate_rrr_k5.npy")
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--norm", default="/home/ubuntu/hf_bundle/gate-drone-pi0/assets/gate_nav")
    ap.add_argument("--n-eps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=12)
    a = ap.parse_args()

    import joint_head
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.training.config as C
    import openpi.training.weight_loaders  # noqa: F401  (keep import order identical to check())
    policy = PC.create_trained_policy(C.get_config(a.config), a.ckpt)
    U = np.load(a.pin_u).astype(np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)

    def oracle(ep, t):
        import openpi.transforms as T
        from openpi.shared.normalize import load as load_ns
        ns = joint_head._oracle_c  # reuse its normalizer plumbing via direct call on synth only
        # zero-pad chunk in normalized units, same as joint training targets
        AD = C.get_config(a.config).model.action_dim
        from openpi.shared.normalize import NormStats
        d = load_ns(a.norm)
        o = {}
        for k, s in d.items():
            n = len(s.mean)
            if n >= AD:
                o[k] = s; continue
            p = AD - n
            ext = lambda x, f: None if x is None else np.concatenate(
                [np.asarray(x, np.float32), np.full(p, f, np.float32)])
            o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
        nrm = T.Normalize(o, use_quantiles=False)
        ac = ep["action"]
        ch = np.zeros((H, AD), np.float32)
        k = min(H, len(ac) - t)
        ch[:k, :7] = ac[t:t + k]
        return (nrm({"actions": ch})["actions"].reshape(-1)) @ U

    real = load_eps("data_gate_real")
    synth = load_eps("data_gate_synth")
    rng = np.random.default_rng(0)
    # per-task synth banks: (positions, headings, ep index, t)
    bank = {}
    for si, ep in enumerate(synth):
        for t in range(0, len(ep["state"]) - H - 1, 4):
            h = heading(ep["action"], t)
            if h is None:
                continue
            bank.setdefault(ep["task"], []).append((ep["state"][t, :3], h, si, t))

    rows = []
    picks = rng.permutation(len(real))[:a.n_eps]
    for ri in picks:
        ep = real[int(ri)]
        if ep["task"] not in bank:
            continue
        B = bank[ep["task"]]
        BP = np.array([b[0] for b in B]); BH = np.array([b[1] for b in B])
        for t in range(0, len(ep["state"]) - H - 1, a.stride):
            h = heading(ep["action"], t)
            if h is None:
                continue
            d = np.linalg.norm(BP - ep["state"][t, :3], axis=1)
            ok = (BH @ h) > 0.3
            if not ok.any():
                continue
            d[~ok] = 1e9
            j = int(d.argmin())
            if d[j] > MAXD:
                continue
            rows.append((int(ri), t, B[j][2], B[j][3], float(d[j])))
    print(f"matched {len(rows)} frame pairs (match dist mean "
          f"{np.mean([r[4] for r in rows]):.3f} m)", flush=True)

    def batch_c(obs):
        out = []
        for i in range(0, len(obs), 8):
            out.append(joint_head.head_c(policy, obs[i:i + 8]))
        return np.concatenate(out, 0)

    raws_r = [{"observation/image": r224(real[ri]["image"][t]),
               "observation/wrist_image": r224(real[ri]["wrist"][t]),
               "observation/state": real[ri]["state"][t], "prompt": real[ri]["lang"]}
              for ri, t, si, st, _ in rows]
    raws_s = [{"observation/image": r224(synth[si]["image"][st]),
               "observation/wrist_image": r224(synth[si]["wrist"][st]),
               "observation/state": synth[si]["state"][st], "prompt": synth[si]["lang"]}
              for ri, t, si, st, _ in rows]
    cr, cs = batch_c(raws_r), batch_c(raws_s)
    orl = np.array([oracle(real[ri], t) for ri, t, si, st, _ in rows])
    osy = np.array([oracle(synth[si], st) for ri, t, si, st, _ in rows])

    def r2(p, y):
        return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))

    task = np.array([real[ri]["task"] for ri, *_ in rows])
    cstd = orl.std(0)
    for tk in sorted(set(task.tolist())):
        m = task == tk
        cos = np.mean([np.dot(cr[i], cs[i]) / (np.linalg.norm(cr[i]) * np.linalg.norm(cs[i]) + 1e-9)
                       for i in np.where(m)[0]])
        gap = np.linalg.norm((cr[m] - cs[m]) / cstd, axis=1).mean()
        print(f"task {tk}: n={m.sum():4d}  cos(c_real,c_sim)={cos:+.3f}  "
              f"|c_real-c_sim|/std={gap:.3f}  "
              f"R2(real vs oracle_real)={r2(cr[m], orl[m]):+.3f}  "
              f"R2(sim vs oracle_sim)={r2(cs[m], osy[m]):+.3f}", flush=True)
    np.savez(f"{RD}/sim_real_c_probe.npz", c_real=cr, c_sim=cs, oracle_real=orl,
             oracle_sim=osy, task=task, rows=np.array([r[:4] for r in rows]))
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
