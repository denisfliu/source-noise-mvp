"""The trained MDN head on REAL frames: accuracy vs real oracle, and sigma* vs the domain gap.

Sim-to-real question for the COMMAND SOURCE half of the pin (box 2026-08-13, b2lam03/K=5: head
was domain-faithful, R2 0.86 on real). Re-measured for the current recipe (gmsig, mh16 K=16,
sigma-conditioned) with the sigma dimension added: if sigma* on real frames is (a) still
error-tracking (rank corr) and (b) elevated only where real behavior actually diverges, the
trust dial has a credible transfer story — the head reads real pixels and flags what does not
transfer. Rows saved like sigma_phase_probe --save for side-by-side with sigrows_gmsig.npz.

  SNMVP_HEAD_GMM=1 ... python real_head_probe.py --ckpt <gmsig ckpt> --pin-u pin_U_mh16.npy
"""
import argparse
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import joint_head
from sigma_phase_probe import gmm_params, spearman

H, AD = 50, 32
REAL = {"left": range(0, 50), "right": range(50, 100)}
PROMPTS = joint_head.PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--eps-per-task", type=int, default=10)
    ap.add_argument("--frame-stride", type=int, default=12)
    ap.add_argument("--save", default=f"{RD}/sigrows_real_gmsig.npz")
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import json
    import openpi.policies.policy_config as PC
    import openpi.training.config as C
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt)
    U = np.load(a.pin_u).astype(np.float32)
    NS = json.load(open(os.path.expanduser(
        "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
    amean, astd = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(0)

    def oracle(ac, t):
        ch = np.zeros((H, AD), np.float32)
        m = min(H, len(ac) - t)
        ch[:m, :7] = (ac[t:t + m] - amean) / (astd + 1e-6)
        return ch.reshape(-1) @ U

    rows = {}
    for task, eps in REAL.items():
        rows[task] = []
        for e in rng.choice(list(eps), a.eps_per_task, replace=False):
            d = np.load(f"{RD}/data_gate_real/ep_{int(e):04d}.npz", allow_pickle=True)
            st = d["state"].astype(np.float32)
            ac = d["action"].astype(np.float32)
            T = len(st)
            raws, metas = [], []
            for t in range(0, T - 2, a.frame_stride):
                raws.append({"observation/image": r224(d["image"][t]),
                             "observation/wrist_image": r224(d["wrist"][t]),
                             "observation/state": st[t], "prompt": PROMPTS[task]})
                metas.append((t / T, t > T - H, int(e), t))
            for i in range(0, len(raws), 16):
                w, mu, logsig = gmm_params(policy, raws[i:i + 16])
                for k in range(len(w)):
                    j = int(w[k].argmax())
                    frac, is_stop, ee, tt = metas[i + k]
                    c_or = oracle(ac, tt)
                    pred = mu[k, j]
                    rows[task].append((frac, is_stop,
                                       float(np.linalg.norm(pred - c_or)),
                                       float(np.linalg.norm(np.exp(logsig[k, j]))),
                                       *pred.tolist(), *c_or.tolist()))
        print(f"[{task}] {len(rows[task])} rows", flush=True)

    np.savez_compressed(a.save, **{t: np.array(r, np.float64) for t, r in rows.items()})
    K = U.shape[1]
    print(f"\n{'task':8s} {'n':>4s} {'c-R2 vs real oracle':>20s} {'mean|err|':>10s} "
          f"{'mean|sig*|':>10s} {'corr(sig,err)':>13s} {'tail corr':>10s}")
    for task, rr in rows.items():
        rr = np.array(rr, np.float64)
        P, O = rr[:, 4:4 + K], rr[:, 4 + K:4 + 2 * K]
        r2 = 1 - ((O - P) ** 2).sum() / (((O - O.mean(0)) ** 2).sum() + 1e-9)
        tm = rr[:, 0] > 0.7
        print(f"{task:8s} {len(rr):4d} {r2:20.3f} {rr[:, 2].mean():10.2f} {rr[:, 3].mean():10.2f} "
              f"{spearman(rr[:, 3], rr[:, 2]):13.3f} {spearman(rr[tm, 3], rr[tm, 2]):10.3f}")
    print(f"saved rows -> {a.save}")


if __name__ == "__main__":
    main()
