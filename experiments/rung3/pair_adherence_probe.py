"""PAIR-RESOLVED adherence (2026-08-24): the pooled probe hid the decision-relevant number.
For frames from LEFT and RIGHT episodes at the compound-switch-relevant phase (post-gate,
frac 0.25-0.6), measure ||c(prompt_a) - c(prompt_b)||/cstd for the specific pairs:

  atomic contrast   : left vs right prompt          (healthy baseline contrast)
  compound contrast : atomic vs its compound prompt (the contrast the switch needs;
                      ~0 = prompt-neighborhood collapse)
  cross-compound    : compound-left vs compound-right

  SNMVP_HEAD_GMM=1 ... python pair_adherence_probe.py --ckpt <ck> --pin-u <U>
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

H = 50
PL = joint_head.PROMPTS["left"]
PR = joint_head.PROMPTS["right"]
CL = "go through the gate on the left, then through the center gate and hover over the stuffed animal"
CR = "go through the gate on the right, then through the center gate and hover over the stuffed animal"
PROMPTS = [PL, PR, CL, CR]
PAIRS = {"left-vs-right": (0, 1), "left-vs-cmpL": (0, 2), "right-vs-cmpR": (1, 3),
         "cmpL-vs-cmpR": (2, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--data-dir", default="data_gate_synth3")
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt,
                                      norm_stats=_nz.load(a.norm))
    U = np.load(a.pin_u).astype(np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(0)
    NS = json.load(open(os.path.expanduser(
        "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
    amean, astd = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
    Cs = []
    for e in range(0, 200, 10):
        d = np.load(f"{RD}/{a.data_dir}/ep_{e:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32)
        for t in range(0, len(ac), 16):
            ch = np.zeros((H, 32), np.float32)
            m = min(H, len(ac) - t)
            ch[:m, :7] = (ac[t:t + m] - amean) / (astd + 1e-6)
            Cs.append(ch.reshape(-1) @ U)
    cstd = float(np.linalg.norm(np.std(np.stack(Cs), axis=0)))

    acc = {k: {"start": [], "switch": []} for k in PAIRS}
    for eps in (range(100, 150), range(150, 200)):     # left episodes, right episodes
        for e in rng.choice(list(eps), 5, replace=False):
            d = np.load(f"{RD}/{a.data_dir}/ep_{int(e):04d}.npz", allow_pickle=True)
            st = d["state"].astype(np.float32)
            T = len(st)
            for t in list(range(0, int(0.1 * T), 8)) + list(range(int(0.25 * T), int(0.6 * T), 12)):
                phase = "start" if t < 0.1 * T else "switch"
                raws = [{"observation/image": r224(d["image"][t]),
                         "observation/wrist_image": r224(d["wrist"][t]),
                         "observation/state": st[t], "prompt": p} for p in PROMPTS]
                w, mu, _ = gmm_params(policy, raws)
                cs = mu[np.arange(len(w)), w.argmax(1)]
                for name, (i, j) in PAIRS.items():
                    acc[name][phase].append(float(np.linalg.norm(cs[i] - cs[j])) / cstd)
    print(f"cstd={cstd:.2f}   |dc|/cstd, phase = start (frac<0.1) vs switch (0.25-0.6)")
    print(f"{'pair':16s} {'start':>8s} {'switch':>8s}")
    for name in PAIRS:
        print(f"{name:16s} {np.mean(acc[name]['start']):8.3f} {np.mean(acc[name]['switch']):8.3f}")


if __name__ == "__main__":
    main()
