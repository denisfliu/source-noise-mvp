"""TEXT-ADHERENCE probe (Denis, 2026-08-24): does the command actually move when ONLY the
prompt changes? The compound failure showed the head can drop language wherever state is a
sufficient statistic on the training data; adherence must be a measured model property, not an
assumption. For sampled frames across tasks and phases, run the head under ALL FOUR task
prompts (+ the two compound prompts) and report, per phase bin:

  |dc|/cstd   mean over prompt pairs of ||mu*(prompt_i) - mu*(prompt_j)|| in units of the
              training c-std norm — 0 means the language channel is dead at those states.

Rows saved per arm for cross-arm comparison; the acceptance idea is a floor on mid-flight
adherence joining the readout gate.

  SNMVP_HEAD_GMM=1 ... python text_adherence_probe.py --ckpt <ck> --pin-u <U> \
      [--data-dir data_gate_synth3] [--save rows.npz]
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
TASKS = {"cfl": range(0, 50), "cfr": range(50, 100), "left": range(100, 150), "right": range(150, 200)}
PROMPTS = [
    joint_head.PROMPTS["left"], joint_head.PROMPTS["right"],
    joint_head.PROMPTS["center_from_left"], joint_head.PROMPTS["center_from_right"],
    "go through the gate on the left, then through the center gate and hover over the stuffed animal",
    "go through the gate on the right, then through the center gate and hover over the stuffed animal",
]
BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--data-dir", default=os.environ.get("SNMVP_DATA_DIR", "data_gate_synth3"))
    ap.add_argument("--eps-per-task", type=int, default=4)
    ap.add_argument("--frame-stride", type=int, default=20)
    ap.add_argument("--save", default="")
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

    # c-std normalizer from this data dir
    NS = json.load(open(os.path.expanduser(
        "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
    amean, astd = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
    Cs = []
    for e in range(0, 200, 5):
        d = np.load(f"{RD}/{a.data_dir}/ep_{e:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32)
        for t in range(0, len(ac), 16):
            ch = np.zeros((H, 32), np.float32)
            m = min(H, len(ac) - t)
            ch[:m, :7] = (ac[t:t + m] - amean) / (astd + 1e-6)
            Cs.append(ch.reshape(-1) @ U)
    cstd = float(np.linalg.norm(np.std(np.stack(Cs), axis=0)))

    rows = []  # (task_id, frac, adherence)
    for ti, (task, eps) in enumerate(TASKS.items()):
        for e in rng.choice(list(eps), a.eps_per_task, replace=False):
            d = np.load(f"{RD}/{a.data_dir}/ep_{int(e):04d}.npz", allow_pickle=True)
            st = d["state"].astype(np.float32)
            T = len(st)
            for t in range(0, T - 2, a.frame_stride):
                raws = [{"observation/image": r224(d["image"][t]),
                         "observation/wrist_image": r224(d["wrist"][t]),
                         "observation/state": st[t], "prompt": p} for p in PROMPTS]
                w, mu, _ = gmm_params(policy, raws)
                cs = mu[np.arange(len(w)), w.argmax(1)]      # argmax-mode command per prompt
                dif = [np.linalg.norm(cs[i] - cs[j]) for i in range(len(cs))
                       for j in range(i + 1, len(cs))]
                rows.append((ti, t / T, float(np.mean(dif)) / cstd))
        print(f"[{task}] done", flush=True)

    rr = np.array(rows)
    print(f"\ncstd={cstd:.2f}  adherence = mean pairwise |dc| across 6 prompts, /cstd")
    print(f"{'phase':12s} {'adherence':>10s}  n")
    for lo, hi in BINS:
        m = (rr[:, 1] >= lo) & (rr[:, 1] < hi)
        print(f"[{lo:.2f},{hi:.2f})  {rr[m, 2].mean():10.3f}  {int(m.sum())}")
    if a.save:
        np.savez_compressed(a.save, rows=rr)
        print(f"saved -> {a.save}")


if __name__ == "__main__":
    main()
