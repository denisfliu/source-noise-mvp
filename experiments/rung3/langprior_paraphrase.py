"""Paraphrase probe for the enumeration-free language prior (Denis's question: does the
embedding interface buy synonym/paraphrase robustness?).

For held-out demo frames, compute c under (a) the canonical training prompt and (b) each
HELD-OUT paraphrase (gate_b_paraphrase.PARAPHRASES — never seen by anything in this line),
through the same lang_pool -> PCA -> prior path used at serving. Report, per task:
  - cos / relative L2 between paraphrase-c and canonical-c (command agreement)
  - agreement relative to the WRONG-task canonical c (the discrimination baseline: a
    paraphrase must stay far from the other task's command)
  - c-R2 of paraphrase-c against the true demo c
A useful head keeps paraphrase-c close to canonical-c and far from the other task's.
CPU-inference after a GPU feature pass; run with CUDA_VISIBLE_DEVICES set.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
from gate_b_paraphrase import PARAPHRASES

RD = os.path.dirname(os.path.abspath(__file__))
HFB = "/home/ubuntu/hf_bundle/gate-drone-pi0"
CKPT = "/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999"
H, AD, K = 50, 32, 5
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = LEFT.replace("left", "right")
CFL = "go through the center gate from the left and hover over the stuffed animal"
CFR = "go through the center gate from the right and hover over the stuffed animal"
TASKS = [CFL, CFR, LEFT, RIGHT]
FRAMES_PER_EP, EPS_PER_TASK = 4, 5


def r224(img):
    return np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC)).astype(np.uint8)


def main():
    import openpi.training.config as C
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as NZ
    from openpi.transforms import NormStats

    ns = NZ.load(f"{HFB}/assets/gate_nav")
    def pads(nsd, dim):
        out = {}
        for k, s in nsd.items():
            n = np.asarray(s.mean).shape[-1]
            if n >= dim: out[k] = s; continue
            p = dim - n
            ext = lambda a, f: None if a is None else np.concatenate([np.asarray(a, np.float32), np.full(p, f, np.float32)])
            out[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
        return out
    cfg = C.get_config("pi0_gate")
    nsp = pads(ns, cfg.model.action_dim)
    policy = PC.create_trained_policy(cfg, CKPT, norm_stats=nsp)
    amean = np.asarray(nsp["actions"].mean); astd = np.asarray(nsp["actions"].std)
    U = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32)

    d = torch.load(f"{RD}/langprior_rrr.pt", map_location="cpu", weights_only=False)
    layers, din = [], d["in_dim"]
    for hdim in d["hidden"]:
        layers += [nn.Linear(din, hdim), nn.SiLU()]; din = hdim
    layers += [nn.Linear(din, K)]
    net = nn.Sequential(*layers); net.load_state_dict(d["state_dict"]); net.eval()
    mu, sd, Em, P = d["mu"], d["sd"], d["Em"], d["P"]

    def c_of(chunk7):
        L = len(chunk7); ch = np.zeros((H, AD), np.float32); m = min(L, H)
        ch[:m, :7] = (chunk7[:m] - amean[:7]) / (astd[:7] + 1e-6)
        if m < H: ch[m:, :7] = ch[m - 1, :7]
        return ch.reshape(-1) @ U

    def predict(raws, states):
        E = []
        for j in range(0, len(raws), 8):
            E.append(gc.lang_pool(policy, raws[j:j + 8]))
        E = np.concatenate(E, 0)
        e64 = (E - Em) @ P
        ms = np.stack([np.asarray(policy._input_transform(dict(r))["state"]).reshape(-1) for r in raws])
        x = np.concatenate([ms, e64], 1).astype(np.float32)
        with torch.no_grad():
            return net(torch.tensor((x - mu) / sd)).numpy()

    rng = np.random.default_rng(0)
    held = set(rng.permutation(200)[160:].tolist())  # same split as training
    print(f"held episodes: {len(held)}", flush=True)

    for ti, task in enumerate(TASKS):
        if task not in PARAPHRASES:
            continue
        eps = [e for e in sorted(held) if e // 50 == ti][:EPS_PER_TASK]
        if not eps:
            continue
        raws, true_c = [], []
        for e in eps:
            dd = np.load(f"{RD}/data_gate_synth/ep_{e:04d}.npz", allow_pickle=True)
            T = len(dd["state"])
            for t in np.linspace(0, T - H - 1, FRAMES_PER_EP).astype(int):
                raws.append({"observation/image": r224(dd["image"][t]),
                             "observation/wrist_image": r224(dd["wrist"][t]),
                             "observation/state": dd["state"][t].astype(np.float32), "prompt": task})
                true_c.append(c_of(dd["action"][t:t + H].astype(np.float32)))
        true_c = np.array(true_c)
        canon = predict(raws, None)
        other = TASKS[3] if task == TASKS[2] else TASKS[2]
        other_c = predict([dict(r, prompt=other) for r in raws], None)
        base_gap = np.linalg.norm(canon - other_c, axis=1).mean()
        print(f"\n== {task.split('gate')[1][:24]!r}  frames={len(raws)}  "
              f"canonical-vs-other-task command gap {base_gap:.2f}", flush=True)
        rows = []
        for p in PARAPHRASES[task]:
            pc = predict([dict(r, prompt=p) for r in raws], None)
            cos = float(np.mean(np.sum(pc * canon, 1) / (np.linalg.norm(pc, axis=1) * np.linalg.norm(canon, axis=1) + 1e-9)))
            rel = float(np.linalg.norm(pc - canon, axis=1).mean() / (base_gap + 1e-9))
            r2 = float(1 - ((true_c - pc) ** 2).sum() / (((true_c - true_c.mean(0)) ** 2).sum() + 1e-9))
            rows.append((rel, cos, r2, p))
        rows.sort()
        for rel, cos, r2, p in rows:
            flag = "ok " if rel < 0.5 else "BAD"
            print(f"  {flag} drift {rel:.2f}x  cos {cos:+.3f}  c-R2 {r2:+.3f}  {p[:58]!r}", flush=True)
        arr = np.array([r[0] for r in rows])
        print(f"  -- median drift {np.median(arr):.2f}x, fraction under 0.5x: {(arr < 0.5).mean():.2f}", flush=True)
    print("\nPARAPHRASE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
