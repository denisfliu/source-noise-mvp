"""Retrain the task selector on the SERVING checkpoint's features (v3 finding: the
gate_both_pin-trained selector confidently mislatches LEFT episodes as CFR on live
gate_both_pin_rrr features — checkpoint domain shift). One process: extract post-fusion
(ctx_pool) features for all 200 synth episodes with authoritative index-map labels,
true + within-family-swapped prompts (dissociates language from scene), grouped episode
split, retrain the 2048-128-4 GELU head, export task_selector_rrr.npz.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc

RD = os.path.dirname(os.path.abspath(__file__))
HFB = "/home/ubuntu/hf_bundle/gate-drone-pi0"
CKPT = "/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999"
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = LEFT.replace("left", "right")
CFL = "go through the center gate from the left and hover over the stuffed animal"
CFR = "go through the center gate from the right and hover over the stuffed animal"
TASKS = [CFL, CFR, LEFT, RIGHT]
SWAP = {CFL: CFR, CFR: CFL, LEFT: RIGHT, RIGHT: LEFT}
STRIDE, BS = 10, 8

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
    policy = PC.create_trained_policy(cfg, CKPT, norm_stats=pads(ns, cfg.model.action_dim))

    cache = f"{RD}/selector_rrr_feats.npz"
    if os.path.exists(cache):
        z = np.load(cache); X, y, ep_ix = z["X"], z["y"], z["ep"]
        print("loaded cache", X.shape, flush=True)
    else:
        rows = []  # (ep_index, t, prompt, label)
        for i in range(200):
            d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
            true_p = TASKS[i // 50]
            frames = [(t, r224(d["image"][t]), r224(d["wrist"][t]), d["state"][t].astype(np.float32))
                      for t in range(0, len(d["state"]), STRIDE)]
            for t, im, wr, st in frames:
                for p in (true_p, SWAP[true_p]):
                    rows.append((i, {"observation/image": im, "observation/wrist_image": wr,
                                     "observation/state": st, "prompt": p}, TASKS.index(p)))
            if i % 20 == 0:
                print(f"  frames prepped ep {i}", flush=True)
        X, y, ep_ix = [], [], []
        for j in range(0, len(rows), BS):
            batch = rows[j:j + BS]
            X.append(gc.ctx_pool(policy, [r[1] for r in batch]))
            y += [r[2] for r in batch]; ep_ix += [r[0] for r in batch]
            if j % (BS * 50) == 0:
                print(f"  feat {j}/{len(rows)}", flush=True)
        X = np.concatenate(X, 0).astype(np.float32)
        y = np.array(y, np.int64); ep_ix = np.array(ep_ix, np.int64)
        np.savez_compressed(cache, X=X, y=y, ep=ep_ix)
        print("extracted", X.shape, flush=True)

    rng = np.random.default_rng(0)
    tr_eps = set(rng.permutation(200)[:160].tolist())
    tr = np.array([e in tr_eps for e in ep_ix])
    mu = X[tr].mean(0); sg = X[tr].std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sg, dtype=torch.float32); yt = torch.tensor(y)
    net = nn.Sequential(nn.Linear(2048, 128), nn.GELU(approximate="tanh"), nn.Linear(128, 4))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    tri = np.where(tr)[0]
    for ep in range(40):
        perm = np.random.permutation(tri)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]; opt.zero_grad()
            nn.functional.cross_entropy(net(Xn[b]), yt[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xn).argmax(1).numpy()
    acc = (pred[~tr] == y[~tr]).mean()
    per = {TASKS[k].split("gate")[1][:14]: float((pred[(~tr) & (y == k)] == k).mean()) for k in range(4)}
    print(f"held acc {acc:.3f} per-task {per}", flush=True)
    L = [m for m in net if isinstance(m, nn.Linear)]
    np.savez(f"{RD}/task_selector_rrr.npz", mu=mu.astype(np.float32), sg=sg.astype(np.float32),
             W1=L[0].weight.detach().numpy().T, b1=L[0].bias.detach().numpy(),
             W2=L[1].weight.detach().numpy().T, b2=L[1].bias.detach().numpy(),
             tasks=np.array(TASKS))
    print("SELECTOR_RRR_DONE", flush=True)

if __name__ == "__main__":
    main()
