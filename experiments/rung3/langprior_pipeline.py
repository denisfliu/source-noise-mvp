"""Enumeration-free command prior (Denis, 2026-08-08: "we shouldn't assume we know what
the tasks are beforehand"): c = MLP([model_state, e64]) where e64 = PCA-64 of the
post-fusion LANGUAGE-token embedding (gate_ctx_common.lang_pool). No task list, no
classifier, no string matching anywhere. Trained on all 200 synth demos, true prompts,
serving checkpoint's features. Embedding-noise augmentation scaled to the within-task
embedding std guards against lookup collapse. Report: held c-R2 overall + per phase
(early/transit/tail) per the 2026-08-08 standing-instrument convention.
Exports: the cache at $CACHE (default langprior_feats.npz) and a prior at $PRIOR_OUT (default
langprior_rrr.pt). BOTH paths must be set when extracting for a new checkpoint: this script used to
write the prior to a HARDCODED langprior_rrr.pt, so every re-extraction silently replaced the prior
other experiments were serving — which is how a matched-feature arm got re-measured with a prior fitted
on a different checkpoint's features and scored 0/10 (2026-08-12). Same bug class as the hardcoded
FEAT_CKPT it replaced: an artifact quietly paired with the wrong feature source.
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
# The checkpoint whose VLM produces the features. This MUST be the checkpoint the prior will be
# served on: lang_pool reads post-fusion language tokens out of the served model, so caching with
# one flow and serving another feeds the prior a representation from different weights (2026-08-11).
CKPT = os.environ.get("FEAT_CKPT",
                      "/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999")
H, AD, K = 50, 32, 5
STRIDE = 6
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = LEFT.replace("left", "right")
CFL = "go through the center gate from the left and hover over the stuffed animal"
CFR = "go through the center gate from the right and hover over the stuffed animal"
TASKS = [CFL, CFR, LEFT, RIGHT]

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
    # PIPE_CONFIG=pi0_gate_full when FEAT_CKPT is raw pi0_base: the gate configs are LoRA and
    # create_trained_policy refuses the missing lora_a/lora_b; the full twin matches pi0_base.
    cfg = C.get_config(os.environ.get("PIPE_CONFIG", "pi0_gate"))
    nsp = pads(ns, cfg.model.action_dim)
    policy = PC.create_trained_policy(cfg, CKPT, norm_stats=nsp)
    amean = np.asarray(nsp["actions"].mean); astd = np.asarray(nsp["actions"].std)
    U = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32)

    def c_of(chunk7):
        L = len(chunk7)
        ch = np.zeros((H, AD), np.float32); m = min(L, H)
        ch[:m, :7] = (chunk7[:m] - amean[:7]) / (astd[:7] + 1e-6)
        if m < H: ch[m:, :7] = ch[m - 1, :7]
        return ch.reshape(-1) @ U

    cache = os.environ.get("CACHE", f"{RD}/langprior_feats.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        E, S, Yc, ep_ix, frac = z["E"], z["S"], z["Yc"], z["ep"], z["frac"]
        print("loaded cache", E.shape, flush=True)
    else:
        E, S, Yc, ep_ix, frac = [], [], [], [], []
        for i in range(200):
            d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
            lang = TASKS[i // 50]
            st = d["state"].astype(np.float32); ac = d["action"].astype(np.float32)
            T = len(st)
            raws, metas = [], []
            for t in range(0, T - 5, STRIDE):
                raws.append({"observation/image": r224(d["image"][t]),
                             "observation/wrist_image": r224(d["wrist"][t]),
                             "observation/state": st[t], "prompt": lang})
                metas.append(t)
            for j in range(0, len(raws), 8):
                E.append(gc.lang_pool(policy, raws[j:j + 8]))
            for t, r in zip(metas, raws):
                ms = np.asarray(policy._input_transform(dict(r))["state"]).reshape(-1)
                S.append(ms); Yc.append(c_of(ac[t:t + H])); ep_ix.append(i); frac.append(t / (T - 1))
            if i % 20 == 0:
                print(f"  ep {i} rows {len(S)}", flush=True)
        E = np.concatenate(E, 0).astype(np.float32)
        S = np.array(S, np.float32); Yc = np.array(Yc, np.float32)
        ep_ix = np.array(ep_ix, np.int64); frac = np.array(frac, np.float32)
        np.savez_compressed(cache, E=E, S=S, Yc=Yc, ep=ep_ix, frac=frac)
        print("extracted", E.shape, flush=True)

    rng = np.random.default_rng(0)
    tr_eps = set(rng.permutation(200)[:160].tolist())
    tr = np.array([e in tr_eps for e in ep_ix])
    # PCA-64 on train embeddings
    Em = E[tr].mean(0)
    _, _, Vt = np.linalg.svd(E[tr] - Em, full_matrices=False)
    P = Vt[:64].T.astype(np.float32)
    E64 = (E - Em) @ P
    # embedding-noise scale = within-task std (guards lookup collapse)
    task_of_ep = (np.asarray(ep_ix) // 50)
    within = np.concatenate([E64[tr & (task_of_ep == k)] - E64[tr & (task_of_ep == k)].mean(0)
                             for k in range(4)])
    emb_sig = within.std(0)
    print("within-task emb std (mean over dims):", float(emb_sig.mean()), flush=True)

    X = np.concatenate([S, E64], 1).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X[tr] - mu) / sd); Yt = torch.tensor(Yc[tr])
    nstate = S.shape[1]
    esig_t = torch.tensor((emb_sig / sd[nstate:]).astype(np.float32))
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, K))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(400):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            j = p[i:i + 1024]; xb = Xt[j].clone()
            xb[:, :nstate] += 0.1 * torch.randn_like(xb[:, :nstate])          # state aug (as prior4)
            xb[:, nstate:] += 2.0 * esig_t * torch.randn_like(xb[:, nstate:])  # embedding aug
            opt.zero_grad(); ((net(xb) - Yt[j]) ** 2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor((X - mu) / sd)).numpy()
    def r2(m):
        return 1 - ((Yc[m] - pred[m]) ** 2).sum() / (((Yc[m] - Yc[m].mean(0)) ** 2).sum() + 1e-9)
    print(f"held c-R2 {r2(~tr):+.4f} (train {r2(tr):+.4f})", flush=True)
    for name, lo, hi in [("early", 0.0, 0.5), ("transit", 0.5, 0.75), ("tail", 0.75, 1.01)]:
        m = (~tr) & (frac >= lo) & (frac < hi)
        print(f"  phase {name:8s} held c-R2 {r2(m):+.4f} n={int(m.sum())}", flush=True)
    torch.save({"kind": "lang_prior", "in_dim": X.shape[1], "hidden": [256, 256], "K": K,
                "nstate": nstate, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                "Em": Em.astype(np.float32), "P": P, "state_dict": net.state_dict()},
               os.environ.get("PRIOR_OUT", f"{RD}/langprior_rrr.pt"))
    print("LANGPRIOR_DONE", flush=True)

if __name__ == "__main__":
    main()
