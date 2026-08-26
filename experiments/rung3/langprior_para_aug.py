"""Paraphrase-augmented language prior (2026-08-09): the canonical-prompt prior flies
paraphrases well on the LEFT scene (3/5 full, matching canonical) but fails on the RIGHT
(0/5 full, 1/5 transit). Remedy: train the prior on training paraphrases as well as the
canonical prompt, so command space is smooth across ways of saying the same thing.

Reuses langprior_feats.npz for the canonical rows and extracts extra rows (train episodes,
every PARA_STRIDE-th sampled frame, one random TRAIN paraphrase each). Refits the PCA on
the combined training embeddings, retrains, exports langprior_para.pt.
Held-out eval paraphrases (gate_b_paraphrase) remain untouched.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
from train_paraphrases import TRAIN_PARAPHRASES

RD = os.path.dirname(os.path.abspath(__file__))
HFB = "/home/ubuntu/hf_bundle/gate-drone-pi0"
CKPT = "/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999"
H, AD, K = 50, 32, 5
STRIDE = 6            # must match langprior_pipeline
PARA_EVERY = 3        # take every 3rd sampled frame for a paraphrase row
TASKS = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]


def r224(img):
    return np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC)).astype(np.uint8)


def main():
    import openpi.training.config as C
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as NZ
    from openpi.transforms import NormStats

    z = np.load(f"{RD}/langprior_feats.npz")
    E0, S0, Yc0, ep0, frac0 = z["E"], z["S"], z["Yc"], z["ep"], z["frac"]
    print("canonical rows", E0.shape, flush=True)

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

    rng = np.random.default_rng(0)
    tr_eps = set(rng.permutation(200)[:160].tolist())

    cache = f"{RD}/langprior_para_feats.npz"
    if os.path.exists(cache):
        zz = np.load(cache)
        Ep, Sp, Ycp, epp, fracp = zz["E"], zz["S"], zz["Yc"], zz["ep"], zz["frac"]
        print("loaded paraphrase cache", Ep.shape, flush=True)
    else:
        prng = np.random.default_rng(1)
        Ep, Sp, Ycp, epp, fracp = [], [], [], [], []
        for i in sorted(tr_eps):
            task = TASKS[i // 50]
            plist = TRAIN_PARAPHRASES[task]
            d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
            st = d["state"].astype(np.float32); T = len(st)
            ts = list(range(0, T - 5, STRIDE))[::PARA_EVERY]
            # canonical rows for these frames are already in E0/Yc0 — reuse their targets
            mask = (ep0 == i)
            fr_all = frac0[mask]; yc_all = Yc0[mask]; s_all = S0[mask]
            all_ts = list(range(0, T - 5, STRIDE))
            idx_of = {t: k for k, t in enumerate(all_ts)}
            raws, keep = [], []
            for t in ts:
                if t not in idx_of or idx_of[t] >= len(yc_all):
                    continue
                raws.append({"observation/image": r224(d["image"][t]),
                             "observation/wrist_image": r224(d["wrist"][t]),
                             "observation/state": st[t],
                             "prompt": plist[prng.integers(len(plist))]})
                keep.append(idx_of[t])
            for j in range(0, len(raws), 8):
                Ep.append(gc.lang_pool(policy, raws[j:j + 8]))
            Sp.append(s_all[keep]); Ycp.append(yc_all[keep])
            epp += [i] * len(keep); fracp += list(fr_all[keep])
            if i % 20 == 0:
                print(f"  para ep {i} rows {sum(len(a) for a in Ycp)}", flush=True)
        Ep = np.concatenate(Ep, 0).astype(np.float32)
        Sp = np.concatenate(Sp, 0).astype(np.float32); Ycp = np.concatenate(Ycp, 0).astype(np.float32)
        epp = np.array(epp, np.int64); fracp = np.array(fracp, np.float32)
        np.savez_compressed(cache, E=Ep, S=Sp, Yc=Ycp, ep=epp, frac=fracp)
        print("paraphrase rows", Ep.shape, flush=True)

    E = np.concatenate([E0, Ep]); S = np.concatenate([S0, Sp])
    Yc = np.concatenate([Yc0, Ycp]); ep = np.concatenate([ep0, epp]); frac = np.concatenate([frac0, fracp])
    tr = np.array([e in tr_eps for e in ep])
    Em = E[tr].mean(0)
    _, _, Vt = np.linalg.svd(E[tr] - Em, full_matrices=False)
    P = Vt[:64].T.astype(np.float32)
    E64 = (E - Em) @ P
    task_of = ep // 50
    within = np.concatenate([E64[tr & (task_of == k)] - E64[tr & (task_of == k)].mean(0) for k in range(4)])
    emb_sig = within.std(0)
    X = np.concatenate([S, E64], 1).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    nstate = S.shape[1]
    Xt = torch.tensor((X[tr] - mu) / sd); Yt = torch.tensor(Yc[tr])
    esig_t = torch.tensor((emb_sig / sd[nstate:]).astype(np.float32))
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, K))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for e in range(400):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            j = p[i:i + 1024]; xb = Xt[j].clone()
            xb[:, :nstate] += 0.1 * torch.randn_like(xb[:, :nstate])
            xb[:, nstate:] += 2.0 * esig_t * torch.randn_like(xb[:, nstate:])
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
               f"{RD}/langprior_para.pt")
    print("LANGPRIOR_PARA_DONE", flush=True)


if __name__ == "__main__":
    main()
