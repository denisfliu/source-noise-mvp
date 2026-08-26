"""Flow-embedding c predictor (#3): predict c=U^T a from pi0's OWN observation embedding. The pin's
prior is only as good as its features; the strongest available feature is the representation the flow
itself conditions on. pi0.embed_prefix maps (images, language) -> prefix tokens [b,s,2048] (state
enters later via embed_suffix, so the prefix is pure perception+instruction). We masked-mean-pool the
prefix to one 2048-vector per frame, cache it, and train a small head -> c on held-out real (episode
split). Compares head(embed), head(embed,state,lang) against the state+lang MLP (0.66) and oracle
(0.97). Embedding taken from a gate-trained checkpoint (its vision tower saw this robot's data)."""
import argparse
import json
import os

import numpy as np
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32


def seg_to_c(seg, amean, astd, U):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1) @ U


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/code/openpi/checkpoints/pi0_gate/gate_both_pin/4999"))
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--norm", default=os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
    ap.add_argument("--raw", default=os.path.join(RD, "data_gate_real"))
    ap.add_argument("--U", default=os.path.join(RD, "pin_U_gate_k5.npy"))
    ap.add_argument("--cache", default=os.path.join(RD, "flowemb_cache.npz"))
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    import jax, jax.numpy as jnp
    import openpi.training.config as _config
    import openpi.policies.policy_config as _policy_config
    import openpi.shared.normalize as _normalize
    from openpi.models import model as _model

    U = np.load(args.U).astype(np.float32)
    ns = _normalize.load(args.norm)
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
    smean, sstd = np.asarray(ns["state"].mean), np.asarray(ns["state"].std)

    meta = json.load(open(os.path.join(args.raw, "meta.json")))
    keys = sorted(meta); tasks = sorted({meta[k]["task"] for k in keys}); tid = {t: i for i, t in enumerate(tasks)}
    eps = []
    for k in keys:
        d = np.load(os.path.join(args.raw, k + ".npz"))
        eps.append({"image": d["image"], "wrist": d["wrist"], "state": d["state"].astype(np.float32),
                    "action": d["action"].astype(np.float32), "task": tid[meta[k]["task"]], "lang": meta[k]["lang"]})
    rng = np.random.default_rng(0); idx = rng.permutation(len(eps)); ntr = int(0.7 * len(eps))
    split = {i: ("tr" if p < ntr else "te") for p, i in enumerate(idx)}

    def r224(im):
        return np.asarray(Image.fromarray(im).resize((224, 224), Image.BILINEAR), np.uint8)

    # build per-frame records (raw obs + c + state + lang-onehot + split)
    recs = []
    for ei, ep in enumerate(eps):
        T = len(ep["action"])
        for t in range(0, T, 3):
            oh = np.zeros(len(tasks), np.float32); oh[ep["task"]] = 1
            recs.append({"obs": {"observation/image": r224(ep["image"][t]),
                                 "observation/wrist_image": r224(ep["wrist"][t]),
                                 "observation/state": ep["state"][t], "prompt": ep["lang"]},
                         "c": seg_to_c(ep["action"][t:], amean, astd, U),
                         "st": (ep["state"][t] - smean) / (sstd + 1e-6), "oh": oh, "sp": split[ei]})
    print(f"frames={len(recs)} tasks={len(tasks)} K={U.shape[1]}", flush=True)

    if os.path.exists(args.cache):
        E = np.load(args.cache)["emb"]
        print(f"loaded cached embeddings {E.shape}", flush=True)
    else:
        policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.ckpt, norm_stats=ns)

        def embed_batch(raws):
            tds = [policy._input_transform(dict(r)) for r in raws]
            batched = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
            obs = _model.preprocess_observation(None, _model.Observation.from_dict(batched), train=False)
            tokens, mask, _ = policy._model.embed_prefix(obs)
            m = mask[..., None].astype(jnp.float32); tk = tokens.astype(jnp.float32)
            return np.asarray((tk * m).sum(1) / jnp.clip(m.sum(1), 1e-6))

        embs = []
        for i in range(0, len(recs), args.bs):
            embs.append(embed_batch([r["obs"] for r in recs[i:i + args.bs]]))
            if i % (args.bs * 20) == 0:
                print(f"  embed {i}/{len(recs)}", flush=True)
        E = np.concatenate(embs, 0).astype(np.float32)
        np.savez_compressed(args.cache, emb=E)
        print(f"embedded {E.shape} -> cached", flush=True)

    st = np.stack([r["st"] for r in recs]); oh = np.stack([r["oh"] for r in recs])
    C = np.stack([r["c"] for r in recs]); sp = np.array([r["sp"] for r in recs])
    tr, te = sp == "tr", sp == "te"
    fm, fs = E[tr].mean(0), E[tr].std(0) + 1e-6
    En = (E - fm) / fs

    import torch, torch.nn as nn
    dev = "cpu"

    def head(din, name, Xtr, Xte, steps=4000):
        net = nn.Sequential(nn.Linear(din, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 256),
                            nn.SiLU(), nn.Linear(256, C.shape[1])).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(C[tr], dtype=torch.float32)
        for _ in range(steps):
            b = torch.randint(0, len(xt), (256,))
            loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(Xte, dtype=torch.float32)).numpy()
        print(f"{name:32s} held R^2 = {r2(pred, C[te]):.3f}", flush=True)

    head(En.shape[1], "head(flow-embed)", En[tr], En[te])
    Xtr = np.concatenate([En[tr], st[tr], oh[tr]], 1); Xte = np.concatenate([En[te], st[te], oh[te]], 1)
    head(Xtr.shape[1], "head(flow-embed,state,lang)", Xtr, Xte)
    print("FLOWEMB_DONE", flush=True)


if __name__ == "__main__":
    main()
