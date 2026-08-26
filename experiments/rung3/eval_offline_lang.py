"""Both-cases offline eval at VLA scale. Frozen sim flow (RRR or PCA pin) driven by priors fit on a
train split of the multi-task LIBERO set (goal 10-19 language-driven, object 20-29 state-driven).
Arms: no_pin; state->c prior; (state+language)->c prior; oracle (c=U^T a). Language = task one-hot
(in-distribution). Reported PER SUITE so we can see: on goal, does adding language beat state-only
and no_pin; on object, does state-only already suffice (language shouldn't hurt). Metrics: subspace
R^2 (pin channel: c_pred vs c_gt) and full-action R^2. Episode-level train/eval split."""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
sys.path.insert(0, RD)
import pca_pin as PP  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.policies.policy_config as _policy_config  # noqa: E402
import openpi.shared.normalize as _normalize  # noqa: E402
H, AD = 50, 32


def resize224(img):
    if img.shape[:2] != (224, 224):
        img = np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC))
    return img.astype(np.uint8)


def make_obs(ep, t, prompt):
    img = resize224(ep["image"][t])
    wrist = resize224(ep["wrist"][t]) if "wrist" in ep else img.copy()
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": ep["state"][t], "prompt": prompt}


def seg_to_c(seg, amean, astd, U):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1) @ U


def fit(X, y):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    W, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return W


def apply(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--U", required=True)
    ap.add_argument("--config", default="pi0_libero_shared")
    ap.add_argument("--norm", default=os.path.join(RD, "norm_shared_libero"))
    ap.add_argument("--raw_dir", default=os.path.join(RD, "data_libero_multi"))
    ap.add_argument("--offsets", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    U = PP.load_U(args.U)
    ns = _normalize.load(args.norm)
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.ckpt, norm_stats=ns)

    meta = json.load(open(os.path.join(args.raw_dir, "meta.json")))
    keys = sorted(meta.keys())
    tasks = sorted({meta[k]["task"] for k in keys})
    tid = {t: i for i, t in enumerate(tasks)}

    def oh(task):
        v = np.zeros(len(tasks)); v[tid[task]] = 1; return v

    eps = []
    for k in keys:
        d = np.load(os.path.join(args.raw_dir, k + ".npz"))
        eps.append({"image": d["image"], "wrist": d["wrist"], "state": d["state"].astype(np.float32),
                    "action": d["action"].astype(np.float32), "task": meta[k]["task"], "lang": meta[k]["lang"]})
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(eps)); ntr = int(0.7 * len(eps))
    train, ev = [eps[i] for i in idx[:ntr]], [eps[i] for i in idx[ntr:]]

    # fit priors
    St, Ct, Oht = [], [], []
    for ep in train:
        T = len(ep["action"])
        for t in range(0, T, max(1, T // 5)):
            St.append(np.asarray(policy._input_transform(make_obs(ep, t, ep["lang"]))["state"]).reshape(-1))
            Ct.append(seg_to_c(ep["action"][t:], amean, astd, U)); Oht.append(oh(ep["task"]))
    St, Ct, Oht = np.asarray(St), np.asarray(Ct), np.asarray(Oht)
    W_s = fit(St, Ct); W_sl = fit(np.concatenate([St, Oht], 1), Ct)
    r2s = lambda W, X: 1 - ((Ct - apply(W, X)) ** 2).sum() / (((Ct - Ct.mean(0)) ** 2).sum() + 1e-9)
    print(f"prior (train-fit) state->c R2={r2s(W_s, St):.3f}  (state+lang)->c R2={r2s(W_sl, np.concatenate([St,Oht],1)):.3f}")

    # MLP (state+lang)->c prior (nonlinear); trained on the same train split
    import torch, torch.nn as nn
    dev = "cpu"
    Xtr = np.concatenate([St, Oht], 1).astype(np.float32)
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, Ct.shape[1])).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), 1e-3)
    xt = torch.tensor(Xtr); yt = torch.tensor(Ct.astype(np.float32))
    for _ in range(3000):
        b = torch.randint(0, len(xt), (256,))
        loss = ((mlp(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()

    def mlp_c(st, ov):
        with torch.no_grad():
            return mlp(torch.tensor(np.concatenate([st, ov])[None].astype(np.float32)))[0].numpy()

    arms = ["no_pin", "state", "state_lang", "mlp", "oracle"]
    suites = {"goal(lang)": range(10, 20), "object(state)": range(20, 30)}
    acc = {s: {a: {"cp": [], "cg": [], "res": np.zeros(7), "sg": np.zeros(7), "sg2": np.zeros(7), "n": 0}
               for a in arms} for s in suites}
    rng2 = np.random.default_rng(args.seed + 1)
    for ep in ev:
        T = len(ep["action"])
        if T < 4:
            continue
        sname = "goal(lang)" if ep["task"] < 20 else "object(state)"
        for t in np.linspace(0, max(0, T - 4), args.offsets).astype(int):
            obs = make_obs(ep, t, ep["lang"])
            st = np.asarray(policy._input_transform(obs)["state"]).reshape(-1)
            ov = oh(ep["task"])
            c_s = apply(W_s, st[None])[0]
            c_sl = apply(W_sl, np.concatenate([st, ov])[None])[0]
            c_mlp = mlp_c(st, ov)
            overlap = min(H, T - t)
            gt = ep["action"][t:t + overlap]
            c_gt = seg_to_c(gt, amean, astd, U)
            g = rng2.standard_normal((H, AD)).astype(np.float32); gf = g.reshape(-1)
            pin = lambda c: (gf - (gf @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
            noises = {"no_pin": g, "state": pin(c_s), "state_lang": pin(c_sl), "mlp": pin(c_mlp), "oracle": pin(c_gt)}
            for a in arms:
                pred = np.asarray(policy.infer(obs, noise=noises[a])["actions"])[:overlap, :7]
                d = acc[sname][a]
                d["cp"].append(seg_to_c(pred, amean, astd, U)); d["cg"].append(c_gt)
                d["res"] += ((pred - gt) ** 2).sum(0); d["sg"] += gt.sum(0); d["sg2"] += (gt ** 2).sum(0); d["n"] += overlap

    w = 1.0 / (astd[:7] ** 2 + 1e-9)
    out = {"ckpt": args.ckpt, "U": args.U, "suites": {}}
    print(f"\n{'suite':>14} {'arm':>11} | {'subspace_R2':>11} | {'full_R2':>8}")
    print("-" * 52)
    for s in suites:
        out["suites"][s] = {}
        for a in arms:
            d = acc[s][a]
            cp, cg = np.asarray(d["cp"]), np.asarray(d["cg"])
            sub = 1 - ((cp - cg) ** 2).sum() / (((cg - cg.mean(0)) ** 2).sum() + 1e-9)
            tot = d["sg2"] - d["sg"] ** 2 / max(d["n"], 1)
            full = 1 - (d["res"] * w).sum() / ((tot * w).sum() + 1e-9)
            out["suites"][s][a] = {"subspace_r2": float(sub), "full_r2": float(full), "n": int(len(cp))}
            print(f"{s:>14} {a:>11} | {sub:11.3f} | {full:8.3f}")
        print()
    json.dump(out, open(args.out, "w"), indent=1)
    print("LANG_EVAL_DONE " + args.out)


if __name__ == "__main__":
    main()
