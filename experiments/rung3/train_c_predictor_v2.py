"""Vision-augmented c predictors for the gate pin, held-out real (episode split). Adds a FROZEN
pretrained encoder to the state+language MLP baseline: where the gate is in the drone's view is
visual, so a good image feature should lift c-predictability past the 0.665 the state+language MLP
reaches, without the from-scratch CNN's overfitting (it hit 0.48). Encoder is ImageNet resnet18 with
the classifier head removed, run in eval/no-grad; only a small head over [feat, state, lang] trains.
Reports held-out R^2 for: MLP(state,lang) baseline, head(resnet), head(resnet,state,lang)."""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import openpi.shared.normalize as NZ
ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
smean, sstd = np.asarray(ns["state"].mean), np.asarray(ns["state"].std)
U = np.load(os.path.join(RD, "pin_U_gate_k5.npy")).astype(np.float32)
IMEAN = np.array([0.485, 0.456, 0.406], np.float32)
ISTD = np.array([0.229, 0.224, 0.225], np.float32)


def seg_to_c(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1) @ U


def load(raw):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    keys = sorted(meta); tasks = sorted({meta[k]["task"] for k in keys}); tid = {t: i for i, t in enumerate(tasks)}
    eps = []
    for k in keys:
        d = np.load(os.path.join(raw, k + ".npz"))
        eps.append((d["image"], d["state"].astype(np.float32), d["action"].astype(np.float32), tid[meta[k]["task"]], len(tasks)))
    return eps, len(tasks)


def build(eps, ntask, stride=3):
    S, L, IMG, C = [], [], [], []
    for imgs, states, acts, ti, nt in eps:
        T = len(acts)
        for t in range(0, T, stride):
            S.append((states[t] - smean) / (sstd + 1e-6))
            oh = np.zeros(ntask, np.float32); oh[ti] = 1; L.append(oh)
            im = np.asarray(Image.fromarray(imgs[t]).resize((224, 224), Image.BILINEAR), np.float32) / 255.0
            im = (im - IMEAN) / ISTD
            IMG.append(im.transpose(2, 0, 1)); C.append(seg_to_c(acts[t:]))
    return (np.asarray(S, np.float32), np.asarray(L, np.float32), np.asarray(IMG, np.float32), np.asarray(C, np.float32))


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


@torch.no_grad()
def resnet_feats(enc, imgs, bs=64):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.tensor(imgs[i:i + bs], device=DEV)
        out.append(enc(x).squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def train_head(din, K, xtr, ytr, xte, yte, name, steps=3000):
    net = nn.Sequential(nn.Linear(din, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, K)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(xtr, device=DEV), torch.tensor(ytr, device=DEV)
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,), device=DEV)
        loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(xte, device=DEV)).cpu().numpy()
    print(f"{name:38s} held R^2 = {r2(pred, yte):.3f}", flush=True)
    return r2(pred, yte)


def main():
    import torchvision
    eps, ntask = load(os.path.join(RD, "data_gate_real"))
    rng = np.random.default_rng(0); idx = rng.permutation(len(eps)); ntr = int(0.7 * len(eps))
    tr = [eps[i] for i in idx[:ntr]]; te = [eps[i] for i in idx[ntr:]]
    Str, Ltr, Itr, Ctr = build(tr, ntask); Ste, Lte, Ite, Cte = build(te, ntask)
    print(f"tasks={ntask} train={len(Str)} test={len(Ste)} K={Ctr.shape[1]} dev={DEV}", flush=True)

    SLtr = np.concatenate([Str, Ltr], 1); SLte = np.concatenate([Ste, Lte], 1)
    train_head(SLtr.shape[1], Ctr.shape[1], SLtr, Ctr, SLte, Cte, "MLP(state,lang) [baseline]")

    enc = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    enc.fc = nn.Identity(); enc = enc.to(DEV).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print("extracting frozen resnet18 features...", flush=True)
    Ftr, Fte = resnet_feats(enc, Itr), resnet_feats(enc, Ite)
    # standardize features on train stats
    fm, fs = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - fm) / fs; Fte = (Fte - fm) / fs

    train_head(Ftr.shape[1], Ctr.shape[1], Ftr, Ctr, Fte, Cte, "head(resnet only)")
    Xtr = np.concatenate([Ftr, SLtr], 1); Xte = np.concatenate([Fte, SLte], 1)
    train_head(Xtr.shape[1], Ctr.shape[1], Xtr, Ctr, Xte, Cte, "head(resnet,state,lang)")
    print("CPRED_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
