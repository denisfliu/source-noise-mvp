"""Stronger c predictor for the gate pin. The current prior is a LINEAR map from (state, which-gate
onehot) that ignores the image -- but for a gate task the right action is set by where the gate is in
the drone's view. This trains and compares, on held-out real (episode split), predictors of c=U^T a:
  (1) linear(state, lang)         -- the current baseline
  (2) MLP(state, lang)            -- nonlinearity, no vision
  (3) CNN(image)+state+lang -> c  -- adds vision
Reports held-out R^2 for each and saves the best. c/state normalization matches the eval exactly
(gate norm stats + gate U). Frame obs at t -> c of the action chunk at t."""
import glob
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
ns = NZ.load(os.path.join(RD, "..", "..", "..", "..", "code", "openpi", "assets", "pi0_gate", "local", "gate_nav")) \
    if False else NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
smean, sstd = np.asarray(ns["state"].mean), np.asarray(ns["state"].std)
U = np.load(os.path.join(RD, "pin_U_gate_k5.npy")).astype(np.float32)


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
            im = np.asarray(Image.fromarray(imgs[t]).resize((64, 64), Image.BILINEAR), np.float32).transpose(2, 0, 1) / 255.0
            IMG.append(im); C.append(seg_to_c(acts[t:]))
    return (np.asarray(S, np.float32), np.asarray(L, np.float32), np.asarray(IMG, np.float32), np.asarray(C, np.float32))


class MLP(nn.Module):
    def __init__(s, din, K):
        super().__init__(); s.net = nn.Sequential(nn.Linear(din, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, K))
    def forward(s, x): return s.net(x)


class CNN(nn.Module):
    def __init__(s, daux, K):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
                               nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten())
        s.head = nn.Sequential(nn.Linear(64 * 16 + daux, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, K))
    def forward(s, img, aux): return s.head(torch.cat([s.conv(img), aux], 1))


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def main():
    eps, ntask = load(os.path.join(RD, "data_gate_real"))
    rng = np.random.default_rng(0); idx = rng.permutation(len(eps)); ntr = int(0.7 * len(eps))
    tr = [eps[i] for i in idx[:ntr]]; te = [eps[i] for i in idx[ntr:]]
    Str, Ltr, Itr, Ctr = build(tr, ntask); Ste, Lte, Ite, Cte = build(te, ntask)
    print(f"tasks={ntask} train_frames={len(Str)} test_frames={len(Ste)} K={Ctr.shape[1]}")

    # (1) linear(state,lang)
    Ftr = np.concatenate([Str, Ltr, np.ones((len(Str), 1), np.float32)], 1)
    Fte = np.concatenate([Ste, Lte, np.ones((len(Ste), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Ftr, Ctr, rcond=None)
    print(f"(1) linear(state,lang)      held R^2 = {r2(Fte @ W, Cte):.3f}")

    def train_torch(model, xtr, xte, name, img=False):
        opt = torch.optim.Adam(model.parameters(), 1e-3); ytr = torch.tensor(Ctr, device=DEV); n = len(Ctr)
        model.to(DEV)
        for step in range(3000):
            b = torch.randint(0, n, (256,), device=DEV)
            if img:
                p = model(xtr[0][b], xtr[1][b])
            else:
                p = model(xtr[b])
            loss = ((p - ytr[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = (model(xte[0], xte[1]) if img else model(xte)).cpu().numpy()
        print(f"{name} held R^2 = {r2(pred, Cte):.3f}  (train loss {loss.item():.4f})")
        return model

    # (2) MLP(state,lang)
    xtr = torch.tensor(np.concatenate([Str, Ltr], 1), device=DEV); xte = torch.tensor(np.concatenate([Ste, Lte], 1), device=DEV)
    train_torch(MLP(Str.shape[1] + ntask, Ctr.shape[1]), xtr, xte, "(2) MLP(state,lang)       ")

    # (3) CNN(image)+state+lang
    aux_tr = torch.tensor(np.concatenate([Str, Ltr], 1), device=DEV); aux_te = torch.tensor(np.concatenate([Ste, Lte], 1), device=DEV)
    itr = torch.tensor(Itr, device=DEV); ite = torch.tensor(Ite, device=DEV)
    m = train_torch(CNN(Str.shape[1] + ntask, Ctr.shape[1]), (itr, aux_tr), (ite, aux_te), "(3) CNN(image)+state+lang ", img=True)
    torch.save(m.state_dict(), os.path.join(RD, "c_predictor_cnn.pt"))
    print("SAVED c_predictor_cnn.pt")


if __name__ == "__main__":
    main()
