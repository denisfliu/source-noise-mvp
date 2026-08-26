"""Temporal-context c predictor. Every prior so far is feedforward on a single frame; the gap to the
oracle (which reads c off the true future action) looks like near-future content a single frame can't
determine. A short history supplies it: recent states give velocity/momentum and recently-executed
actions give the current motion direction -- both legitimately available at deployment (they are past,
not future). Features at anchor t: past W states incl. current (front zero-padded), past W-1 actions
strictly before t (momentum), first-difference velocity, and language one-hot. Held-out real, episode
split (seed 0), matched to the other c-predictor studies. Compares against the MLP(state,lang) 0.66
and oracle 0.97, and ablates state-history vs +action-history to see if momentum is the missing signal."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD, W = 50, 32, 8
import openpi.shared.normalize as NZ
ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
smean, sstd = np.asarray(ns["state"].mean), np.asarray(ns["state"].std)
U = np.load(os.path.join(RD, "pin_U_gate_k5.npy")).astype(np.float32)
SD = len(smean)


def seg_to_c(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1) @ U


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def main():
    raw = os.path.join(RD, "data_gate_real")
    meta = json.load(open(os.path.join(raw, "meta.json")))
    keys = sorted(meta); tasks = sorted({meta[k]["task"] for k in keys}); tid = {t: i for i, t in enumerate(tasks)}
    eps = []
    for k in keys:
        d = np.load(os.path.join(raw, k + ".npz"))
        eps.append((d["state"].astype(np.float32), d["action"].astype(np.float32), tid[meta[k]["task"]]))
    rng = np.random.default_rng(0); idx = rng.permutation(len(eps)); ntr = int(0.7 * len(eps))
    sp = {i: ("tr" if p < ntr else "te") for p, i in enumerate(idx)}

    # feature banks: cur=[state,lang]; hist_s adds past-W states + velocity; hist_sa adds past actions
    cur, hs, hsa, C, S = [], [], [], [], []
    ntask = len(tasks)
    for ei, (states, acts, ti) in enumerate(eps):
        T = len(acts)
        sn = (states - smean) / (sstd + 1e-6)          # [T, SD]
        an = (acts - amean[:acts.shape[1]]) / (astd[:acts.shape[1]] + 1e-6)  # [T, ad]
        oh = np.zeros(ntask, np.float32); oh[ti] = 1
        for t in range(0, T, 3):
            ps = np.zeros((W, SD), np.float32)          # past states incl current, front zero-pad
            pa = np.zeros((W, an.shape[1]), np.float32)  # past actions strictly before t
            for j in range(W):
                si = t - (W - 1) + j
                if si >= 0:
                    ps[j] = sn[si]
                ai = t - W + j                           # ...t-1
                if 0 <= ai < t:
                    pa[j] = an[ai]
            vel = sn[t] - (sn[t - 1] if t > 0 else sn[t])
            cur.append(np.concatenate([sn[t], oh]))
            hs.append(np.concatenate([ps.reshape(-1), vel, oh]))
            hsa.append(np.concatenate([ps.reshape(-1), pa.reshape(-1), vel, oh]))
            C.append(seg_to_c(acts[t:])); S.append(sp[ei])
    cur, hs, hsa = np.asarray(cur, np.float32), np.asarray(hs, np.float32), np.asarray(hsa, np.float32)
    C = np.asarray(C, np.float32); S = np.asarray(S)
    tr, te = S == "tr", S == "te"
    print(f"frames tr={tr.sum()} te={te.sum()} K={C.shape[1]} W={W} dims cur={cur.shape[1]} hs={hs.shape[1]} hsa={hsa.shape[1]}", flush=True)

    import torch, torch.nn as nn

    def head(X, name, steps=4000):
        Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - Xm) / Xs
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 256),
                            nn.SiLU(), nn.Linear(256, C.shape[1]))
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        xt, yt = torch.tensor(Xn[tr]), torch.tensor(C[tr])
        for _ in range(steps):
            b = torch.randint(0, len(xt), (256,))
            loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(Xn[te])).numpy()
        print(f"{name:40s} held R^2 = {r2(pred, C[te]):.3f}", flush=True)

    head(cur, "MLP(state,lang) [baseline]")
    head(hs, "MLP(state-history+vel, lang)")
    head(hsa, "MLP(state-history+action-history+vel, lang)")
    print("TEMPORAL_DONE", flush=True)


if __name__ == "__main__":
    main()
