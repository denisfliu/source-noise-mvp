"""Ladder step 1: does ONE VLM-grounded c cohere across embodiments? Build a single VLM-RRR subspace U
on the pooled arm-OXE corpus (shared 6-D EE-delta action, per-embodiment RMS-normalized so scale differs
but shape is shared), then test whether one VLM->c prior predicts c (a) across all embodiments in-dist and
(b) on a HELD-OUT embodiment it never trained on. Held-out-embodiment >> 0 would mean the VLM-grounded
instruction coordinate transfers to a new robot -- the shared-coordinate pillar of the meta-transport."""
import os
import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
DSS = ["bridge", "berkeley_autolab_ur5", "toto", "viola"]
K = 5


def rrr_U(X, Y, K):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Yh = Xb @ W; Yc = Yh - Yh.mean(0); C = (Yc.T @ Yc) / len(Yc)
    _, V = np.linalg.eigh(C); return V[:, ::-1][:, :K].astype(np.float32)


def r2(p, y):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def fit_mlp(Xtr, Ytr, Xte, steps=4000):
    import torch, torch.nn as nn
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.SiLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(((Xtr - m) / s).astype(np.float32)), torch.tensor(Ytr.astype(np.float32))
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,)); loss = ((net(xt[b]) - yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(((Xte - m) / s).astype(np.float32))).numpy()


def main():
    D = {}
    for ds in DSS:
        f = os.path.join(RD, f"vlm_feat_oxe_{ds}.npz")
        if not os.path.exists(f):
            print(f"MISSING {f}", flush=True); continue
        z = np.load(f, allow_pickle=True); X = z["X"].astype(np.float32)
        Y = z["chunks"].astype(np.float32).reshape(len(z["chunks"]), -1)      # (M, 96) raw 6-D chunk
        Y = Y - Y.mean(0); Y = Y / (np.sqrt((Y ** 2).mean()) + 1e-9)          # per-embodiment RMS norm
        D[ds] = (X, Y)
        print(f"{ds}: X{X.shape} Y{Y.shape}", flush=True)
    ds_ok = list(D)
    Xall = np.concatenate([D[d][0] for d in ds_ok]); Yall = np.concatenate([D[d][1] for d in ds_ok])
    dom = np.concatenate([[i] * len(D[d][0]) for i, d in enumerate(ds_ok)])
    U = rrr_U(Xall, Yall, K)                                                  # ONE shared VLM-RRR subspace
    Call = Yall @ U
    print(f"\nshared VLM-RRR U on {len(ds_ok)} embodiments; K={K}\n", flush=True)

    # (a) pooled prior fit on ALL (70/30 within each), per-embodiment held R^2
    rng = np.random.default_rng(0)
    trm = np.zeros(len(Yall), bool)
    for i in range(len(ds_ok)):
        idx = np.where(dom == i)[0]; sub = rng.permutation(len(idx)); trm[idx[sub[:int(0.7 * len(idx))]]] = True
    pooled = fit_mlp(Xall[trm], Call[trm], Xall[~trm])
    print("(a) POOLED prior (fit all embodiments) -- held R^2 per embodiment:", flush=True)
    for i, d in enumerate(ds_ok):
        m = (~trm) & (dom == i); print(f"     {d:24s} {r2(pooled[m[~trm]], Call[~trm][m[~trm]]):+.3f}", flush=True) if False else None
    dom_te = dom[~trm]
    for i, d in enumerate(ds_ok):
        m = dom_te == i; print(f"     {d:24s} R2={r2(pooled[m], Call[~trm][m]):+.3f}  n={m.sum()}", flush=True)

    # (b) HELD-OUT EMBODIMENT: fit on the other embodiments, eval on the held-out one
    print("\n(b) HELD-OUT EMBODIMENT (prior fit on the OTHER robots, eval on this one):", flush=True)
    for i, d in enumerate(ds_ok):
        tr = dom != i; te = dom == i
        pred = fit_mlp(Xall[tr], Call[tr], Xall[te])
        # (c) per-embodiment upper bound
        idx = np.where(te)[0]; sub = rng.permutation(len(idx)); cut = int(0.7 * len(idx))
        ub = fit_mlp(Xall[idx[sub[:cut]]], Call[idx[sub[:cut]]], Xall[idx[sub[cut:]]])
        print(f"     {d:24s} held-out R2={r2(pred, Call[te]):+.3f}   (own-data UB={r2(ub, Call[idx[sub[cut:]]]):+.3f})  n={te.sum()}", flush=True)
    print("SHARED_C_DONE", flush=True)


if __name__ == "__main__":
    main()
