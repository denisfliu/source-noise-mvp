"""Falsification test for the mode-averaging thesis (Denis discussion, 2026-08-13).

Claim: command failures concentrate where p(c | o) is MULTIMODAL, because a regression head
outputs the conditional mean and the mean of two modes is not a valid future. Test: fit the usual
MLP head on cached features -> c, then examine held-out residuals by trajectory segment. If the
thesis is right, residual distributions at BRANCH states (episode tails: go-vs-stop) should be
bimodal, and mid-transit residuals should be unimodal. Numpy-only 1-D two-component GMM via EM;
evidence metric = BIC(2 components) - BIC(1 component) (negative = bimodal preferred) plus the
fitted mode separation in sigma units.
"""
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)


def fit_mlp(Xtr, Ytr, Xte, seed=0, steps=3000):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt = torch.tensor(((Xtr - m) / s).astype(np.float32)); yt = torch.tensor(Ytr.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,), generator=g)
        loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(((Xte - m) / s).astype(np.float32))).numpy()


def gmm_bic(x, k, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    n = len(x)
    if k == 1:
        mu, var = x.mean(), x.var() + 1e-9
        ll = -0.5 * n * (np.log(2 * np.pi * var) + 1)
        return -2 * ll + 2 * np.log(n), (mu,), (np.sqrt(var),), (1.0,)
    mu = np.percentile(x, [25, 75]).astype(float)
    var = np.full(2, x.var() + 1e-9); w = np.array([0.5, 0.5])
    for _ in range(iters):
        p = np.stack([w[j] / np.sqrt(2 * np.pi * var[j]) *
                      np.exp(-0.5 * (x - mu[j]) ** 2 / var[j]) for j in range(2)])
        r = p / (p.sum(0) + 1e-300)
        nk = r.sum(1) + 1e-9
        w = nk / n
        mu = (r * x).sum(1) / nk
        var = (r * (x - mu[:, None]) ** 2).sum(1) / nk + 1e-9
    ll = np.log(np.stack([w[j] / np.sqrt(2 * np.pi * var[j]) *
                          np.exp(-0.5 * (x - mu[j]) ** 2 / var[j]) for j in range(2)]).sum(0)
                + 1e-300).sum()
    return -2 * ll + 5 * np.log(n), tuple(mu), tuple(np.sqrt(var)), tuple(w)


def main():
    z = np.load(f"{RD}/langprior_feats_base.npz")
    E, Yc, ep, frac = (z["E"].astype(np.float32), z["Yc"].astype(np.float32), z["ep"], z["frac"])
    rng = np.random.default_rng(0)
    tr_eps = set(rng.permutation(200)[:160].tolist())
    tr = np.array([e in tr_eps for e in ep]); te = ~tr
    pred = fit_mlp(E[tr], Yc[tr], E[te])
    res = Yc[te] - pred
    fr = frac[te]
    segs = {"start": fr < 0.08, "mid-transit": (fr > 0.3) & (fr < 0.6), "tail": fr > 0.85}
    K = Yc.shape[1]
    print(f"held rows {te.sum()}; residual bimodality per segment/dim "
          f"(dBIC<0 => 2 modes preferred; sep = |mu1-mu2| in pooled-sigma units)")
    for sname, sm in segs.items():
        row = []
        for k in range(K):
            x = res[sm, k]
            b1, *_ = gmm_bic(x, 1)
            b2, mus, sds, ws = gmm_bic(x, 2)
            sep = abs(mus[0] - mus[1]) / (x.std() + 1e-9)
            flag = "BIMODAL" if (b2 - b1 < -10 and sep > 1.2 and min(ws) > 0.15) else "-"
            row.append(f"c{k}: dBIC={b2 - b1:+7.1f} sep={sep:.2f} w={min(ws):.2f} {flag}")
        print(f"  {sname:12s} n={sm.sum():4d}")
        for r in row:
            print(f"    {r}")
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
