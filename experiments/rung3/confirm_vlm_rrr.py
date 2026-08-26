"""Confirmation run for the RRR-from-VLA-features claim (original: vlm_rrr_libero.py, 2026-08-01,
held-out tasks 18,19,28,29: VLM-context c-R^2 +0.40 vs state+onehot -1.90).

The original compared recipes that differ in BOTH basis and prior predictor, on ONE held-out split.
This runs the clean factorial so the basis effect is isolated from the predictor effect:

    basis  in {RRR(VLM-context feats -> chunks), PCA(chunks)}
    prior  in {MLP(VLM-context), MLP(state + task-onehot)}

over the in-distribution episode split plus THREE disjoint task-heldout splits. Uses the exact rec
construction of vlm_rrr_libero.py (sorted meta keys, stride 8, seg_to_Y zero-pad) against the cached
post-fusion features vlm_feat_context.npz — row alignment re-verified (3483 both ways) before this
was written. CPU-only; torch seeded per fit.

Metrics per cell: held c-R^2 (per suite + all), capture = share of TOTAL chunk variance the basis
spans on test (comparable across bases, unlike the original's self-referenced coverage).
"""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD, K, STRIDE = 50, 32, 5, 8
SPLITS = [("in-dist", None),
          ("held 18,19,28,29 (original)", {18, 19, 28, 29}),
          ("held 12,13,22,23", {12, 13, 22, 23}),
          ("held 15,16,25,26", {15, 16, 25, 26})]


def seg_to_Y(seg, amean, astd):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def rrr_U(X, Y, k):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Yhat = Xb @ W
    Yc = Yhat - Yhat.mean(0)
    w, V = np.linalg.eigh((Yc.T @ Yc) / len(Yc))
    return V[:, ::-1][:, :k].astype(np.float32)


def pca_U(Y, k):
    Yc = Y - Y.mean(0)
    w, V = np.linalg.eigh((Yc.T @ Yc) / len(Yc))
    return V[:, ::-1][:, :k].astype(np.float32)


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def fit_mlp(Xtr, Ytr, Xte, seed=0, steps=4000):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.SiLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(((Xtr - m) / s).astype(np.float32)), torch.tensor(Ytr.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,), generator=g)
        loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(((Xte - m) / s).astype(np.float32))).numpy()


def main():
    import openpi.shared.normalize as _normalize
    ns = _normalize.load(os.path.join(RD, "norm_shared_libero"))
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)

    meta = json.load(open(os.path.join(RD, "data_libero_multi", "meta.json")))
    keys = sorted(meta); tasks = sorted({meta[k]["task"] for k in keys})
    tid = {t: i for i, t in enumerate(tasks)}
    Ys, rtask, rei = [], [], []
    for ei, k in enumerate(keys):
        d = np.load(os.path.join(RD, "data_libero_multi", k + ".npz"))
        act = d["action"].astype(np.float32)
        for t in range(0, len(act), STRIDE):
            Ys.append(seg_to_Y(act[t:], amean, astd)); rtask.append(meta[k]["task"]); rei.append(ei)
    Y = np.stack(Ys); rtask = np.array(rtask); rei = np.array(rei)

    z = np.load(os.path.join(RD, "vlm_feat_context.npz"))
    X, ST = z["X"].astype(np.float32), z["ST"].astype(np.float32)
    assert len(X) == len(Y), (len(X), len(Y))
    oh = np.zeros((len(Y), len(tasks)), np.float32)
    oh[np.arange(len(Y)), [tid[t] for t in rtask]] = 1
    Xso = np.concatenate([ST.reshape(len(ST), -1), oh], 1)
    suite = np.where(rtask < 20, "goal", "object")

    rng = np.random.default_rng(0)
    idx = rng.permutation(rei.max() + 1); ntr = int(0.7 * (rei.max() + 1))
    trep = set(idx[:ntr].tolist())
    Yvar_tot = None

    for sname, held in SPLITS:
        if held is None:
            tr = np.isin(rei, list(trep)); te = ~tr
        else:
            te = np.isin(rtask, list(held)); tr = ~te
        bases = {"RRR(vla-feat)": rrr_U(X[tr], Y[tr], K), "PCA(chunks)": pca_U(Y[tr], K)}
        priors = {"MLP(vlm-ctx)": X, "MLP(state+onehot)": Xso}
        print(f"\n==== split: {sname}  (train {tr.sum()} / test {te.sum()}) ====", flush=True)
        for bname, U in bases.items():
            C = Y @ U
            cap = float((C[te] - C[te].mean(0)).var(0).sum()
                        / ((Y[te] - Y[te].mean(0)).var(0).sum() + 1e-9))
            for pname, F in priors.items():
                pred = fit_mlp(F[tr], C[tr], F[te])
                row = f"  {bname:14s} x {pname:18s} capture={cap:.3f}  all={r2(pred, C[te]):+.3f}"
                for su in ("goal", "object"):
                    m = suite[te] == su
                    row += f"  {su}={r2(pred[m], C[te][m]):+.3f}"
                print(row, flush=True)
    print("\nCONFIRM_DONE", flush=True)


if __name__ == "__main__":
    main()
