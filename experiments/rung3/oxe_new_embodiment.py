"""The vision test: 'to learn a new embodiment we just fit a little transport, the big VLA stays frozen.'
A reference prior P_ref (VLM->shared-c) is trained on the SOURCE embodiments. For a HELD-OUT embodiment D
we (1) apply P_ref directly (no adaptation) and (2) fit a small adapter/transport A_D that maps [P_ref's
c-guess, VLM-context] -> D's c using only N samples of D, then read out the rest of D. The data-efficiency
curve R^2(N) vs the no-adapt baseline and D's own-data upper bound says how CHEAP new-embodiment adaptation
is -- the quantitative form of 'a little transport is enough'. Shared c from the pooled VLM-RRR subspace."""
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


def fit_mlp(din, dout, X, Y, Xe, hid=256, steps=4000, wd=1e-4):
    import torch, torch.nn as nn
    m, s = X.mean(0), X.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(din, hid), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, dout))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd)
    xt, yt = torch.tensor(((X - m) / s).astype(np.float32)), torch.tensor(Y.astype(np.float32))
    for _ in range(steps):
        b = torch.randint(0, len(xt), (min(256, len(xt)),)); loss = ((net(xt[b]) - yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(((Xe - m) / s).astype(np.float32))).numpy()


def main():
    D = {}
    for ds in DSS:
        f = os.path.join(RD, f"vlm_feat_oxe_{ds}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True); X = z["X"].astype(np.float32)
        Y = z["chunks"].astype(np.float32).reshape(len(z["chunks"]), -1)
        Y = Y - Y.mean(0); Y = Y / (np.sqrt((Y ** 2).mean()) + 1e-9); D[ds] = (X, Y)
    ds_ok = list(D)
    Xall = np.concatenate([D[d][0] for d in ds_ok]); Yall = np.concatenate([D[d][1] for d in ds_ok])
    U = rrr_U(Xall, Yall, K)
    C = {d: (D[d][0], (D[d][1] @ U).astype(np.float32)) for d in ds_ok}      # (X, c) per embodiment
    # context = VLM projected to 32
    xm = Xall.mean(0); _, _, Vx = np.linalg.svd(Xall - xm, full_matrices=False); Pc = Vx[:32].T.astype(np.float32)
    ctx = {d: (((C[d][0] - xm) @ Pc)).astype(np.float32) for d in ds_ok}

    Ns = [0, 50, 100, 200]
    print(f"embodiments={ds_ok} K={K}\nadapter data-efficiency: R^2 on held-out embodiment test set vs N adapt samples\n", flush=True)
    for held in ds_ok:
        src = [d for d in ds_ok if d != held]
        Xs = np.concatenate([C[d][0] for d in src]); Cs = np.concatenate([C[d][1] for d in src])
        Pref = None  # reference prior on sources
        Xh, Ch, ctxh = C[held][0], C[held][1], ctx[held]
        rng = np.random.default_rng(0); idx = rng.permutation(len(Xh)); cut = int(0.7 * len(Xh))
        adapt_pool, test = idx[:cut], idx[cut:]
        # P_ref predictions
        guess_all = fit_mlp(Xs.shape[1], K, Xs, Cs, Xh)                       # P_ref(VLM(D)) -> shared-c guess
        base = r2(guess_all[test], Ch[test])
        line = f"  held={held:22s} | no-adapt(P_ref)={base:+.3f}"
        for N in Ns:
            if N == 0:
                continue
            if N > len(adapt_pool):
                line += f" | N={N}: n/a"; continue
            a = adapt_pool[:N]
            # little transport: [P_ref guess, ctx] -> D's c, fit on N samples
            Ain = np.concatenate([guess_all, ctx[held]], 1)
            pred = fit_mlp(Ain.shape[1], K, Ain[a], Ch[a], Ain[test], hid=128, steps=3000)
            line += f" | N={N}: {r2(pred, Ch[test]):+.3f}"
        # own-data upper bound (full adapt pool, direct prior)
        ub = r2(fit_mlp(Xh.shape[1], K, Xh[adapt_pool], Ch[adapt_pool], Xh[test]), Ch[test])
        line += f" | UB(full,{len(adapt_pool)})={ub:+.3f}"
        print(line, flush=True)
    print("NEW_EMB_DONE", flush=True)


if __name__ == "__main__":
    main()
