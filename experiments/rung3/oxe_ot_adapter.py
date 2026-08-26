"""Unpaired OT-flow adapter for a new embodiment (the vla^2 transport, on OXE cross-embodiment). A
reference prior P_ref (VLM->shared-c) is trained on source embodiments. To adapt to a held-out robot D we
learn a conditional flow-matching transport v(c_t, t, VLM-ctx) that carries P_ref's guess (x0) to D's c
(x1), the big VLA/subspace frozen. Compares, at matched adapt-budget N: MLP adapter (paired regression) |
FLOW adapter (paired coupling) | FLOW-unpaired (x1 SHUFFLED within the batch -> distribution-level, no
per-sample correspondence -- the setting real new embodiments / sim->real actually have). If FLOW-unpaired
~ MLP, the transport needs only marginals, not pairs. Held-out robots: toto, viola, ur5 (bridge excluded:
its own c is ~unpredictable, UB 0.06)."""
import os
import numpy as np
import torch
import torch.nn as nn

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
DSS = ["bridge", "berkeley_autolab_ur5", "toto", "viola"]
GOOD = ["toto", "viola", "berkeley_autolab_ur5"]
K, CTX = 5, 32


def rrr_U(X, Y, K):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None); Yh = Xb @ W; Yc = Yh - Yh.mean(0)
    _, V = np.linalg.eigh((Yc.T @ Yc) / len(Yc)); return V[:, ::-1][:, :K].astype(np.float32)


def r2(p, y):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def mlp(din, dout, X, Y, Xe, hid=128, steps=3500):
    m, s = X.mean(0), X.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(din, hid), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, dout))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(((X - m) / s).astype(np.float32)), torch.tensor(Y.astype(np.float32))
    for _ in range(steps):
        b = torch.randint(0, len(xt), (min(256, len(xt)),)); loss = ((net(xt[b]) - yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(((Xe - m) / s).astype(np.float32))).numpy()


class Flow(nn.Module):
    def __init__(s):
        super().__init__(); s.net = nn.Sequential(nn.Linear(K + 1 + CTX, 128), nn.SiLU(),
                                                  nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, K))
    def forward(s, c, t, ctx):
        return s.net(torch.cat([c, t, ctx], 1))


def flow_adapter(x0, x1, ctx, x0e, ctxe, shuffle=False, steps=4000, nstep=20):
    net = Flow(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    X0, X1, C = torch.tensor(x0), torch.tensor(x1), torch.tensor(ctx)
    for _ in range(steps):
        b = torch.randint(0, len(X0), (min(256, len(X0)),))
        a = X0[b]; d = X1[b[torch.randperm(len(b))]] if shuffle else X1[b]; cc = C[b]
        t = torch.rand(len(b), 1); xt = (1 - t) * a + t * d
        loss = ((net(xt, t, cc) - (d - a)) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        x = torch.tensor(x0e); cc = torch.tensor(ctxe); dt = 1.0 / nstep
        for i in range(nstep):
            x = x + dt * net(x, torch.full((len(x), 1), i * dt), cc)
    return x.numpy()


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
    C = {d: (D[d][0], (D[d][1] @ U).astype(np.float32)) for d in ds_ok}
    xm = Xall.mean(0); _, _, Vx = np.linalg.svd(Xall - xm, full_matrices=False); Pc = Vx[:CTX].T.astype(np.float32)
    ctx = {d: (((C[d][0] - xm) @ Pc)).astype(np.float32) for d in ds_ok}

    Ns = [50, 100, 200]
    print(f"OT-flow adapter vs MLP; held-out in {GOOD}; K={K} ctx={CTX}\n", flush=True)
    for held in GOOD:
        src = [d for d in ds_ok if d != held]
        Xs = np.concatenate([C[d][0] for d in src]); Cs = np.concatenate([C[d][1] for d in src])
        Xh, Ch, ctxh = C[held][0], C[held][1], ctx[held]
        rng = np.random.default_rng(0); idx = rng.permutation(len(Xh)); cut = int(0.7 * len(Xh))
        pool, test = idx[:cut], idx[cut:]
        guess = mlp(Xs.shape[1], K, Xs, Cs, Xh)                                  # P_ref guess on D
        base = r2(guess[test], Ch[test])
        print(f"held={held:22s} no-adapt(P_ref)={base:+.3f}", flush=True)
        for N in Ns:
            a = pool[:N]
            Ain = np.concatenate([guess, ctxh], 1)
            pm = mlp(Ain.shape[1], K, Ain[a], Ch[a], Ain[test])                  # MLP adapter (paired)
            pf = flow_adapter(guess[a], Ch[a], ctxh[a], guess[test], ctxh[test], shuffle=False)  # flow paired
            pu = flow_adapter(guess[a], Ch[a], ctxh[a], guess[test], ctxh[test], shuffle=True)    # flow UNPAIRED
            print(f"    N={N:3d} | MLP={r2(pm,Ch[test]):+.3f} | FLOW={r2(pf,Ch[test]):+.3f} | FLOW-unpaired={r2(pu,Ch[test]):+.3f}", flush=True)
    print("OT_ADAPTER_DONE", flush=True)


if __name__ == "__main__":
    main()
