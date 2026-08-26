"""The 'little flow' reconstruction: adapt sim->real as an OPTIMAL-TRANSPORT flow on the pin coordinate,
conditioned on VLM context, instead of an MLP regressor. A small velocity field v(c_t, t, ctx) transports
the sim prior's guess (x0 = P_sim(VLM(real_obs))) to real-c (x1), trained by flow matching (linear path,
target velocity x1-x0). Learning a new domain/embodiment = fitting this little flow; the big VLA is frozen.
We compare, on drone c: P-alone | P+MLP (regression translation) | P+flow (this) | few-shot ref, both in-
distribution and CROSS-INSTRUCTION (fit on one instruction, test on the unseen one) -- the generalization
test the MLP failed. Reuses cached drone_vlm_feat.npz + base PCA c."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD, SMOOTH, KDIM, K, CTX = 50, 32, 7, 64, 5, 32
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = "go through the gate on the right and hover over the stuffed animal"
import openpi.shared.normalize as NZ
ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
import torch
import torch.nn as nn


def seg_to_Y(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    k = np.ones(SMOOTH, np.float32) / SMOOTH
    seg = np.stack([np.convolve(seg[:, j], k, "same") for j in range(r)], 1)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def load(raw):
    meta = json.load(open(os.path.join(raw, "meta.json"))); Y, G = [], []
    for k in sorted(meta):
        if meta[k]["lang"] not in (LEFT, RIGHT):
            continue
        d = np.load(os.path.join(raw, k + ".npz")); acts = d["action"].astype(np.float32); T = len(acts)
        for t in range(0, T, 6):
            Y.append(seg_to_Y(acts[t:])); G.append(0 if meta[k]["lang"] == LEFT else 1)
    return np.asarray(Y, np.float32), np.asarray(G)


def r2(p, y):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def mlp(din, dout, X, Y, Xe, hid=128, steps=4000):
    m, s = X.mean(0), X.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(din, hid), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, dout))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(((X - m) / s).astype(np.float32)), torch.tensor(Y.astype(np.float32))
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,)); loss = ((net(xt[b]) - yt[b]) ** 2).mean()
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


def flow_transport(x0_tr, x1_tr, ctx_tr, x0_te, ctx_te, steps=5000, nsteal=20):
    net = Flow(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    X0, X1, C = torch.tensor(x0_tr), torch.tensor(x1_tr), torch.tensor(ctx_tr)
    for _ in range(steps):
        b = torch.randint(0, len(X0), (256,))
        t = torch.rand(256, 1)
        xt = (1 - t) * X0[b] + t * X1[b]
        v = net(xt, t, C[b]); loss = ((v - (X1[b] - X0[b])) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():                                   # Euler integrate x0 -> x1 over t in [0,1]
        x = torch.tensor(x0_te); ctx = torch.tensor(ctx_te); dt = 1.0 / nsteal
        for i in range(nsteal):
            tt = torch.full((len(x), 1), i * dt)
            x = x + dt * net(x, tt, ctx)
    return x.numpy()


def main():
    Ys, gs = load(os.path.join(RD, "data_gate_synth")); Yr, gr = load(os.path.join(RD, "data_gate_real"))
    z = np.load(os.path.join(RD, "drone_vlm_feat.npz")); XS, XR = z["XS"], z["XR"]
    Yall = np.concatenate([Ys, Yr]); ym = Yall.mean(0)
    _, _, Vt = np.linalg.svd(Yall - ym, full_matrices=False); P = Vt[:KDIM].T.astype(np.float32)
    Zs, Zr = (Ys - ym) @ P, (Yr - ym) @ P
    _, _, v = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False); U = v[:K].T
    Cs, Cr = (Zs @ U).astype(np.float32), (Zr @ U).astype(np.float32)
    # VLM context projected to CTX dims (fit on pooled), standardized
    Xp = np.concatenate([XS, XR]); xm = Xp.mean(0)
    _, _, Vx = np.linalg.svd(Xp - xm, full_matrices=False); Pc = Vx[:CTX].T.astype(np.float32)
    ctxS = ((XS - xm) @ Pc); ctxR = ((XR - xm) @ Pc)
    cm, csd = ctxS.mean(0), ctxS.std(0) + 1e-6; ctxS = ((ctxS - cm) / csd).astype(np.float32); ctxR = ((ctxR - cm) / csd).astype(np.float32)
    print(f"sim={len(Cs)} real={len(Cr)} K={K} ctx={CTX}", flush=True)

    Cr_pred = mlp(XS.shape[1], K, XS, Cs, XR).astype(np.float32)     # sim prior guess on real obs (x0)

    def arms(tr, te, tag):
        pm = mlp(K, K, Cr_pred[tr], Cr[tr], Cr_pred[te])            # MLP translation
        pf = flow_transport(Cr_pred[tr], Cr[tr], ctxR[tr], Cr_pred[te], ctxR[te])  # flow transport
        fs = mlp(XR.shape[1], K, XR[tr], Cr[tr], XR[te])           # few-shot ref
        print(f"  {tag:22s} | P-alone={r2(Cr_pred[te],Cr[te]):+.3f} | P+MLP={r2(pm,Cr[te]):+.3f} "
              f"| P+flow={r2(pf,Cr[te]):+.3f} | few-shot={r2(fs,Cr[te]):+.3f}", flush=True)

    N = len(Cr); rng = np.random.default_rng(0); idx = rng.permutation(N)
    trm = np.zeros(N, bool); trm[idx[:int(0.7 * N)]] = True
    print("\n== IN-DISTRIBUTION ==", flush=True); arms(trm, ~trm, "in-dist")
    print("\n== CROSS-INSTRUCTION (fit one, test the unseen one) ==", flush=True)
    arms(gr == 0, gr == 1, "fit LEFT -> test RIGHT"); arms(gr == 1, gr == 0, "fit RIGHT -> test LEFT")
    print("FLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
