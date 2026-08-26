"""How to find the pairing at scale: does LANGUAGE-based coupling recover the free VLM pairing? Drone
sim->real transport of c (sim & real are unpaired -- different episodes -- but share the instructions
left/right). We train a conditional flow v(c_t,t,VLM-ctx) transporting the sim-convention guess x0 =
P_sim(VLM(real_obs)) to real-c x1, under three couplings of the TRAINING pairs:
  free     x0 & x1 from the SAME obs (VLM pairing; upper reference == the paired adapter)
  random   x0 from a globally random obs (no correspondence; the failed baseline)
  lang     x0 from a random obs with the SAME INSTRUCTION (semantic pairing -- what we'd have at scale)
  lang+ctx x0 from the same-instruction obs with NEAREST VLM-ctx (semantic + phase)
If lang(+ctx) ~ free and >> random, semantic pairing stands in for the correspondence. Inference is
identical across couplings: x0 = P_sim(VLM(test obs)), integrate the flow, compare to real-c."""
import json
import os
import numpy as np
import torch
import torch.nn as nn

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD, SMOOTH, KDIM, K, CTX = 50, 32, 7, 64, 5, 32
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = "go through the gate on the right and hover over the stuffed animal"
import openpi.shared.normalize as NZ
ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)


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


def mlp(din, dout, X, Y, Xe, hid=128, steps=3500):
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
        super().__init__(); s.net = nn.Sequential(nn.Linear(K + 1 + CTX, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, K))
    def forward(s, c, t, ctx):
        return s.net(torch.cat([c, t, ctx], 1))


def flow(x0src, x1, ctx, x0e, ctxe, coupler, steps=4000, nstep=20):
    """coupler(batch_idx) -> source index array for x0 (defines the pairing)."""
    net = Flow(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    X0, X1, C = torch.tensor(x0src), torch.tensor(x1), torch.tensor(ctx)
    for _ in range(steps):
        b = torch.randint(0, len(X1), (256,))
        s0 = torch.tensor(coupler(b.numpy()))
        a, d, cc = X0[s0], X1[b], C[b]
        t = torch.rand(len(b), 1); xt = (1 - t) * a + t * d
        loss = ((net(xt, t, cc) - (d - a)) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        x = torch.tensor(x0e); cc = torch.tensor(ctxe); dt = 1.0 / nstep
        for i in range(nstep):
            x = x + dt * net(x, torch.full((len(x), 1), i * dt), cc)
    return x.numpy()


def main():
    Ys, gs = load(os.path.join(RD, "data_gate_synth")); Yr, gr = load(os.path.join(RD, "data_gate_real"))
    z = np.load(os.path.join(RD, "drone_vlm_feat.npz")); XS, XR = z["XS"], z["XR"]
    Yall = np.concatenate([Ys, Yr]); ym = Yall.mean(0)
    _, _, Vt = np.linalg.svd(Yall - ym, full_matrices=False); P = Vt[:KDIM].T.astype(np.float32)
    Zs, Zr = (Ys - ym) @ P, (Yr - ym) @ P
    _, _, v = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False); U = v[:K].T
    Cs, Cr = (Zs @ U).astype(np.float32), (Zr @ U).astype(np.float32)
    xm = np.concatenate([XS, XR]).mean(0); _, _, Vx = np.linalg.svd(np.concatenate([XS, XR]) - xm, full_matrices=False); Pc = Vx[:CTX].T.astype(np.float32)
    ctxR = (((XR - xm) @ Pc)).astype(np.float32)
    guess = mlp(XS.shape[1], K, XS, Cs, XR).astype(np.float32)          # sim-convention guess on real obs = x0
    print(f"real={len(Cr)} K={K} ctx={CTX}", flush=True)

    N = len(Cr); rng = np.random.default_rng(0); idx = rng.permutation(N); cut = int(0.7 * N)
    tr, te = idx[:cut], idx[cut:]
    g_tr = gr[tr]; groups = {0: np.where(g_tr == 0)[0], 1: np.where(g_tr == 1)[0]}
    # nearest same-instruction neighbor by ctx (within train), precomputed
    near = np.zeros(len(tr), int)
    for gi in (0, 1):
        gid = groups[gi]
        cc = ctxR[tr][gid]
        d2 = ((cc[:, None, :] - cc[None, :, :]) ** 2).sum(-1); np.fill_diagonal(d2, 1e9)
        near[gid] = gid[d2.argmin(1)]

    x0tr, Crtr, ctxtr, g_tr_arr = guess[tr], Cr[tr], ctxR[tr], gr[tr]

    def c_free(b): return b
    def c_random(b): return rng.integers(0, len(b := np.arange(len(tr))), size=len(b)) if False else rng.permutation(len(tr))[:len(b)]
    def c_random2(bi): return rng.integers(0, len(tr), size=len(bi))
    def c_lang(bi): return np.array([rng.choice(groups[g_tr_arr[i]]) for i in bi])
    def c_langctx(bi): return near[bi]

    pm = mlp(K + CTX, K, np.concatenate([x0tr, ctxtr], 1), Crtr, np.concatenate([guess[te], ctxR[te]], 1))
    print(f"  MLP(paired) R2      = {r2(pm, Cr[te]):+.3f}", flush=True)
    for name, cp in [("free", c_free), ("random", c_random2), ("lang", c_lang), ("lang+ctx", c_langctx)]:
        pf = flow(x0tr, Crtr, ctxtr, guess[te], ctxR[te], cp)
        print(f"  flow[{name:8s}] R2   = {r2(pf, Cr[te]):+.3f}", flush=True)
    print("LANG_OT_DONE", flush=True)


if __name__ == "__main__":
    main()
