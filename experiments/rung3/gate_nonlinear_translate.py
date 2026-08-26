"""Nonlinear sim->real translation on the pin coordinate, and the SCALABILITY test. Pipeline: prior P
fit on SIM (VLM->sim-c); on a real obs it predicts a sim-c; a translation T maps sim-c -> real-c. T is
learned once from a little real data. The affine T reached break-even; here T is a small MLP (captures
the nonlinear, phase-dependent domain shift). The decisive question for scaling across tasks/embodiments:
is the sim->real map INSTRUCTION-AGNOSTIC? We fit T on ONE instruction's real data and test zero-shot on
the HELD-OUT instruction. If it transfers, you learn the domain map once and reuse it for new tasks.
Arms per split: P-alone (no translation) | P+affine | P+MLP | few-shot (prior refit on real, reference)."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD, SMOOTH, KDIM, K = 50, 32, 7, 64, 5
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
    meta = json.load(open(os.path.join(raw, "meta.json")))
    Y, G = [], []
    for k in sorted(meta):
        if meta[k]["lang"] not in (LEFT, RIGHT):
            continue
        d = np.load(os.path.join(raw, k + ".npz")); acts = d["action"].astype(np.float32); T = len(acts)
        for t in range(0, T, 6):
            Y.append(seg_to_Y(acts[t:])); G.append(0 if meta[k]["lang"] == LEFT else 1)
    return np.asarray(Y, np.float32), np.asarray(G)


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def mlp(din, dout, X, Y, Xe, hid=128, steps=4000, wd=1e-4):
    import torch, torch.nn as nn
    m, s = X.mean(0), X.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(din, hid), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, dout))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd)
    xt, yt = torch.tensor(((X - m) / s).astype(np.float32)), torch.tensor(Y.astype(np.float32))
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,)); loss = ((net(xt[b]) - yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    import torch as T
    with T.no_grad():
        return net(T.tensor(((Xe - m) / s).astype(np.float32))).numpy()


def affine(Xtr, Ytr, Xte):
    Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1); W, *_ = np.linalg.lstsq(Xb, Ytr, rcond=None)
    return np.concatenate([Xte, np.ones((len(Xte), 1))], 1) @ W


def main():
    Ys, gs = load(os.path.join(RD, "data_gate_synth"))
    Yr, gr = load(os.path.join(RD, "data_gate_real"))
    z = np.load(os.path.join(RD, "drone_vlm_feat.npz")); XS, XR = z["XS"], z["XR"]
    Yall = np.concatenate([Ys, Yr]); ym = Yall.mean(0)
    _, _, Vt = np.linalg.svd(Yall - ym, full_matrices=False); P = Vt[:KDIM].T.astype(np.float32)
    Zs, Zr = (Ys - ym) @ P, (Yr - ym) @ P
    _, _, v = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False); U = v[:K].T
    Cs, Cr = Zs @ U, Zr @ U                                            # base c (sim & real)
    print(f"sim={len(Cs)} real={len(Cr)} K={K}", flush=True)

    # sim prior P (VLM -> sim-c); its prediction on real obs = input to the translation
    Cr_pred = mlp(XS.shape[1], K, XS, Cs, XR)                          # P(VLM(real)) in sim-c space
    Cs_pred = mlp(XS.shape[1], K, XS, Cs, XS)                          # P on sim (for building T inputs on sim too)

    def arms(tr, te, tag):
        # translation T fit on TRAIN real: input = sim-prior prediction on real, target = real c
        pa = affine(Cr_pred[tr], Cr[tr], Cr_pred[te])
        pn = mlp(K, K, Cr_pred[tr], Cr[tr], Cr_pred[te], hid=128, steps=4000)
        fs = mlp(XR.shape[1], K, XR[tr], Cr[tr], XR[te])              # few-shot: prior refit on real (reference)
        print(f"  {tag:22s} | P-alone={r2(Cr_pred[te], Cr[te]):+.3f} | P+affine={r2(pa, Cr[te]):+.3f} "
              f"| P+MLP={r2(pn, Cr[te]):+.3f} | few-shot={r2(fs, Cr[te]):+.3f}  (n_tr={tr.sum()} n_te={te.sum()})", flush=True)

    N = len(Cr); rng = np.random.default_rng(0); idx = rng.permutation(N)
    trm = np.zeros(N, bool); trm[idx[:int(0.7 * N)]] = True
    print("\n== IN-DISTRIBUTION (T fit on 70% real, both instructions) ==", flush=True)
    arms(trm, ~trm, "in-dist")
    print("\n== CROSS-INSTRUCTION (T fit on ONE instruction, tested on the UNSEEN one) ==", flush=True)
    arms(gr == 0, gr == 1, "fit LEFT -> test RIGHT")
    arms(gr == 1, gr == 0, "fit RIGHT -> test LEFT")
    print("NLT_DONE", flush=True)


if __name__ == "__main__":
    main()
