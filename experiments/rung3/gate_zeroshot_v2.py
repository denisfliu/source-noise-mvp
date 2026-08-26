"""Zero-shot v2: c is not per-instruction constant (it evolves along the trajectory), so language alone
predicts ~0 even in-domain. c ~ f(phase, gate); phase lives in state, which is the domain-shifted
quantity. This tests whether a DOMAIN-INVARIANT phase signal recovers zero-shot transfer. Arms, all
scored by pin-channel subspace R^2 on REAL chunks (shared instructions only):
  UB_real   MLP(state,lang) fit REAL(train) -> REAL(test)     in-domain upper bound (~0.66)
  Z_state   MLP(state,lang) fit SIM         -> REAL           does the state->c map transfer?
  Z_prog    MLP(progress,lang) fit SIM      -> REAL           does the c-vs-progress SHAPE transfer?
  Z_sp      MLP(state,progress,lang) fit SIM-> REAL
Also prints c-vs-progress SHAPE overlap: per (gate, progress-decile) mean c gap sim-vs-real."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32
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


def load(raw, shared):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    S, P, G, C = [], [], [], []
    for k in sorted(meta):
        if meta[k]["lang"] not in shared:
            continue
        d = np.load(os.path.join(raw, k + ".npz"))
        acts = d["action"].astype(np.float32); states = d["state"].astype(np.float32); T = len(acts)
        for t in range(0, T, 3):
            S.append((states[t] - smean) / (sstd + 1e-6)); P.append(t / max(T - 1, 1))
            G.append(shared.index(meta[k]["lang"])); C.append(seg_to_c(acts[t:]))
    g = np.asarray(G); oh = np.zeros((len(g), len(shared)), np.float32); oh[np.arange(len(g)), g] = 1
    return np.asarray(S, np.float32), np.asarray(P, np.float32)[:, None], oh, np.asarray(C, np.float32)


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def main():
    rm = json.load(open(os.path.join(RD, "data_gate_real", "meta.json")))
    sm = json.load(open(os.path.join(RD, "data_gate_synth", "meta.json")))
    shared = sorted({v["lang"] for v in rm.values()} & {v["lang"] for v in sm.values()})
    Ss, Ps, Gs, Cs = load(os.path.join(RD, "data_gate_synth"), shared)
    Sr, Pr, Gr, Cr = load(os.path.join(RD, "data_gate_real"), shared)
    print(f"sim chunks={len(Cs)} real chunks={len(Cr)} K={U.shape[1]}")

    # c-vs-progress shape overlap: mean c per (gate, decile), sim vs real
    print("\nc-vs-progress shape gap (|mean c_sim - mean c_real| per gate, progress decile):")
    for gi in range(len(shared)):
        gaps = []
        for db in range(10):
            lo, hi = db / 10, (db + 1) / 10
            ms = Cs[(Gs[:, gi] == 1) & (Ps[:, 0] >= lo) & (Ps[:, 0] < hi)]
            mr = Cr[(Gr[:, gi] == 1) & (Pr[:, 0] >= lo) & (Pr[:, 0] < hi)]
            if len(ms) and len(mr):
                gaps.append(np.linalg.norm(ms.mean(0) - mr.mean(0)))
        print(f"  gate[{gi}] mean decile gap = {np.mean(gaps):.3f}  (vs left-right sep real {np.linalg.norm(Cr[Gr[:,0]==1].mean(0)-Cr[Gr[:,1]==1].mean(0)):.3f})")

    import torch, torch.nn as nn

    def mlp(din):
        return nn.Sequential(nn.Linear(din, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 256),
                             nn.SiLU(), nn.Linear(256, U.shape[1]))

    def fit_eval(Xtr, Ytr, Xte, Yte, name, steps=4000):
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
        net = mlp(Xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        xt, yt = torch.tensor((Xtr - m) / s), torch.tensor(Ytr)
        for _ in range(steps):
            b = torch.randint(0, len(xt), (256,))
            loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor((Xte - m) / s)).numpy()
        print(f"{name:52s} subspace R^2 (real) = {r2(pred, Yte):.3f}")

    # real train/test split (episode-agnostic index split ok for UB reference)
    rng = np.random.default_rng(0); idx = rng.permutation(len(Cr)); cut = int(0.7 * len(Cr))
    tr, te = idx[:cut], idx[cut:]
    XSr = np.concatenate([Sr, Gr], 1)
    print()
    fit_eval(XSr[tr], Cr[tr], XSr[te], Cr[te], "UB_real  MLP(state,lang) fit REAL -> REAL")
    fit_eval(np.concatenate([Ss, Gs], 1), Cs, XSr, Cr, "Z_state  MLP(state,lang) fit SIM -> REAL")
    fit_eval(np.concatenate([Ps, Gs], 1), Cs, np.concatenate([Pr, Gr], 1), Cr, "Z_prog   MLP(progress,lang) fit SIM -> REAL")
    fit_eval(np.concatenate([Ss, Ps, Gs], 1), Cs, np.concatenate([Sr, Pr, Gr], 1), Cr, "Z_sp     MLP(state,progress,lang) fit SIM -> REAL")
    # also: progress prior fit on REAL (in-domain, is progress even enough?)
    fit_eval(np.concatenate([Pr, Gr], 1)[tr], Cr[tr], np.concatenate([Pr, Gr], 1)[te], Cr[te], "ref_prog MLP(progress,lang) fit REAL -> REAL")
    print("ZS_V2_DONE")


if __name__ == "__main__":
    main()
