"""Does a PROGRESS-NORMALIZED trajectory c unlock zero-shot sim->real? The raw-action c is dominated by
a sampling-rate artifact: a fixed 50-step chunk covers a different fraction of the path in sim (long MPC
episodes) vs real (shorter teleop), so 'the next 50 actions' is not comparable across domains. Here c is
built from FUTURE POSE DISPLACEMENT resampled on normalized progress (domain-invariant if the paths
match): at anchor progress p, c = [pose(p+d)-pose(p) for d in OFFS] over dims x,y,z,yaw, projected to K
via PCA fit on the pooled data. We then repeat the zero-shot transfer: fit (progress,lang)->c and
(state,lang)->c on SIM, evaluate on REAL, versus the in-domain upper bound. Compare against the raw-
action c numbers (Z_state=-0.78, Z_prog=-0.17)."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = "go through the gate on the right and hover over the stuffed animal"
DIMS = [0, 1, 2, 3]
OFFS = [0.05, 0.1, 0.2, 0.35, 0.5]  # future progress offsets
import openpi.shared.normalize as NZ
ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
smean, sstd = np.asarray(ns["state"].mean), np.asarray(ns["state"].std)


def load(raw, lang):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    out = []
    for k in sorted(meta):
        if meta[k]["lang"] != lang:
            continue
        d = np.load(os.path.join(raw, k + ".npz"))
        out.append((d["state"].astype(np.float32), d["action"].astype(np.float32)))
    return out


def build(eps, gate):
    """Return per-anchor: normalized state, progress, gate onehot, and raw future-displacement c_raw."""
    S, P, G, Craw = [], [], [], []
    for states, acts in eps:
        T = len(states); pose = states[:, DIMS]
        pr = np.arange(T) / max(T - 1, 1)
        for t in range(0, T, 3):
            p = pr[t]
            disp = []
            for d in OFFS:
                j = np.searchsorted(pr, min(p + d, 1.0)); j = min(j, T - 1)
                disp.append(pose[j] - pose[t])
            S.append((states[t] - smean) / (sstd + 1e-6)); P.append(p)
            G.append([1, 0] if gate == 0 else [0, 1]); Craw.append(np.concatenate(disp))
    return (np.asarray(S, np.float32), np.asarray(P, np.float32)[:, None],
            np.asarray(G, np.float32), np.asarray(Craw, np.float32))


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def main():
    RR, SS = os.path.join(RD, "data_gate_real"), os.path.join(RD, "data_gate_synth")
    print("episode length: real median/mean =",
          int(np.median([len(a) for _, a in load(RR, LEFT) + load(RR, RIGHT)])),
          round(np.mean([len(a) for _, a in load(RR, LEFT) + load(RR, RIGHT)]), 1),
          "| sim =", int(np.median([len(a) for _, a in load(SS, LEFT) + load(SS, RIGHT)])),
          round(np.mean([len(a) for _, a in load(SS, LEFT) + load(SS, RIGHT)]), 1))

    Ss0, Ps0, Gs0, Cs0 = build(load(SS, LEFT), 0); Ss1, Ps1, Gs1, Cs1 = build(load(SS, RIGHT), 1)
    Sr0, Pr0, Gr0, Cr0 = build(load(RR, LEFT), 0); Sr1, Pr1, Gr1, Cr1 = build(load(RR, RIGHT), 1)
    Ss, Ps, Gs, Craw_s = np.concatenate([Ss0, Ss1]), np.concatenate([Ps0, Ps1]), np.concatenate([Gs0, Gs1]), np.concatenate([Cs0, Cs1])
    Sr, Pr, Gr, Craw_r = np.concatenate([Sr0, Sr1]), np.concatenate([Pr0, Pr1]), np.concatenate([Gr0, Gr1]), np.concatenate([Cr0, Cr1])

    # PCA basis (K=5) on pooled displacement, standardized
    Call = np.concatenate([Craw_s, Craw_r]); mu, sd = Call.mean(0), Call.std(0) + 1e-6
    Zc = (Call - mu) / sd
    _, _, Vt = np.linalg.svd(Zc - Zc.mean(0), full_matrices=False)
    Upca = Vt[:5].T.astype(np.float32)
    to_c = lambda Craw: ((Craw - mu) / sd) @ Upca
    Cs, Cr = to_c(Craw_s), to_c(Craw_r)

    # domain-invariance of trajectory c: |mean c_sim - mean c_real| per gate vs left-right sep
    ml_s, mr_s = Cs[Gs[:, 0] == 1].mean(0), Cs[Gs[:, 1] == 1].mean(0)
    ml_r, mr_r = Cr[Gr[:, 0] == 1].mean(0), Cr[Gr[:, 1] == 1].mean(0)
    dom = 0.5 * (np.linalg.norm(ml_s - ml_r) + np.linalg.norm(mr_s - mr_r)); sep = np.linalg.norm(ml_r - mr_r)
    print(f"\ntrajectory-c domain gap={dom:.3f}  left-right sep(real)={sep:.3f}  ratio={dom/(sep+1e-9):.3f}  (raw-action ratio was 3.5)")

    import torch, torch.nn as nn

    def mlp(din):
        return nn.Sequential(nn.Linear(din, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 5))

    def fit_eval(Xtr, Ytr, Xte, Yte, name, steps=4000):
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
        net = mlp(Xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        xt, yt = torch.tensor(((Xtr - m) / s).astype(np.float32)), torch.tensor(Ytr)
        for _ in range(steps):
            b = torch.randint(0, len(xt), (256,))
            loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(((Xte - m) / s).astype(np.float32))).numpy()
        print(f"{name:48s} subspace R^2 (real) = {r2(pred, Yte):.3f}")

    rng = np.random.default_rng(0); idx = rng.permutation(len(Cr)); cut = int(0.7 * len(Cr)); tr, te = idx[:cut], idx[cut:]
    XSr, XSs = np.concatenate([Sr, Gr], 1), np.concatenate([Ss, Gs], 1)
    XPr, XPs = np.concatenate([Pr, Gr], 1), np.concatenate([Ps, Gs], 1)
    print("\n-- raw trajectory-c --")
    fit_eval(XSr[tr], Cr[tr], XSr[te], Cr[te], "UB_real  MLP(state,lang) fit REAL -> REAL")
    fit_eval(XPr[tr], Cr[tr], XPr[te], Cr[te], "ref_prog MLP(progress,lang) fit REAL -> REAL")
    fit_eval(XSs, Cs, XSr, Cr, "Z_state  MLP(state,lang) fit SIM -> REAL")
    fit_eval(XPs, Cs, XPr, Cr, "Z_prog   MLP(progress,lang) fit SIM -> REAL")

    # remove each domain's GLOBAL MEAN from c (a low-dim domain offset; left-vs-right structure preserved).
    # For zero-shot this offset needs only a few real samples to estimate (few-shot domain calibration).
    mcs, mcr = Cs.mean(0), Cr.mean(0)
    Csd, Crd = Cs - mcs, Cr - mcr
    domoff = np.linalg.norm(mcs - mcr)
    print(f"\n-- domain-centered trajectory-c (removed offset |mean_sim-mean_real|={domoff:.3f}) --")
    fit_eval(XSs, Csd, XSr, Crd, "Z_state* MLP(state,lang) fit SIM -> REAL (centered)")
    fit_eval(XPs, Csd, XPr, Crd, "Z_prog*  MLP(progress,lang) fit SIM -> REAL (centered)")
    print("TRAJFIX_DONE")


if __name__ == "__main__":
    main()
