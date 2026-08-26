"""Why did the predictor-matched basis fly so much worse, and what does that say about what
makes a basis good? (Denis, 2026-08-10)

Compares the deployed RRR basis against lang_rrr on properties that offline capture and
predictability do not see:
  1. geometry           principal angles; per-axis expressivity; temporal weighting
  2. command SCALE      per-dim std of c (the pin writes c into the source noise, so a basis
                        whose c is far from unit scale pushes the source far from N(0,I))
  3. TASK SEPARABILITY  how far apart left-task and right-task commands sit at matched states,
                        relative to within-task spread — if a basis cannot separate the routes,
                        the command cannot select one
  4. per-task predictability  R2 restricted to right-task rows vs left-task rows
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RD = os.path.dirname(os.path.abspath(__file__))
H, AD = 50, 32
STRIDE = 6
AX = ["x", "y", "z", "yaw"]


def main():
    import openpi.shared.normalize as NZ
    import openpi.training.config as C
    from openpi import transforms as T
    from openpi.transforms import NormStats

    ns = NZ.load(os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))

    def pads(nsd, dim):
        o = {}
        for k, s in nsd.items():
            n = np.asarray(s.mean).shape[-1]
            if n >= dim:
                o[k] = s; continue
            p = dim - n
            ext = lambda a, f: None if a is None else np.concatenate(
                [np.asarray(a, np.float32), np.full(p, f, np.float32)])
            o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
        return o
    nrm = T.Normalize(pads(ns, C.get_config("pi0_gate").model.action_dim), use_quantiles=False)

    z = np.load(f"{RD}/langprior_feats.npz")
    S, ep, frac = z["S"], z["ep"], z["frac"]
    Y = np.zeros((len(S), H * AD), np.float32)
    for i in sorted(set(ep.tolist())):
        d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32); Tn = len(d["state"])
        for r, t in zip(np.where(ep == i)[0], range(0, Tn - 5, STRIDE)):
            ch = np.zeros((H, AD), np.float32); m = min(H, len(ac) - t)
            ch[:m, :7] = ac[t:t + m]
            if m < H: ch[m:, :7] = ac[min(t + m, len(ac)) - 1]
            Y[r] = nrm({"actions": ch})["actions"].reshape(-1)
    task = ep // 50                      # 0 CFL, 1 CFR, 2 LEFT, 3 RIGHT
    rng = np.random.default_rng(0)
    tr = np.array([e in set(rng.permutation(200)[:160].tolist()) for e in ep])

    B = {"deployed RRR": np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float64),
         "lang_rrr": np.load(f"{RD}/pin_U_lang_rrr_gate.npy").astype(np.float64)}

    def angles(A, Bm):
        s = np.linalg.svd(A.T @ Bm, compute_uv=False)
        return np.degrees(np.arccos(np.clip(s, -1, 1))).round(1)
    print("principal angles deployed vs lang_rrr:", angles(B["deployed RRR"], B["lang_rrr"]), flush=True)

    print("\nper-axis expressivity (fraction of a net-displacement nudge inside the subspace):")
    for name, U in B.items():
        row = []
        for j in range(4):
            m = np.zeros((H, AD)); m[:, j] = 1.0 / H; m = m.reshape(-1)
            row.append(f"{AX[j]} {np.linalg.norm(U @ (U.T @ m))/np.linalg.norm(m):.2f}")
        print(f"  {name:14s} " + "  ".join(row), flush=True)

    print("\ntemporal weighting (share of each basis vector's mass in the first vs second half):")
    for name, U in B.items():
        early = np.linalg.norm(U.reshape(H, AD, -1)[:H // 2]) ** 2
        late = np.linalg.norm(U.reshape(H, AD, -1)[H // 2:]) ** 2
        print(f"  {name:14s} early {early/(early+late)*100:.0f}%  late {late/(early+late)*100:.0f}%", flush=True)

    print("\ncommand scale — per-dim std of c over demo rows (source noise is N(0,1) per dim):")
    for name, U in B.items():
        c = Y @ U
        print(f"  {name:14s} {np.round(c.std(0), 2)}   |c| mean {np.linalg.norm(c,axis=1).mean():.1f}", flush=True)

    print("\nTASK SEPARABILITY between the left and right gate tasks (matched by progress bin):")
    for name, U in B.items():
        c = Y @ U
        seps, wins = [], []
        for lo in np.arange(0, 1.0, 0.1):
            mL = (task == 2) & (frac >= lo) & (frac < lo + 0.1)
            mR = (task == 3) & (frac >= lo) & (frac < lo + 0.1)
            if mL.sum() < 20 or mR.sum() < 20:
                continue
            seps.append(np.linalg.norm(c[mL].mean(0) - c[mR].mean(0)))
            wins.append(0.5 * (c[mL].std(0).mean() + c[mR].std(0).mean()))
        sep, win = float(np.mean(seps)), float(np.mean(wins))
        print(f"  {name:14s} between-task gap {sep:.2f}   within-task spread {win:.2f}   ratio {sep/win:.2f}", flush=True)

    print("\nper-task predictability from state+language (held-out R2):")
    E = z["E"]; Em = E[tr].mean(0)
    _, _, Vt = np.linalg.svd(E[tr] - Em, full_matrices=False)
    X = np.concatenate([S, (E - Em) @ Vt[:64].T], 1).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    for name, U in B.items():
        c = (Y @ U).astype(np.float32)
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                            nn.Linear(256, c.shape[1]))
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        Xt = torch.tensor((X[tr] - mu) / sd); Yt = torch.tensor(c[tr])
        for _ in range(150):
            p = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 1024):
                j = p[i:i + 1024]
                opt.zero_grad(); ((net(Xt[j]) - Yt[j]) ** 2).mean().backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pr = net(torch.tensor((X - mu) / sd)).numpy()
        def r2(m):
            return float(1 - ((c[m]-pr[m])**2).sum() / (((c[m]-c[m].mean(0))**2).sum() + 1e-9))
        print(f"  {name:14s} LEFT {r2((~tr)&(task==2)):+.3f}   RIGHT {r2((~tr)&(task==3)):+.3f}   "
              f"all {r2(~tr):+.3f}", flush=True)
    print("\nBASIS_DIAGNOSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
