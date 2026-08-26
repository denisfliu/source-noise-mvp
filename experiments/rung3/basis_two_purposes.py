"""U serves two purposes at once (Denis, 2026-08-10): (1) a high-variance summary of the
action chunk, (2) a target that is EASY TO PREDICT from the deployed inputs. This measures
both for every candidate basis, and adds a basis fit against the inputs we actually use.

The deployed RRR basis was fit as the top-K eigenvectors of Cov(Yhat) with
Yhat = OLS(VLM prefix features -> chunk). But the command source that flies is a function of
(model state, instruction). So the existing basis maximises predictability from the WRONG
input set. `state_rrr` and `lang_rrr` repeat the RRR construction with the deployed inputs.

Reported per basis:
  capture    fraction of chunk variance inside the subspace  (purpose 1)
  R2 state   held-out R2 of a state -> c predictor           (purpose 2, deployed inputs)
  R2 s+lang  held-out R2 of a [state, language embedding] -> c predictor
Rows and the language embedding come from langprior_feats.npz; chunks are recomputed so any
basis can be evaluated on identical rows.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RD = os.path.dirname(os.path.abspath(__file__))
H, AD, K = 50, 32, 5
STRIDE = 6


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
    E, S, ep, frac = z["E"], z["S"], z["ep"], z["frac"]
    Y = np.zeros((len(S), H * AD), np.float32)
    for i in sorted(set(ep.tolist())):
        d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32); Tn = len(d["state"])
        rows = np.where(ep == i)[0]; ts = list(range(0, Tn - 5, STRIDE))
        for r, t in zip(rows, ts):
            ch = np.zeros((H, AD), np.float32); m = min(H, len(ac) - t)
            ch[:m, :7] = ac[t:t + m]
            if m < H: ch[m:, :7] = ac[min(t + m, len(ac)) - 1]
            Y[r] = nrm({"actions": ch})["actions"].reshape(-1)
    rng = np.random.default_rng(0)
    tr_eps = set(rng.permutation(200)[:160].tolist())
    tr = np.array([e in tr_eps for e in ep])
    # 64-d language embedding, as deployed
    Em = E[tr].mean(0); _, _, Vt = np.linalg.svd(E[tr] - Em, full_matrices=False)
    E64 = (E - Em) @ Vt[:64].T
    print(f"rows {len(Y)} (train {int(tr.sum())})", flush=True)

    def rrr(X, Yt, k, lam=10.0):
        Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1).astype(np.float64)
        W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Yt)
        Yh = Xb @ W; Yc = Yh - Yh.mean(0)
        w, V = np.linalg.eigh((Yc.T @ Yc) / len(Yc))
        return V[:, ::-1][:, :k].astype(np.float32)

    Ytr = Y[tr].astype(np.float64)
    bases = {
        "deployed RRR (VLM feats)": np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32),
        "state_rrr (fit on state)": rrr(S[tr], Ytr, K),
        "lang_rrr (state+lang)": rrr(np.concatenate([S, E64], 1)[tr], Ytr, K),
        "PCA-5 (max capture)": np.linalg.svd(Y[tr] - Y[tr].mean(0), full_matrices=False)[2][:K].T,
        "half-split K=8": np.load(f"{RD}/pin_U_half8_gate.npy").astype(np.float32),
    }

    def capture(B):
        Yc = Y - Y.mean(0)
        return float((((Yc @ B) @ B.T) ** 2).sum() / (Yc ** 2).sum())

    def predict_r2(B, X):
        c = (Y @ B).astype(np.float32)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                            nn.Linear(256, c.shape[1]))
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        Xt = torch.tensor((X[tr] - mu) / sd); Yt_ = torch.tensor(c[tr])
        for _ in range(150):
            p = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 1024):
                j = p[i:i + 1024]
                opt.zero_grad(); ((net(Xt[j]) - Yt_[j]) ** 2).mean().backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pr = net(torch.tensor((X - mu) / sd)).numpy()
        m = ~tr
        return float(1 - ((c[m] - pr[m]) ** 2).sum() / (((c[m] - c[m].mean(0)) ** 2).sum() + 1e-9))

    SL = np.concatenate([S, E64], 1).astype(np.float32)
    print(f"\n{'basis':26s} {'capture':>8s} {'R2 state':>9s} {'R2 s+lang':>10s}")
    for name, B in bases.items():
        print(f"{name:26s} {capture(B)*100:7.1f}% {predict_r2(B, S):9.3f} {predict_r2(B, SL):10.3f}", flush=True)
    for name, B in bases.items():
        if name.startswith(("state_rrr", "lang_rrr")):
            np.save(f"{RD}/pin_U_{name.split()[0]}_gate.npy", B)
            print(f"saved pin_U_{name.split()[0]}_gate.npy", flush=True)
    print("\nBASIS_TWO_PURPOSES_DONE", flush=True)


if __name__ == "__main__":
    main()
