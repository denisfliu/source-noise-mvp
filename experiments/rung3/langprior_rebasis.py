"""Retarget the enumeration-free language prior to a different pin basis without redoing the
VLM pass. The cached rows (langprior_feats.npz) hold the language embedding E, model state S,
episode index and progress fraction — all basis-independent. Only the targets c = U^T y
change, and those are recomputed from the demos on CPU.

    UPATH=pin_U_half8_gate.npy OUT=langprior_half8.pt python langprior_rebasis.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pin_basis

RD = os.path.dirname(os.path.abspath(__file__))
UPATH = os.environ.get("UPATH", f"{RD}/pin_U_half8_gate.npy")
# the flow whose VLM produced langprior_feats.npz (langprior_pipeline.py CKPT)
CKPT_DEFAULT = "/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999"
H, AD = 50, 32
STRIDE = 6            # must match langprior_pipeline
TASKS = ["go through the center gate from the left and hover over the stuffed animal",
         "go through the center gate from the right and hover over the stuffed animal",
         "go through the gate on the left and hover over the stuffed animal",
         "go through the gate on the right and hover over the stuffed animal"]


def main():
    import openpi.shared.normalize as NZ
    import openpi.training.config as C
    from openpi import transforms as T
    from openpi.transforms import NormStats

    U = np.load(UPATH).astype(np.float32)
    K = U.shape[1]
    out_path = os.environ.get("OUT", f"{RD}/langprior_half8.pt")
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

    z = np.load(os.environ.get("CACHE", f"{RD}/langprior_feats.npz"))
    E, S, ep, frac = z["E"], z["S"], z["ep"], z["frac"]
    print(f"cached rows {E.shape}, {len(set(ep.tolist()))} episodes", flush=True)

    # recompute targets under the new basis, walking each episode exactly as the cache did
    Yc = np.zeros((len(E), K), np.float32)
    filled = 0
    for i in sorted(set(ep.tolist())):
        d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32)
        Tn = len(d["state"])
        rows = np.where(ep == i)[0]
        ts = list(range(0, Tn - 5, STRIDE))
        if len(ts) != len(rows):
            raise SystemExit(f"row count mismatch for ep {i}: {len(ts)} vs {len(rows)}")
        for r, t in zip(rows, ts):
            ch = np.zeros((H, AD), np.float32)
            m = min(H, len(ac) - t)
            ch[:m, :7] = ac[t:t + m]
            if m < H and os.environ.get("SNMVP_ZERO_PAD_ACTIONS") != "1":
                ch[m:, :7] = ac[min(t + m, len(ac)) - 1]
            Yc[r] = (nrm({"actions": ch})["actions"].reshape(-1)) @ U
            filled += 1
    print(f"recomputed {filled} targets under K={K}", flush=True)

    rng = np.random.default_rng(0)
    tr_eps = set(rng.permutation(200)[:160].tolist())
    tr = np.array([e in tr_eps for e in ep])
    Em = E[tr].mean(0)
    _, _, Vt = np.linalg.svd(E[tr] - Em, full_matrices=False)
    P = Vt[:64].T.astype(np.float32)
    E64 = (E - Em) @ P
    task_of = ep // 50
    within = np.concatenate([E64[tr & (task_of == k)] - E64[tr & (task_of == k)].mean(0) for k in range(4)])
    emb_sig = within.std(0)
    X = np.concatenate([S, E64], 1).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    nstate = S.shape[1]
    Xt = torch.tensor((X[tr] - mu) / sd); Yt = torch.tensor(Yc[tr])
    TAILW = float(os.environ.get("TAILW", "1"))   # no tail weighting by default:
    # it was compensating for the padding defect, not fixing a modelling gap (Denis, 2026-08-11)
    W = torch.tensor(np.where(frac[tr] >= 0.75, TAILW, 1.0).astype(np.float32))
    esig = torch.tensor((emb_sig / sd[nstate:]).astype(np.float32))
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, K))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for e in range(400):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            j = p[i:i + 1024]; xb = Xt[j].clone()
            xb[:, :nstate] += 0.1 * torch.randn_like(xb[:, :nstate])
            xb[:, nstate:] += 2.0 * esig * torch.randn_like(xb[:, nstate:])
            opt.zero_grad()
            (W[j] * ((net(xb) - Yt[j]) ** 2).mean(1)).sum().div(W[j].sum()).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor((X - mu) / sd)).numpy()
    r2 = lambda m: 1 - ((Yc[m] - pred[m]) ** 2).sum() / (((Yc[m] - Yc[m].mean(0)) ** 2).sum() + 1e-9)
    print(f"held c-R2 {r2(~tr):+.4f} (train {r2(tr):+.4f})", flush=True)
    for nm, lo, hi in (("early", 0.0, 0.5), ("transit", 0.5, 0.75), ("tail", 0.75, 1.01)):
        m = (~tr) & (frac >= lo) & (frac < hi)
        print(f"  phase {nm:8s} held c-R2 {r2(m):+.4f} n={int(m.sum())}", flush=True)
    torch.save({"kind": "lang_prior", "in_dim": X.shape[1], "hidden": [256, 256], "K": K,
                "nstate": nstate, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                "Em": Em.astype(np.float32), "P": P, "state_dict": net.state_dict(),
                **pin_basis.stamp(UPATH, feat_ckpt=os.environ.get("FEAT_CKPT", CKPT_DEFAULT))}, out_path)
    print("LANGPRIOR_REBASIS_DONE", out_path, flush=True)


if __name__ == "__main__":
    main()
