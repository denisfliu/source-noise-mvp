"""Domain-invariant pin subspace: build U to capture the shared SUBSTANCE of the action (instruction)
and suppress the sim-vs-real execution difference, so c means the same thing in both domains and a sim-
fit prior transfers zero-shot. Current U (variance/RRR) aligns to the domain difference (c-domain-ratio
~4). We use a little real data ONLY to align the subspace (not to fit the prior), then test zero-shot
(VLM prior fit on SIM, eval REAL). Methods:
  base        PCA of sim action chunks (variance basis, the failing baseline)
  proj_domain remove the matched-pair sim-vs-real difference directions, then PCA in the complement
  gen_eig     max between-instruction / between-domain scatter (Fisher ratio, domain = nuisance)
  affine_T    baseline c + a learned affine map sim-c -> real-c (minimal-effort translation)
Reports per method: c-domain-ratio (invariant if <1), left-right separation kept, zero-shot & few-shot
c-R^2. Reuses cached drone_vlm_feat.npz (VLM feats) + recomputed action chunks."""
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


DISP_H = [12, 25, 37, 49]      # horizons (steps) for integrated displacement
DISP_D = [0, 1, 2, 3]          # control dims x,y,z,yaw


def seg_to_disp(seg):
    """Integrated displacement = cumsum of normalized actions, sampled at horizons on control dims.
    This is 'where the action goes' (the route/substance), a linear coordinate of the action chunk."""
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    chn = (seg[:, :r] - amean[:r]) / (astd[:r] + 1e-6)
    cs = np.cumsum(chn, axis=0)                                   # [H, r] displacement
    return np.concatenate([cs[h, DISP_D] for h in DISP_H]).astype(np.float32)


def load(raw):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    Y, CD, G, PR = [], [], [], []
    for k in sorted(meta):
        if meta[k]["lang"] not in (LEFT, RIGHT):
            continue
        d = np.load(os.path.join(raw, k + ".npz")); acts = d["action"].astype(np.float32); T = len(acts)
        for t in range(0, T, 6):
            Y.append(seg_to_Y(acts[t:])); CD.append(seg_to_disp(acts[t:]))
            G.append(0 if meta[k]["lang"] == LEFT else 1); PR.append(t / max(T - 1, 1))
    return np.asarray(Y, np.float32), np.asarray(CD, np.float32), np.asarray(G), np.asarray(PR, np.float32)


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def fit_mlp(Xtr, Ytr, Xte, steps=3500):
    import torch, torch.nn as nn
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.SiLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(((Xtr - m) / s).astype(np.float32)), torch.tensor(Ytr.astype(np.float32))
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,)); loss = ((net(xt[b]) - yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    import torch as T
    with T.no_grad():
        return net(T.tensor(((Xte - m) / s).astype(np.float32))).numpy()


def bin_diffs(Zs, gs, ps, Zr, gr, pr, nb=10):
    """matched (instruction, progress-bin) mean differences real-sim; and instruction means (pooled)."""
    diffs, imeans = [], {}
    for g in (0, 1):
        gm = np.concatenate([Zs[gs == g], Zr[gr == g]]).mean(0); imeans[g] = gm
        for b in range(nb):
            lo, hi = b / nb, (b + 1) / nb
            ms = Zs[(gs == g) & (ps >= lo) & (ps < hi)]; mr = Zr[(gr == g) & (pr >= lo) & (pr < hi)]
            if len(ms) >= 3 and len(mr) >= 3:
                diffs.append(mr.mean(0) - ms.mean(0))
    return np.array(diffs), imeans


def main():
    Ys, CDs, gs, ps = load(os.path.join(RD, "data_gate_synth"))
    Yr, CDr, gr, pr = load(os.path.join(RD, "data_gate_real"))
    z = np.load(os.path.join(RD, "drone_vlm_feat.npz")); XS, XR = z["XS"], z["XR"]
    assert len(XS) == len(Ys) and len(XR) == len(Yr), f"{XS.shape}{Ys.shape}{XR.shape}{Yr.shape}"
    print(f"sim={len(Ys)} real={len(Yr)} smooth={SMOOTH}", flush=True)

    # reduce action-chunk space (pooled PCA) for conditioning
    Yall = np.concatenate([Ys, Yr]); ym = Yall.mean(0)
    _, _, Vt = np.linalg.svd(Yall - ym, full_matrices=False); P = Vt[:KDIM].T.astype(np.float32)
    Zs = (Ys - ym) @ P; Zr = (Yr - ym) @ P                       # reduced action features

    D, imeans = bin_diffs(Zs, gs, ps, Zr, gr, pr)
    instr_axis = imeans[1] - imeans[0]

    def make_U(kind):
        if kind == "base":
            _, _, v = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False); return v[:K].T
        if kind == "proj_domain":
            _, _, vd = np.linalg.svd(D - D.mean(0), full_matrices=False); Q = vd[:3].T  # domain nuisance dirs
            Zp = Zs - (Zs @ Q) @ Q.T
            _, _, v = np.linalg.svd(Zp - Zp.mean(0), full_matrices=False); return v[:K].T
        if kind == "gen_eig":
            Sd = (D - D.mean(0)).T @ (D - D.mean(0)) / max(len(D), 1) + 1e-3 * np.eye(KDIM)
            mu = 0.5 * (imeans[0] + imeans[1]); M = np.stack([imeans[0] - mu, imeans[1] - mu])
            Si = M.T @ M
            w, V = np.linalg.eig(np.linalg.solve(Sd, Si))
            return np.real(V[:, np.argsort(-np.real(w))[:K]])
        raise ValueError

    def evalU(U, Zs, Zr, name, translate=False):
        U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-9)
        Cs, Cr = Zs @ U, Zr @ U
        ml_s, mr_s = Cs[gs == 0].mean(0), Cs[gs == 1].mean(0)
        ml_r, mr_r = Cr[gr == 0].mean(0), Cr[gr == 1].mean(0)
        dom = 0.5 * (np.linalg.norm(ml_s - ml_r) + np.linalg.norm(mr_s - mr_r)); sep = np.linalg.norm(ml_r - mr_r)
        # affine translation of predicted c: fit sim-c-binmean -> real-c-binmean, apply at eval
        Tt = None
        if translate:
            xs, xr = [], []
            for g in (0, 1):
                for b in range(10):
                    lo, hi = b / 10, (b + 1) / 10
                    a = Cs[(gs == g) & (ps >= lo) & (ps < hi)]; c = Cr[(gr == g) & (pr >= lo) & (pr < hi)]
                    if len(a) >= 3 and len(c) >= 3:
                        xs.append(a.mean(0)); xr.append(c.mean(0))
            Xs_ = np.concatenate([np.array(xs), np.ones((len(xs), 1))], 1)
            Tt, *_ = np.linalg.lstsq(Xs_, np.array(xr), rcond=None)  # sim-c -> real-c
        zs_pred = fit_mlp(XS, Cs, XR)
        if translate:
            zs_pred = np.concatenate([zs_pred, np.ones((len(zs_pred), 1))], 1) @ Tt
        zs = r2(zs_pred, Cr)
        rng = np.random.default_rng(0); idx = rng.permutation(len(Cr)); cut = int(0.7 * len(Cr))
        fs = r2(fit_mlp(XR[idx[:cut]], Cr[idx[:cut]], XR[idx[cut:]]), Cr[idx[cut:]])
        print(f"  {name:16s} | c-domain-ratio={dom/(sep+1e-9):5.2f} | L-R sep={sep:5.2f} | ZERO-SHOT R2={zs:+.3f} | few-shot={fs:+.3f}", flush=True)

    def eval_c(Cs, Cr, name, translate=False):
        ml_s, mr_s = Cs[gs == 0].mean(0), Cs[gs == 1].mean(0)
        ml_r, mr_r = Cr[gr == 0].mean(0), Cr[gr == 1].mean(0)
        dom = 0.5 * (np.linalg.norm(ml_s - ml_r) + np.linalg.norm(mr_s - mr_r)); sep = np.linalg.norm(ml_r - mr_r)
        pred = fit_mlp(XS, Cs, XR)
        if translate:  # affine sim-c -> real-c fit on matched (instr,progress) bin means
            xs, xr = [], []
            for g in (0, 1):
                for b in range(10):
                    lo, hi = b / 10, (b + 1) / 10
                    a = Cs[(gs == g) & (ps >= lo) & (ps < hi)]; c = Cr[(gr == g) & (pr >= lo) & (pr < hi)]
                    if len(a) >= 3 and len(c) >= 3:
                        xs.append(a.mean(0)); xr.append(c.mean(0))
            Xb = np.concatenate([np.array(xs), np.ones((len(xs), 1))], 1)
            Tt, *_ = np.linalg.lstsq(Xb, np.array(xr), rcond=None)
            pred = np.concatenate([pred, np.ones((len(pred), 1))], 1) @ Tt
        zs = r2(pred, Cr)
        rng = np.random.default_rng(0); idx = rng.permutation(len(Cr)); cut = int(0.7 * len(Cr))
        fs = r2(fit_mlp(XR[idx[:cut]], Cr[idx[:cut]], XR[idx[cut:]]), Cr[idx[cut:]])
        print(f"  {name:18s} | c-domain-ratio={dom/(sep+1e-9):5.2f} | L-R sep={sep:5.2f} | ZERO-SHOT R2={zs:+.3f} | few-shot={fs:+.3f}", flush=True)

    print("\n== domain-invariant U (real used only to ALIGN subspace; zero-shot prior fit on SIM) ==", flush=True)
    evalU(make_U("base"), Zs, Zr, "base(PCA-sim)")
    evalU(make_U("proj_domain"), Zs, Zr, "proj_domain")
    evalU(make_U("gen_eig"), Zs, Zr, "gen_eig")
    evalU(make_U("base"), Zs, Zr, "base+affineT", translate=True)
    # displacement basis: c = integrated 'where it goes' (route substance), standardized on sim
    dm, ds = CDs.mean(0), CDs.std(0) + 1e-6
    eval_c((CDs - dm) / ds, (CDr - dm) / ds, "disp(route)")
    eval_c((CDs - dm) / ds, (CDr - dm) / ds, "disp+affineT", translate=True)
    print("INV_U_DONE", flush=True)


if __name__ == "__main__":
    main()
