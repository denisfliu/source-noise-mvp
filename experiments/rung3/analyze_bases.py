"""Gate: which pin basis handles BOTH regimes -- state-based instructions (object/spatial: language
picks the target, motion ~ state-driven) and language-based (goal: different verbs -> different
motion)? Compare U_pca (max variance), U_lda (between/within task), and U_rrr (reduced-rank
regression: the action subspace predictable from state+language jointly).

Decisive metric V_useful(U, predictor) = fraction of TOTAL action variance that is both captured by
U and recoverable from the conditioning = sum_k Var(c_k)*R2_k / Var(a)_total, with c = U^T a and R2_k
the held-out predictability of coordinate k. This is what the pin can actually set correctly: a basis
is good for a regime iff V_useful is high there. Expectation: PCA wins state-based, LDA wins
language-based, RRR wins BOTH. Two streaming passes over the single-scene suites (10-39). CPU."""
import os

import numpy as np
import lerobot.common.datasets.lerobot_dataset as L
import openpi.shared.normalize as N

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD, D, K = 50, 32, 1600, 5
ns = N.load(os.path.join(RD, "norm_shared_libero"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
smean, sstd = np.asarray(ns["state"].mean), np.asarray(ns["state"].std)
U_pca = np.load(os.path.join(RD, "pin_U_pca_k5_shared.npy")).astype(np.float64)
U_lda = np.load(os.path.join(RD, "pin_U_lda_k5_shared.npy")).astype(np.float64)
TASKS = list(range(10, 40))
TID = {t: i for i, t in enumerate(TASKS)}
SD = len(smean)                         # state dim
DF = SD + len(TASKS) + 1                # state + task onehot + bias


def chunkX(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def feats(S, ti):
    n = len(S)
    oh = np.zeros((n, len(TASKS)))
    oh[np.arange(n), [TID[t] for t in ti]] = 1
    return np.concatenate([(S - smean) / (sstd + 1e-6), oh, np.ones((n, 1))], 1)


def main():
    ds = L.LeRobotDataset("physical-intelligence/libero")
    frm, to = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    hf = ds.hf_dataset.with_format("numpy")
    acts = np.asarray(hf["actions"], dtype=np.float32)
    states = np.asarray(hf["state"], dtype=np.float32)
    tix = np.asarray(hf["task_index"])
    eps = [(frm[e], to[e], int(tix[frm[e]])) for e in range(len(frm)) if int(tix[frm[e]]) in TID]

    # ---- pass 1: sufficient stats for RRR-U and total variance ----
    FtF = np.zeros((DF, DF)); FtX = np.zeros((DF, D)); Xsum = np.zeros(D); sumXsq = 0.0; Ntot = 0
    for a, b, ti in eps:
        X = np.stack([chunkX(acts[a:b][t:t + H]) for t in range(b - a)])
        F = feats(states[a:b], [ti] * (b - a))
        FtF += F.T @ F; FtX += F.T @ X; Xsum += X.sum(0); sumXsq += (X ** 2).sum(); Ntot += len(X)
    W = np.linalg.solve(FtF + 1e-3 * np.eye(DF), FtX)          # (DF, D)
    mF = (FtF[:, -1] / Ntot)                                   # column of means via bias row
    CovF = FtF / Ntot - np.outer(mF, mF)
    CovYh = W.T @ CovF @ W
    wv, Vv = np.linalg.eigh(CovYh)
    U_rrr = Vv[:, ::-1][:, :K]
    total_var = sumXsq / Ntot - (Xsum / Ntot) @ (Xsum / Ntot)
    np.save(os.path.join(RD, "pin_U_rrr_k5_shared.npy"), U_rrr.astype(np.float32))

    # ---- pass 2: collect c for each basis + state + task ----
    bases = {"PCA": U_pca, "LDA": U_lda, "RRR": U_rrr}
    C = {k: [] for k in bases}; S_all = []; TI_all = []
    for a, b, ti in eps:
        X = np.stack([chunkX(acts[a:b][t:t + H]) for t in range(b - a)])
        for k, U in bases.items():
            C[k].append(X @ U)
        S_all.append(states[a:b]); TI_all.append(np.full(b - a, ti))
    C = {k: np.concatenate(v) for k, v in C.items()}
    S_all = np.concatenate(S_all); TI_all = np.concatenate(TI_all)

    def r2_per_dim(F, y):                                      # held-out R^2 per column
        ntr = int(0.85 * len(F)); rng = np.random.default_rng(0)
        idx = rng.permutation(len(F)); tr, te = idx[:ntr], idx[ntr:]
        Wp, *_ = np.linalg.lstsq(F[tr], y[tr], rcond=None)
        p = F[te] @ Wp
        ss_res = ((y[te] - p) ** 2).sum(0); ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum(0) + 1e-9
        return 1 - ss_res / ss_tot

    regimes = {"object(state)": range(20, 30), "spatial(state)": range(30, 40),
               "goal(lang)": range(10, 20), "all": range(10, 40)}
    print(f"{'regime':>15} {'basis':>5} {'cover%':>7} | Vuseful%  state / language / state+lang")
    print("-" * 74)
    for rn, rng in regimes.items():
        mask = np.isin(TI_all, list(rng))
        Sr = (S_all[mask] - smean) / (sstd + 1e-6)
        oh = np.zeros((mask.sum(), len(list(rng))))
        oh[np.arange(mask.sum()), [list(rng).index(t) for t in TI_all[mask]]] = 1
        tv = total_var  # normalize all V_useful by the same global action variance
        for bk, U in bases.items():
            c = C[bk][mask]
            cover = c.var(0).sum() / tv * 100
            feats_by = {"state": np.concatenate([Sr, np.ones((len(Sr), 1))], 1),
                        "lang": np.concatenate([oh, np.ones((len(oh), 1))], 1),
                        "both": np.concatenate([Sr, oh, np.ones((len(oh), 1))], 1)}
            V = {}
            for pn, F in feats_by.items():
                r2 = np.clip(r2_per_dim(F, c), 0, None)
                V[pn] = (c.var(0) * r2).sum() / tv * 100
            print(f"{rn:>15} {bk:>5} {cover:7.2f} |  {V['state']:6.2f} / {V['lang']:6.2f} / {V['both']:6.2f}")
        print()
    print("BASES_DONE saved pin_U_rrr_k5_shared.npy")


if __name__ == "__main__":
    main()
