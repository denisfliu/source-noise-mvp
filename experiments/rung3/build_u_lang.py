"""E1b: does an instruction-aligned pin subspace exist? PCA-U captures dominant motion (mostly
state/phase). Build U_lang = top-K eigenvectors of the BETWEEN-instruction scatter (directions
along which per-instruction mean actions differ) over the single-scene suites, and compare the
between/within variance split of c under PCA-U vs U_lang. Uses the law of total variance
analytically via accumulated total and between-instruction covariances (one pass). Saves U_lang."""
import os

import numpy as np
import lerobot.common.datasets.lerobot_dataset as L
import openpi.shared.normalize as N

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD, D, K = 50, 32, 1600, 5
ns = N.load(os.path.join(RD, "norm_shared_libero"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
U_pca = np.load(os.path.join(RD, "pin_U_pca_k5_shared.npy")).astype(np.float64)


def chunkX(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    segn = (seg - amean[:r]) / (astd[:r] + 1e-6)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = segn
    return ch.reshape(-1)


def main():
    ds = L.LeRobotDataset("physical-intelligence/libero")
    frm, to = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    hf = ds.hf_dataset.with_format("numpy")
    acts = np.asarray(hf["actions"], dtype=np.float32)
    tidx = np.asarray(hf["task_index"])
    targets = set(range(10, 40))
    XtX = np.zeros((D, D)); Xsum = np.zeros(D); Ntot = 0
    tsum, tcnt = {}, {}
    for e in range(len(frm)):
        a, b = frm[e], to[e]
        ti = int(tidx[a])
        if ti not in targets:
            continue
        ep = acts[a:b]
        Xep = np.stack([chunkX(ep[t:t + H]) for t in range(len(ep))]).astype(np.float64)
        XtX += Xep.T @ Xep; Xsum += Xep.sum(0); Ntot += len(Xep)
        tsum[ti] = tsum.get(ti, 0) + Xep.sum(0); tcnt[ti] = tcnt.get(ti, 0) + len(Xep)

    m = Xsum / Ntot
    Sig_tot = XtX / Ntot - np.outer(m, m)
    Sig_btw = np.zeros((D, D))
    for ti in tsum:
        mk = tsum[ti] / tcnt[ti]
        Sig_btw += (tcnt[ti] / Ntot) * np.outer(mk - m, mk - m)

    w, V = np.linalg.eigh(Sig_btw)
    U_lang = V[:, ::-1][:, :K]   # top-K between-instruction directions (max between magnitude)

    # LDA: max between/within ratio -> directions where language most reliably sets c
    Sig_wth = Sig_tot - Sig_btw
    lam = 1e-3 * np.trace(Sig_wth) / D
    A = np.linalg.solve(Sig_wth + lam * np.eye(D), Sig_btw)
    wl, Vl = np.linalg.eig(A)
    order = np.argsort(-wl.real)
    Q, _ = np.linalg.qr(Vl[:, order[:K]].real)   # orthonormal basis of the LDA subspace
    U_lda = Q[:, :K]

    def frac(U):
        tot = np.trace(U.T @ Sig_tot @ U)
        btw = np.trace(U.T @ Sig_btw @ U)
        return btw, tot, btw / tot

    for name, U in [("PCA-U (max variance)", U_pca), ("U_lang (max between mag)", U_lang),
                    ("U_lda (max between/within)", U_lda)]:
        btw, tot, f = frac(U)
        print(f"{name:>30}: between/language={f*100:5.1f}%  (captured total var={tot:.3f})")
    np.save(os.path.join(RD, "pin_U_lang_k5_shared.npy"), U_lang.astype(np.float32))
    np.save(os.path.join(RD, "pin_U_lda_k5_shared.npy"), U_lda.astype(np.float32))
    print("E1b_DONE saved pin_U_lang_k5_shared.npy, pin_U_lda_k5_shared.npy")


if __name__ == "__main__":
    main()
