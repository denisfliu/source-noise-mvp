"""Re-fit the RRR basis on POST-FUSION features from the served checkpoint, and measure how far it
sits from the deployed one.

Two things are conflated in the deployed basis `pin_U_gate_rrr_k5`: it was fitted on PRE-fusion prefix
features (`embed_prefix`), and it was fitted with a different checkpoint's weights
(`gate_both_pin`) than anything we now serve. Post-fusion language-token pooling is what we
established for the command path, so this re-fits against that, using the same RRR recipe as
`tmp_scripts_rescue/make_u_rrr_gate.py`: OLS from features to normalized chunk, then the top-K
eigenvectors of Cov(Yhat).

Distance is reported as principal angles between subspaces, which is the only basis-invariant
comparison — two bases can be identical as subspaces while their columns differ.

  python refit_rrr_basis.py --cache langprior_feats_zp.npz --k 5
"""
import argparse
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
H = 50
STRIDE = 6


def rrr_U(X, Y, k):
    """Top-k eigenvectors of Cov(Yhat), Yhat = OLS(features -> chunk). Same recipe as the deployed."""
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Yc = Xb @ W
    Yc = Yc - Yc.mean(0)
    C = (Yc.T @ Yc) / len(Yc)
    _, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k].astype(np.float32)


def pca_U(Y, k):
    Yc = Y - Y.mean(0)
    C = (Yc.T @ Yc) / len(Yc)
    _, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k].astype(np.float32)


def principal_angles(A, B):
    """Angles in degrees between the subspaces spanned by A and B (orthonormal columns)."""
    qa = np.linalg.qr(A)[0]
    qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def chunks(ep, frac_rows, norm_dir, zero_pad=True):
    """Normalized action chunks for the cached rows, in the cache's own row order."""
    import openpi.training.config as C
    import openpi.transforms as T
    from openpi.shared.normalize import NormStats, load as load_ns
    AD = C.get_config("pi0_gate").model.action_dim

    def pads(d, dim):
        o = {}
        for k, s in d.items():
            n = len(s.mean)
            if n >= dim:
                o[k] = s
                continue
            p = dim - n
            ext = lambda a, f: None if a is None else np.concatenate(
                [np.asarray(a, np.float32), np.full(p, f, np.float32)])
            o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
        return o

    nrm = T.Normalize(pads(load_ns(norm_dir), AD), use_quantiles=False)
    Y = np.zeros((len(ep), H * AD), np.float32)
    for i in sorted(set(ep.tolist())):
        d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32)
        rows = np.where(ep == i)[0]
        ts = list(range(0, len(d["state"]) - 5, STRIDE))
        if len(ts) != len(rows):
            raise SystemExit(f"row/timestep mismatch for ep {i}: {len(ts)} vs {len(rows)}")
        for r, t in zip(rows, ts):
            ch = np.zeros((H, AD), np.float32)
            m = min(H, len(ac) - t)
            ch[:m, :7] = ac[t:t + m]
            if m < H and not zero_pad:
                ch[m:, :7] = ac[min(t + m, len(ac)) - 1]
            Y[r] = nrm({"actions": ch})["actions"].reshape(-1)
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{RD}/langprior_feats_zp.npz",
                    help="post-fusion feature cache from the SERVED checkpoint")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--norm", default="/home/dfliu/hf_bundle/gate-drone-pi0/assets/gate_nav")
    ap.add_argument("--out", default=f"{RD}/pin_U_postfusion_zp_k5.npy")
    ap.add_argument("--compare", nargs="+",
                    default=["pin_U_gate_rrr_k5.npy", "pin_U_lang_rrr_gate.npy", "pin_U_gate_k5.npy"])
    a = ap.parse_args()

    z = np.load(a.cache)
    E, ep = z["E"], z["ep"]
    Y = chunks(ep, None, a.norm)   # zero-padded, matching the corrected recipe
    rng = np.random.default_rng(0)
    tr = np.array([e in set(rng.permutation(200)[:160].tolist()) for e in ep])
    print(f"cache {os.path.basename(a.cache)}: {E.shape[0]} rows, feature dim {E.shape[1]}, "
          f"{tr.sum()} train rows")

    Unew = rrr_U(E[tr], Y[tr], a.k)
    Upca = pca_U(Y[tr], a.k)
    np.save(a.out, Unew)
    print(f"saved {a.out}\n")

    def variance_kept(U):
        Yc = Y[~tr] - Y[~tr].mean(0)
        return float(((Yc @ U) ** 2).sum() / (Yc ** 2).sum())

    print("principal angles (degrees) against the newly fitted post-fusion basis:")
    rows = [("PCA of chunks (feature-free)", Upca)]
    for f in a.compare:
        p = f if os.path.isabs(f) else f"{RD}/{f}"
        if not os.path.exists(p):
            print(f"  {f}: missing")
            continue
        U = np.load(p).astype(np.float32)
        if U.shape[1] != a.k:
            print(f"  {f}: K={U.shape[1]}, skipping (need {a.k})")
            continue
        rows.append((f, U))
    for name, U in rows:
        ang = principal_angles(Unew, U)
        print(f"  {name:32s} angles {np.round(ang, 1)}  max {ang.max():5.1f}deg  "
              f"mean {ang.mean():5.1f}deg")
    print("\nheld-out chunk variance captured by each K-dim subspace (the 'summary' purpose):")
    print(f"  {'newly fitted post-fusion':32s} {variance_kept(Unew):.4f}")
    print(f"  {'PCA of chunks (upper bound)':32s} {variance_kept(Upca):.4f}")
    for name, U in rows[1:]:
        print(f"  {name:32s} {variance_kept(U):.4f}")


if __name__ == "__main__":
    main()
