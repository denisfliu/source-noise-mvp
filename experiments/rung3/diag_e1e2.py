"""E1 + E2 diagnostics for the grid-Laplacian action representation.

Reuses the controlled task x embodiment data (bank/vertical/slalom x Panda/IIWA/UR5e,
data_taskembod/), the same per-(task,arm) zero-mean unit-RMS normalization, and the
same grid-Laplacian coherence subspace (Sigma_scene/Sigma_body top-k, w=0.5) as
taskembod_study.py.

The question: the grid-Laplacian subspace transfers across arms but leaves ~.50 rel
error (~75% variance) on single-detour tasks and fails on the slalom. Is the missing
variance (a) SHARED structure a richer transferable basis could capture, or (b)
BODY-SPECIFIC realization no shared basis can capture?

E1  Residual-transfer decomposition. Project onto the grid-Laplacian subspace fit on
    the two in-sample arms; take the residual. Fit a PCA subspace on the IN-SAMPLE
    arms' residuals and ask how well it reconstructs the HELD-OUT arm's residual.
      transfer_err ~= in_sample_err  and both << random  -> residual is SHARED (Fork A)
      transfer_err >> in_sample_err  ~= random            -> residual is REALIZATION (Fork B)

E2  Coverage-vs-transfer knee. Sweep the grid-Laplacian subspace dimension K. Report
    in-sample coverage (reconstruction of the fitting arms), transfer coverage
    (reconstruction of the held-out arm), the PCA lower bound, and the transfer GAP
    (held-out minus in-sample) at each K. The knee is the K where added modes stop
    transferring (gap widens) = the empirical structure/realization boundary.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import basis_lab as BL                  # noqa: E402
import laplacian_basis as LB            # noqa: E402

H, C = 32, 6
D = H * C
BL.H = H
TASKS = ["bank", "vertical", "slalom"]
ARMS = ["Panda", "IIWA", "UR5e"]
DATA = os.path.join(HERE, "data_taskembod")
RNG = np.random.default_rng(0)


def load(task, arm):
    ch = np.load(os.path.join(DATA, f"{task}_{arm}.npz"))["chunks"].astype(float)
    S, N = ch.shape[:2]
    X = ch.reshape(S, N, D)
    flat = X.reshape(-1, D)
    X = X - flat.mean(axis=0)
    return X / (np.sqrt((X ** 2).mean()) + 1e-9)


def coherence_U(Xby, arms, k, w=0.5):
    bmean = np.stack([Xby[a].mean(axis=1) for a in arms], axis=1)   # (S,|A|,D)
    Sb, Sw = BL.covariances(bmean, D)
    return LB.basis_gridlap(Sb, Sw, k, H, C, w)


def rel_err(X, U):
    """Relative reconstruction error of X (any leading shape) in subspace U (D,k)."""
    Xf = X.reshape(-1, D)
    proj = Xf @ (U @ U.T)
    return float(np.sqrt(((Xf - proj) ** 2).sum() / ((Xf ** 2).sum() + 1e-12)))


def pca(X, k):
    Xf = X.reshape(-1, D)
    _, _, Vt = np.linalg.svd(Xf - Xf.mean(axis=0), full_matrices=False)
    return Vt[:k].T


def randU(k):
    return np.linalg.qr(RNG.normal(size=(D, D)))[0][:, :k]


def residual(X, U):
    Xf = X.reshape(-1, D)
    return Xf - Xf @ (U @ U.T)


# --------------------------------------------------------------------------- E1
def e1(data, K=10, ms=(10, 20)):
    """Is the grid-Laplacian residual shared across arms or body-specific?"""
    out = {}
    for t in TASKS:
        # residual energy fraction = how much variance the grid-Laplacian leaves
        res_frac = []
        for held in ARMS:
            setA = [a for a in ARMS if a != held]
            U = coherence_U(data[t], setA, K)
            res_frac.append(rel_err(data[t][held], U))
        rec = {"residual_energy_frac": round(float(np.mean(res_frac)), 3), "by_m": {}}
        for m in ms:
            in_s, tr, orc, rnd = [], [], [], []
            for held in ARMS:
                setA = [a for a in ARMS if a != held]
                U = coherence_U(data[t], setA, K)
                Rin = np.concatenate([residual(data[t][a], U) for a in setA], axis=0)
                Rhd = residual(data[t][held], U)
                V = pca(Rin, m)                       # residual subspace from in-sample arms
                in_s.append(rel_err(Rin, V))          # fits the arms it was built on
                tr.append(rel_err(Rhd, V))            # transfers to the held-out arm?
                orc.append(rel_err(Rhd, pca(Rhd, m))) # best possible m-dim on held-out
                rnd.append(rel_err(Rhd, randU(m)))    # random m-dim baseline
            rec["by_m"][m] = {
                "in_sample": round(float(np.mean(in_s)), 3),
                "transfer": round(float(np.mean(tr)), 3),
                "oracle": round(float(np.mean(orc)), 3),
                "random": round(float(np.mean(rnd)), 3),
            }
        out[t] = rec
    return out


# --------------------------------------------------------------------------- E2
def e2(data, Ks=(2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96)):
    """Coverage vs transfer as the grid-Laplacian subspace grows."""
    out = {}
    for t in TASKS:
        rows = []
        Xt = np.concatenate([data[t][a] for a in ARMS], axis=0)
        for K in Ks:
            in_s, tr = [], []
            for held in ARMS:
                setA = [a for a in ARMS if a != held]
                U = coherence_U(data[t], setA, K)
                in_s.append(rel_err(np.concatenate([data[t][a] for a in setA], 0), U))
                tr.append(rel_err(data[t][held], U))
            i, r = float(np.mean(in_s)), float(np.mean(tr))
            rows.append({
                "K": K,
                "in_sample": round(i, 3),
                "transfer": round(r, 3),
                "gap": round(r - i, 3),                       # transfer penalty
                "oracle_pca": round(rel_err(Xt, pca(Xt, K)), 3),
                "random": round(rel_err(Xt, randU(K)), 3),
            })
        out[t] = rows
    return out


# -------------------------------------------------------------------------- E2b
def coherence_genU(Xby, arms, k, eps=1e-3):
    """Coherence generalized-eigenbasis over ALL orthonormal directions (not
    restricted to grid-Laplacian modes): top-k directions maximizing
    e^T Sb e / e^T Sw e (scene signal / body variation), then orthonormalized."""
    bmean = np.stack([Xby[a].mean(axis=1) for a in arms], axis=1)   # (S,|A|,D)
    Sb, Sw = BL.covariances(bmean, D)
    Sw = Sw + eps * np.trace(Sw) / D * np.eye(D)
    w, Q = np.linalg.eigh(Sw)
    Wih = Q @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-12, None))) @ Q.T   # Sw^{-1/2}
    M = Wih @ Sb @ Wih
    _, U = np.linalg.eigh(0.5 * (M + M.T))
    E = Wih @ U[:, ::-1][:, :k]                     # back to original coords, top-k
    return np.linalg.qr(E)[0][:, :k]                # orthonormalize -> pass-through


def e2b(data, Ks=(4, 8, 12, 16, 24, 32)):
    """Does a data-adaptive basis fit on the IN-SAMPLE arms transfer to the held-out
    arm, and is it more mode-efficient than the fixed grid-Laplacian?"""
    out = {}
    for t in TASKS:
        rows = []
        for K in Ks:
            glap, pcaT, genT = [], [], []
            for held in ARMS:
                setA = [a for a in ARMS if a != held]
                Xin = np.concatenate([data[t][a] for a in setA], axis=0)
                Xhd = data[t][held]
                glap.append(rel_err(Xhd, coherence_U(data[t], setA, K)))     # fixed basis
                pcaT.append(rel_err(Xhd, pca(Xin, K)))                        # PCA on in-sample arms
                genT.append(rel_err(Xhd, coherence_genU(data[t], setA, K)))  # coherence gen-eig
            rows.append({"K": K,
                         "gridlap_transfer": round(float(np.mean(glap)), 3),
                         "pca_insample_transfer": round(float(np.mean(pcaT)), 3),
                         "coherence_geneig_transfer": round(float(np.mean(genT)), 3)})
        out[t] = rows
    return out


def main():
    data = {t: {a: load(t, a) for a in ARMS} for t in TASKS}
    res = {"E1_residual_transfer": e1(data), "E2_coverage_transfer_knee": e2(data),
           "E2b_datafit_transfer": e2b(data)}
    json.dump(res, open(os.path.join(HERE, "diag_e1e2_result.json"), "w"), indent=2)

    print("=== E1  residual-transfer decomposition (K=10) ===")
    print("  transfer~=in_sample<<random -> SHARED (Fork A); transfer>>in_sample~=random -> REALIZATION (Fork B)")
    for t, rec in res["E1_residual_transfer"].items():
        print(f"  [{t}] grid-Lap leaves {rec['residual_energy_frac']} of energy as residual")
        for m, v in rec["by_m"].items():
            print(f"      m={m}: in_sample={v['in_sample']}  transfer={v['transfer']}  "
                  f"oracle={v['oracle']}  random={v['random']}")
    print("=== E2  coverage-vs-transfer knee ===")
    for t, rows in res["E2_coverage_transfer_knee"].items():
        print(f"  [{t}]")
        for r in rows:
            print(f"      K={r['K']:>3}  in_sample={r['in_sample']}  transfer={r['transfer']}  "
                  f"gap={r['gap']:+.3f}  pca={r['oracle_pca']}  rand={r['random']}")
    print("=== E2b  data-fit basis transfer at matched K (held-out arm rel err) ===")
    print("  can a basis fit on in-sample arms transfer AND beat grid-Lap's efficiency?")
    for t, rows in res["E2b_datafit_transfer"].items():
        print(f"  [{t}]")
        for r in rows:
            print(f"      K={r['K']:>3}  gridlap={r['gridlap_transfer']}  "
                  f"pca_insample={r['pca_insample_transfer']}  "
                  f"coherence_geneig={r['coherence_geneig_transfer']}")
    print("DIAG_E1E2_DONE=ok")


if __name__ == "__main__":
    main()
