"""Offline subspace-transfer study on Open X-Embodiment action chunks. For each
held-out robot, a k-dimensional subspace fit on the other robots is used to
reconstruct the held-out robot's action chunks, and the relative reconstruction
error is compared across four subspaces: the cross-channel grid-Laplacian (fixed,
defined by the graph), a per-channel Laplacian (the grid Laplacian with zero
channel coupling), the principal components of the other robots' chunks (data-driven
transfer), and a random orthonormal subspace. A within-robot principal-component
subspace fit on the held-out robot itself is included as a lower bound. Lower error
means the subspace captures more of the held-out robot's action-chunk structure.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import laplacian_basis as LB            # noqa: E402

H, C = 16, 6
D = H * C
KS = [4, 8, 12]
ROBOTS = os.environ.get("SNMVP_OXE_ROBOTS", "berkeley_autolab_ur5,bridge,toto,viola").split(",")
DATA = os.path.join(HERE, "data_oxe")


def load(ds):
    return np.load(os.path.join(DATA, f"{ds}.npz"))["chunks"].reshape(-1, D)


def rel_err(X, U):
    proj = X @ (U @ U.T)
    return float(np.sqrt(((X - proj) ** 2).sum() / ((X ** 2).sum() + 1e-12)))


def pca(X, k):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def main():
    data = {r: load(r) for r in ROBOTS}
    for r in ROBOTS:
        print(f"{r}: {data[r].shape[0]} chunks", flush=True)
    glap_all = LB.grid_laplacian_dirs(H, C, 0.5).T          # (D,D), smoothest-first columns
    perch_all = LB.grid_laplacian_dirs(H, C, 0.0).T
    rng = np.random.default_rng(0)
    rand_all = np.linalg.qr(rng.normal(size=(D, D)))[0]

    out = {}
    for held in ROBOTS:
        setA = [r for r in ROBOTS if r != held]
        XA = np.concatenate([data[r] for r in setA], axis=0)
        XB = data[held]
        row = {}
        for k in KS:
            row[f"k{k}"] = {
                "GLAP": round(rel_err(XB, glap_all[:, :k]), 3),
                "perchannel": round(rel_err(XB, perch_all[:, :k]), 3),
                "PCA_setA": round(rel_err(XB, pca(XA, k)), 3),
                "PCA_heldout": round(rel_err(XB, pca(XB, k)), 3),
                "random": round(rel_err(XB, rand_all[:, :k]), 3)}
        out[held] = row
        print(f"held={held}: {json.dumps(row)}", flush=True)
    json.dump({"robots": ROBOTS, "KS": KS, "result": out},
              open(os.path.join(HERE, "oxe_transfer_result.json"), "w"), indent=2)
    print("OXE_TRANSFER_DONE=ok")


if __name__ == "__main__":
    main()
