"""Controlled decomposition of transfer into an embodiment axis and a task axis. For
each task, a grid-Laplacian coherence subspace is fit on two of the three arms and
used to reconstruct the held-out arm's chunks of the same task, which measures
transfer across embodiment with the task held. For each ordered pair of tasks, a
subspace fit on all arms of the source task is used to reconstruct the target task's
chunks, which measures transfer across task. Lower relative reconstruction error
means the subspace captures more of the target's structure. A principal-component
subspace fit on the target is the lower bound and a random subspace the upper bound.
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
K = 10
TASKS = ["bank", "vertical", "slalom"]
ARMS = ["Panda", "IIWA", "UR5e"]
DATA = os.path.join(HERE, "data_taskembod")


def load(task, arm):
    ch = np.load(os.path.join(DATA, f"{task}_{arm}.npz"))["chunks"].astype(float)
    S, N = ch.shape[:2]
    X = ch.reshape(S, N, D)
    flat = X.reshape(-1, D)
    X = X - flat.mean(axis=0)
    return X / (np.sqrt((X ** 2).mean()) + 1e-9)


def coherence_U(Xby, arms, k):
    bmean = np.stack([Xby[a].mean(axis=1) for a in arms], axis=1)   # (S,|A|,D)
    Sb, Sw = BL.covariances(bmean, D)
    return LB.basis_gridlap(Sb, Sw, k, H, C, 0.5)


def rel_err(X, U):
    Xf = X.reshape(-1, D)
    proj = Xf @ (U @ U.T)
    return float(np.sqrt(((Xf - proj) ** 2).sum() / ((Xf ** 2).sum() + 1e-12)))


def pca(X, k):
    Xf = X.reshape(-1, D)
    _, _, Vt = np.linalg.svd(Xf - Xf.mean(axis=0), full_matrices=False)
    return Vt[:k].T


def main():
    data = {t: {a: load(t, a) for a in ARMS} for t in TASKS}
    rU = np.linalg.qr(np.random.default_rng(0).normal(size=(D, D)))[0][:, :K]

    cross_emb = {}
    for t in TASKS:
        errs = []
        for held in ARMS:
            setA = [a for a in ARMS if a != held]
            errs.append(rel_err(data[t][held], coherence_U(data[t], setA, K)))
        cross_emb[t] = round(float(np.mean(errs)), 3)

    cross_task = {}
    for tsrc in TASKS:
        U = coherence_U(data[tsrc], ARMS, K)
        for ttgt in TASKS:
            Xt = np.concatenate([data[ttgt][a] for a in ARMS], axis=0)
            cross_task[f"{tsrc}->{ttgt}"] = round(rel_err(Xt, U), 3)

    oracle, random_b = {}, {}
    for t in TASKS:
        Xt = np.concatenate([data[t][a] for a in ARMS], axis=0)
        oracle[t] = round(rel_err(Xt, pca(Xt, K)), 3)
        random_b[t] = round(rel_err(Xt, rU), 3)

    out = {"K": K, "cross_embodiment_within_task": cross_emb, "cross_task": cross_task,
           "oracle_pca": oracle, "random": random_b}
    json.dump(out, open(os.path.join(HERE, "taskembod_result.json"), "w"), indent=2)
    print("CROSS-EMBODIMENT within task (held-out arm, same task):", json.dumps(cross_emb))
    print("CROSS-TASK (source->target):", json.dumps(cross_task))
    print("ORACLE (PCA on target):", json.dumps(oracle), " RANDOM:", json.dumps(random_b))
    print("TASKEMBOD_DONE=ok")


if __name__ == "__main__":
    main()
