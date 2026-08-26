"""Toy test of a transfer-function (Laplace / z-transform) factoring against the
additive subspace, across a dynamics gap.

A shared planned trajectory (the task) is passed through a second-order linear
time-invariant filter (a stand-in for a body-and-controller), producing an achieved
trajectory. Different filters, meaning different poles (natural frequency w and
damping zeta), stand in for different bodies. The question is how well each of two
representations reconstructs a held-out body's achieved trajectories as the pole gap
from the training bodies grows.

  additive : the best k-dimensional subspace fit on the training bodies' ACHIEVED
             trajectories, applied to the held-out body (frozen subspace, oracle
             coordinate = orthogonal projection).
  laplace  : identify the held-out body's filter from a few of its demonstrations,
             reconstruct the shared PLANNED path from a subspace fit on the training
             bodies' planned paths, and re-apply the identified filter.

Reported per held-out damping value: relative reconstruction error of the held-out
body's achieved trajectories for each method, and the identified filter parameters.
Pure numpy.
"""
import json
import numpy as np

H = 32
K = 6
N_SCENES = 240
N_HELD = 60
N_ID = 5                                   # held-out demos used to identify the filter
DT = 0.08
SET_A = [(5.0, 1.0), (5.0, 1.2), (5.0, 1.4)]     # (w, zeta) training bodies, overdamped range
HELD_ZETA = [1.0, 0.8, 0.6, 0.4]                 # held-out damping sweep, into underdamped (growing gap)
HELD_W = 5.0
SEEDS = [0, 1, 2]


def planned(amp, s0):
    t = np.linspace(0, 1, H)
    return amp * np.exp(-((t - s0) ** 2) / (2 * 0.08 ** 2))


def sim_filter(ref, w, zeta):
    """Second-order LTI tracker: y'' + 2*zeta*w*y' + w^2*y = w^2*ref, explicit Euler."""
    y = 0.0; v = 0.0; out = np.empty(H)
    for n in range(H):
        acc = w * w * (ref[n] - y) - 2 * zeta * w * v
        v = v + DT * acc
        y = y + DT * v
        out[n] = y
    return out


def scenes(rng, n):
    amp = rng.uniform(-1.0, 1.0, n)
    s0 = rng.uniform(0.4, 0.6, n)
    return amp, s0


def top_k_subspace(M, k):
    """Columns of M are vectors; return the k left singular vectors capturing them."""
    U, _, _ = np.linalg.svd(M - M.mean(axis=1, keepdims=True), full_matrices=False)
    return U[:, :k]


def rel_err(true, recon):
    return float(np.sqrt(((true - recon) ** 2).sum() / ((true - true.mean()) ** 2).sum()))


def identify(planned_list, achieved_list):
    """Grid-search the filter (w, zeta) that best maps planned to achieved."""
    ws = np.linspace(2.5, 9.0, 34); zs = np.linspace(0.3, 1.6, 40)
    best = (None, np.inf)
    for w in ws:
        for z in zs:
            e = sum(((sim_filter(p, w, z) - a) ** 2).sum() for p, a in zip(planned_list, achieved_list))
            if e < best[1]:
                best = ((w, z), e)
    return best[0]


def main():
    out = {"config": {"SET_A": SET_A, "HELD_W": HELD_W, "K": K, "N_ID": N_ID}, "sweep": {}}
    for hz in HELD_ZETA:
        errs_add, errs_lap, ids = [], [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            amp, s0 = scenes(rng, N_SCENES)
            he = np.arange(N_HELD); tr = np.arange(N_HELD, N_SCENES)
            P = np.stack([planned(amp[i], s0[i]) for i in range(N_SCENES)], axis=1)   # (H, S)
            ach_A = {b: np.stack([sim_filter(P[:, i], *b) for i in range(N_SCENES)], axis=1)
                     for b in SET_A}
            ach_B = np.stack([sim_filter(P[:, i], HELD_W, hz) for i in range(N_SCENES)], axis=1)

            # additive: subspace on training-body achieved (training scenes), project held-out body
            M_A = np.concatenate([ach_A[b][:, tr] for b in SET_A], axis=1)
            U_A = top_k_subspace(M_A, K)
            B_he = ach_B[:, he]
            recon_add = U_A @ (U_A.T @ B_he)
            errs_add.append(rel_err(B_he, recon_add))

            # laplace: planned subspace (body-invariant) + identified held-out filter
            U_P = top_k_subspace(P[:, tr], K)
            wz = identify([P[:, i] for i in tr[:N_ID]], [ach_B[:, i] for i in tr[:N_ID]])
            P_he_recon = U_P @ (U_P.T @ P[:, he])
            recon_lap = np.stack([sim_filter(P_he_recon[:, j], *wz) for j in range(N_HELD)], axis=1)
            errs_lap.append(rel_err(B_he, recon_lap))
            ids.append(wz)
        out["sweep"][f"zeta{hz}"] = {
            "additive_err": round(float(np.mean(errs_add)), 3),
            "laplace_err": round(float(np.mean(errs_lap)), 3),
            "identified_wz": [round(float(np.mean([i[0] for i in ids])), 2),
                              round(float(np.mean([i[1] for i in ids])), 2)],
            "true_wz": [HELD_W, hz]}
        r = out["sweep"][f"zeta{hz}"]
        print(f"held zeta={hz}: additive_err={r['additive_err']} laplace_err={r['laplace_err']} "
              f"id_wz={r['identified_wz']} true={r['true_wz']}", flush=True)
    json.dump(out, open("laplace_toy_result.json", "w"), indent=2)
    print("LAPLACE_TOY_DONE=ok")


if __name__ == "__main__":
    main()
