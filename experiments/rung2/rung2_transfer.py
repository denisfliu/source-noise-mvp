"""Rung 2 transfer: cross-arm coherence + freeze-and-adapt on REAL robosuite
arm data (offline, reuses the toy flow/coherence/pin machinery).

Data: per-arm .npz of planar EE-reach chunks (N,H,2) to a SHARED target list, so
demo i is the same scene across arms. Set A = Panda/Sawyer/IIWA (frozen frame +
prior); held-out = UR5e (adapt only the executor on few demos). Success (offline)
= generated chunk's net displacement within TOL of the commanded target.
"""

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import flow_embod as fe
import coherence_xembod as cx
fe.OBS_DIM = 2                      # scene = 2D target displacement
fe.H = 32                           # robosuite reach horizon (toy default was 20)
fe.SET_A = ["Panda", "Sawyer", "IIWA"]
from pin import extract_phases, extract_mags   # noqa: E402

DATA = os.path.join(HERE, "data")
ARMS = ["Panda", "Sawyer", "IIWA", "UR5e"]
SET_A = ["Panda", "Sawyer", "IIWA"]
HELD = "UR5e"
H = 32
TOL = 0.03
NS = [5, 10, 20]
SEEDS = [0, 1]


def load():
    d = {a: np.load(os.path.join(DATA, f"{a}.npz")) for a in ARMS}
    disp = d[ARMS[0]]["disp"]                      # shared across arms (N,2)
    angles = np.arctan2(disp[:, 1], disp[:, 0])
    chunks = {a: d[a]["chunks"].astype(float) for a in ARMS}
    return chunks, disp, angles, {a: d[a]["success"] for a in ARMS}


def endpoint_success(gen_chunks, disp_eval, tol):
    """gen (R,M,H,2) -> fraction whose net displacement is within tol of disp."""
    end = gen_chunks.sum(axis=2)                   # (R,M,2)
    err = np.linalg.norm(end - disp_eval[None], axis=-1)
    return float((err < tol).mean())


def eval_arm(params, obs, angles, disp, pins, targets, mags, feats, n_roll, seed, tol):
    rng = np.random.default_rng(seed)
    M = obs.shape[0]
    obs_r, ang_r = np.tile(obs, (n_roll, 1)), np.tile(angles, n_roll)
    tg = np.tile(targets, (n_roll, 1)) if targets is not None else None
    mg = np.tile(mags, (n_roll, 1)) if mags is not None else None
    fr = np.tile(feats, (n_roll, 1)) if feats is not None else None
    ch = fe.rollout(params, obs_r, ang_r, pins, tg, fr, rng, mag_targets=mg)
    return endpoint_success(ch.reshape(n_roll, M, H, 2), disp, tol)


def main():
    chunks, disp, angles, demo_succ = load()
    N = disp.shape[0]
    print("demo success (robosuite):",
          {a: round(float(demo_succ[a].mean()), 3) for a in ARMS})

    # normalize actions to O(1) (pitfall #1: raw ~0.006 m/step deltas underfit).
    SCALE = 1.0 / np.abs(np.concatenate([chunks[a] for a in SET_A])).mean()
    for a in ARMS:
        chunks[a] = chunks[a] * SCALE
    disp = disp * SCALE
    tol = TOL * SCALE
    print(f"ACT_SCALE={SCALE:.1f}  (scaled tol={tol:.3f}, scaled mean|disp|="
          f"{np.linalg.norm(disp, axis=1).mean():.2f})")

    # --- coherence frame over set A ---
    S_A, _ = fe.freeze_frame(chunks, angles, set_a=SET_A)
    print("S_A (%d pins):" % len(S_A),
          [(tuple(np.round(p["axis"], 2)), p["omega"], p["mode"], p.get("mag")) for p in S_A])
    align = {a: cx.align_to_consensus(chunks, angles, SET_A, a, S_A) for a in ARMS}
    print("alignment to set-A consensus:", json.dumps(align))

    A_pool = np.concatenate([fe.tfd.to_canonical(chunks[b], angles) for b in SET_A], axis=0)
    mag_norm = extract_mags(A_pool, S_A).mean(axis=0)

    # --- split held-out arm scenes: adaptation vs eval ---
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    results = {}
    for seed in SEEDS:
        prior = fe.build_shared_prior(chunks, disp, angles, SET_A, S_A, 100 + seed)
        rand = fe.rand_frame(seed)
        prior_r = fe.build_shared_prior(chunks, disp, angles, SET_A, rand, 200 + seed)
        for n in NS:
            ad = perm[:n]
            ev = perm[n:]
            obsB, chB, angB, dispB = disp[ad], chunks[HELD][ad], angles[ad], disp[ad]
            oE, aE, dE = disp[ev], angles[ev], disp[ev]
            pT = fe.train_executor(obsB, chB, angB, S_A, None, mag_norm, seed, 6000)
            pS = fe.train_executor(obsB, chB, angB, [], None, mag_norm, seed, 6000)
            pC = fe.train_executor(obsB, chB, angB, [], S_A, mag_norm, seed, 6000)
            pR = fe.train_executor(obsB, chB, angB, rand, None, mag_norm, seed, 6000)
            pt, pm, _ = fe.prior_predict(prior, oE, S_A)
            ptr, pmr, _ = fe.prior_predict(prior_r, oE, rand)
            condfe = _feats(pt, pm, S_A, mag_norm)
            row = {
                "T": eval_arm(pT, oE, aE, dE, S_A, pt, pm, None, 4, 50, tol),
                "S": eval_arm(pS, oE, aE, dE, [], None, None, None, 4, 51, tol),
                "Cond": eval_arm(pC, oE, aE, dE, [], None, None, condfe, 4, 52, tol),
                "Trand": eval_arm(pR, oE, aE, dE, rand, ptr, pmr, None, 4, 53, tol),
            }
            results[f"s{seed}_n{n}"] = {k: round(v, 3) for k, v in row.items()}
            print(f"seed{seed} n{n}: {results[f's{seed}_n{n}']}", flush=True)

    # pooled
    pooled = {}
    for k in ["T", "S", "Cond", "Trand"]:
        for n in NS:
            vs = [results[f"s{s}_n{n}"][k] for s in SEEDS]
            pooled.setdefault(f"n{n}", {})[k] = round(float(np.mean(vs)), 3)
    out = {"demo_success": {a: round(float(demo_succ[a].mean()), 3) for a in ARMS},
           "alignment": align, "S_A_pins": len(S_A), "per_config": results,
           "pooled": pooled}
    json.dump(out, open(os.path.join(HERE, "transfer_result.json"), "w"), indent=2)
    print("POOLED:", json.dumps(pooled))
    print("RUNG2_TRANSFER_DONE=ok")


def _feats(targets, mags, pins, mag_norm):
    feats = []
    for k, p in enumerate(pins):
        mult = 2.0 if p["mode"] == "modpi" else 1.0
        ang = np.where(np.isnan(targets[:, k]), 0.0, targets[:, k])
        feats += [np.cos(mult * ang), np.sin(mult * ang)]
    for k, p in enumerate(pins):
        if p.get("mag"):
            m = np.where(np.isnan(mags[:, k]), mag_norm[k], mags[:, k])
            feats.append(m / mag_norm[k])
    return np.stack(feats, axis=-1)


if __name__ == "__main__":
    main()
