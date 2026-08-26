"""Fast end-to-end smoke of Steps 2-4 for ONE (body B, n_scenes, seed) config.

Validates: freeze S_A -> shared prior on set A -> adapt body-B executors
(T/S/Cond/Trand) on few demos -> eval on held-out B scenes (T/S/Cond/Trand +
Toracle). Reduced iters / n_roll for speed. Prints the transfer table.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embodiments as emb
import mb_dataset as ds
import flow_embod as fe
from pin import extract_mags, extract_phases

H = ds.H


def flat_body(chunks, obs, angles, body, n_demos):
    ch = chunks[body]                        # (M,N,H,2)
    M, N = ch.shape[0], ch.shape[1]
    return (np.repeat(obs, N, axis=0), ch.reshape(-1, H, 2), np.repeat(angles, N))


def success_rate(scenes, chs):
    return float(np.mean([[ds.success(sc, chs[r, i]) for i, sc in enumerate(scenes)]
                          for r in range(chs.shape[0])]))


def eval_arm(params, scenes, obs, angles, pins, targets, mags, cond_feats,
             n_roll, seed):
    rng = np.random.default_rng(seed)
    M = obs.shape[0]
    obs_r, ang_r = np.tile(obs, (n_roll, 1)), np.tile(angles, n_roll)
    tgt_r = np.tile(targets, (n_roll, 1)) if targets is not None else None
    mag_r = np.tile(mags, (n_roll, 1)) if mags is not None else None
    fe_r = np.tile(cond_feats, (n_roll, 1)) if cond_feats is not None else None
    ch = fe.rollout(params, obs_r, ang_r, pins, tgt_r, fe_r, rng, mag_targets=mag_r)
    return success_rate(scenes, ch.reshape(n_roll, M, H, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="point")
    ap.add_argument("--n", type=int, default=10)       # adaptation scenes
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--n_roll", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    bodies = emb.make_bodies()

    # --- set-A training data (frame + prior) ---
    A_scenes, A_obs, A_angles, A_chunks = ds.make_dataset(bodies, 200, 8,
                                                          np.random.default_rng(7))
    S_A, diag = fe.freeze_frame(A_chunks, A_angles)
    print("S_A frozen (%d pins):" % len(S_A))
    for p in S_A:
        print("   ", p)
    A_pool_canon = np.concatenate(
        [fe.tfd.to_canonical(A_chunks[b], A_angles[:, None]) for b in fe.SET_A],
        axis=1).reshape(-1, H, 2)
    mag_norm = extract_mags(A_pool_canon, S_A).mean(axis=0)

    prior = fe.build_shared_prior(A_chunks, A_obs, A_angles, fe.SET_A, S_A,
                                  seed=100 + args.seed)
    rand = fe.rand_frame(args.seed)
    prior_r = fe.build_shared_prior(A_chunks, A_obs, A_angles, fe.SET_A, rand,
                                    seed=200 + args.seed)

    # --- adaptation data for body B (few scenes) ---
    ad_scenes, ad_obs, ad_angles, ad_chunks = ds.make_dataset(
        bodies, args.n, 8, np.random.default_rng(1234 + args.seed))
    obsB, chB, angB = flat_body(ad_chunks, ad_obs, ad_angles, args.body, 8)
    canB = fe.tfd.to_canonical(chB, angB)

    print(f"adapting body={args.body} on n={args.n} scenes "
          f"({len(chB)} demos)...", flush=True)
    # T: pin S_A ; S: plain ; Cond: invariant conditioned ; Trand: random frame
    pT = fe.train_executor(obsB, chB, angB, S_A, None, mag_norm, args.seed, args.iters)
    pS = fe.train_executor(obsB, chB, angB, [], None, mag_norm, args.seed, args.iters)
    pC = fe.train_executor(obsB, chB, angB, [], S_A, mag_norm, args.seed, args.iters)
    pR = fe.train_executor(obsB, chB, angB, rand, None, mag_norm, args.seed, args.iters)

    # --- held-out B eval ---
    he_scenes, he_obs, he_angles, he_chunks = ds.make_dataset(
        bodies, 100, 8, np.random.default_rng(7777))
    heB0 = fe.tfd.to_canonical(he_chunks[args.body][:, 0], he_angles)  # (100,H,2) GT
    pt, pm, _ = fe.prior_predict(prior, he_obs, S_A)
    ptr, pmr, _ = fe.prior_predict(prior_r, he_obs, rand)
    ot = extract_phases(heB0, S_A)
    om = extract_mags(heB0, S_A)
    om = np.where([p.get("mag", False) for p in S_A], om, np.nan)
    condfe = fe.features(heB0, S_A, mag_norm)   # cond arm eval uses GT invariant feats? use prior:
    # Cond eval should use the SAME command source as T (prior) for fairness:
    cond_pt, cond_pm, _ = fe.prior_predict(prior, he_obs, S_A)
    condfe_prior = _feats_from_targets(cond_pt, cond_pm, S_A, mag_norm)

    res = {
        "T_prior": eval_arm(pT, he_scenes, he_obs, he_angles, S_A, pt, pm, None, args.n_roll, 50),
        "T_oracle": eval_arm(pT, he_scenes, he_obs, he_angles, S_A, ot, om, None, args.n_roll, 51),
        "S_plain": eval_arm(pS, he_scenes, he_obs, he_angles, [], None, None, None, args.n_roll, 52),
        "Cond_prior": eval_arm(pC, he_scenes, he_obs, he_angles, [], None, None, condfe_prior, args.n_roll, 53),
        "Trand_prior": eval_arm(pR, he_scenes, he_obs, he_angles, rand, ptr, pmr, None, args.n_roll, 54),
    }
    res = {k: round(v, 4) for k, v in res.items()}
    print("SMOKE RESULT (held-out %s success):" % args.body, json.dumps(res))
    print(f"T>S: {res['T_prior'] - res['S_plain']:+.3f}   "
          f"T>Cond: {res['T_prior'] - res['Cond_prior']:+.3f}   "
          f"T>Trand: {res['T_prior'] - res['Trand_prior']:+.3f}")
    print(f"done in {time.time()-t0:.0f}s")
    print("SMOKE_DONE=ok")


def _feats_from_targets(targets, mags, pins, mag_norm):
    """Build cond features from (phase,mag) targets (NaN->0), matching
    fe.features layout: (cos,sin) per pin then mag/mag_norm per mag pin."""
    n = targets.shape[0]
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
