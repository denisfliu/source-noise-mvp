"""Complexity-crossover battery — tests Denis's hypothesis that exact source-noise
pinning / the coherence frame win only because the single-obstacle toy is simple
(~10 bits, linearly FFT-pinnable).

Sweep task complexity = number of obstacles n_obst in {1,2,3} (richer, more
nonlinear shared detour structure), hold adaptation data fixed, and compare the
OAT-prefix invariant (conditioning + subspace-pin) against the hand-built
coherence frame on the SAME data. Predicted signature: coherence leads at
n_obst=1 (its best case) and OAT catches up / overtakes as complexity grows,
because coherence's fixed few-pin exact-phase binding can't represent a high-bit
nonlinear shared structure while OAT's learned ordered bottleneck scales.

Reuses oat_transfer's executor/frame/prior (dataset-agnostic) with the wider
multi-obstacle obs, and flow_embod's coherence arms (OBS_DIM monkeypatched).
Eval uses mb_dataset_hard.success (multi-obstacle).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
import mb_dataset_hard as ds       # noqa: E402
import embodiments as emb          # noqa: E402
import dataset as tfd              # noqa: E402
import flow_embod as fe            # noqa: E402
import transfer_smoke as sm        # noqa: E402  (flat_body only; dataset-agnostic)
import oat_transfer as ot          # noqa: E402
from pin import extract_mags       # noqa: E402

# widen obs everywhere to the multi-obstacle vector
fe.OBS_DIM = ds.OBS_DIM
ot.OBS = ds.OBS_DIM

H = ds.H
SET_A = fe.SET_A
NOBS = [1, 2, 3]
BODIES_B = ["point", "arm4"]
SEEDS = [0, 1, 2]
N = 25                              # fixed adaptation budget (OAT's competitive regime)
ITERS = 10000
N_ROLL = 8
N_EVAL = 100
OUT = os.path.join(HERE, "results", "oat_complexity")


def success_rate(scenes, chs):     # chs (n_roll, M, H, 2)
    return float(np.mean([[ds.success(scenes[i], chs[r, i]) for i in range(len(scenes))]
                          for r in range(chs.shape[0])]))


def eval_oat(params, scenes, obs, inv, cond, seed):
    rng = np.random.default_rng(seed)
    M = obs.shape[0]
    obs_r = np.tile(obs, (N_ROLL, 1))
    inv_r = np.tile(inv, (N_ROLL, 1)) if inv is not None else None
    ch = ot.rollout(params, obs_r, inv_r, cond, rng).reshape(N_ROLL, M, H, 2)
    return success_rate(scenes, ch)


def eval_coh(params, scenes, obs, angles, pins, targets, mags, feats, seed):
    rng = np.random.default_rng(seed)
    M = obs.shape[0]
    obs_r, ang_r = np.tile(obs, (N_ROLL, 1)), np.tile(angles, N_ROLL)
    tgt_r = np.tile(targets, (N_ROLL, 1)) if targets is not None else None
    mag_r = np.tile(mags, (N_ROLL, 1)) if mags is not None else None
    fe_r = np.tile(feats, (N_ROLL, 1)) if feats is not None else None
    ch = fe.rollout(params, obs_r, ang_r, pins, tgt_r, fe_r, rng,
                    mag_targets=mag_r).reshape(N_ROLL, M, H, 2)
    return success_rate(scenes, ch)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "rows.jsonl")
    open(rows_path, "w").close()
    t0 = time.time()
    bodies = emb.make_bodies()
    rows = []

    for nob in NOBS:
        A_s, A_obs, A_ang, A_ch = ds.make_dataset(bodies, 200, 8,
                                                  np.random.default_rng(7), n_obst=nob)
        frame = ot.fit_frame(A_ch, A_ang, seed=0)
        S_A, _ = fe.freeze_frame(A_ch, A_ang)
        A_pool = np.concatenate([tfd.to_canonical(A_ch[b], A_ang[:, None])
                                 for b in SET_A], axis=1).reshape(-1, H, 2)
        mag_norm = extract_mags(A_pool, S_A).mean(axis=0)
        he_s, he_o, he_a, he_ch = ds.make_dataset(bodies, N_EVAL, 8,
                                                  np.random.default_rng(7777), n_obst=nob)
        y_shared = ot.shared_prefix_target(frame, A_ch, A_ang)
        print(f"[{time.time()-t0:.0f}s] n_obst={nob}: coherence S_A={len(S_A)} pins, "
              f"OAT prefix dim={frame.m}", flush=True)

        oat_prior, coh_prior = {}, {}
        for s in SEEDS:
            oat_prior[s] = ot.train_prefix_prior(A_obs, y_shared, seed=300 + s)
            coh_prior[s] = fe.build_shared_prior(A_ch, A_obs, A_ang, SET_A, S_A, 100 + s)

        for B in BODIES_B:
            heB0 = tfd.to_canonical(he_ch[B][:, 0], he_a)
            inv_oracle = frame.encode(heB0)
            for s in SEEDS:
                oat_cmd = ot.predict_prefix(oat_prior[s], he_o)
                ct, cm, _ = fe.prior_predict(coh_prior[s], he_o, S_A)
                coh_feats = sm._feats_from_targets(ct, cm, S_A, mag_norm)
                ad_s, ad_o, ad_a, ad_c = ds.make_dataset(bodies, N, 8,
                                                         np.random.default_rng(1234 + s), n_obst=nob)
                obsB, chB, angB = sm.flat_body(ad_c, ad_o, ad_a, B, 8)
                canB = tfd.to_canonical(chB, angB)
                invB = frame.encode(canB)

                pCon = ot.train_exec(obsB, chB, canB, invB, True, s, ITERS)
                pPin = ot.train_exec(obsB, chB, canB, invB, False, s, ITERS)
                pS = ot.train_exec(obsB, chB, canB, None, False, s, ITERS)
                pcohT = fe.train_executor(obsB, chB, angB, S_A, None, mag_norm, s, ITERS)
                pcohC = fe.train_executor(obsB, chB, angB, [], S_A, mag_norm, s, ITERS)

                row = {"n_obst": nob, "B": B, "seed": s, "n": N, "n_pins": len(S_A),
                       "OATcond": eval_oat(pCon, he_s, he_o, oat_cmd, True, 51),
                       "OATpin": eval_oat(pPin, he_s, he_o, oat_cmd, False, 50),
                       "OATorac": eval_oat(pCon, he_s, he_o, inv_oracle, True, 52),
                       "S": eval_oat(pS, he_s, he_o, None, False, 54),
                       "cohT": eval_coh(pcohT, he_s, he_o, he_a, S_A, ct, cm, None, 55),
                       "cohCond": eval_coh(pcohC, he_s, he_o, he_a, [], None, None, coh_feats, 56)}
                row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                rows.append(row)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{time.time()-t0:.0f}s] nob{nob} {B} s{s}: "
                      f"OATcond={row['OATcond']} OATpin={row['OATpin']} S={row['S']} "
                      f"cohT={row['cohT']} cohCond={row['cohCond']} "
                      f"(orac={row['OATorac']})", flush=True)

    verdict = summarize(rows)
    json.dump({"rows": rows, "verdict": verdict},
              open(os.path.join(OUT, "battery.json"), "w"), indent=2)
    print("VERDICT:", json.dumps(verdict, indent=2))
    print(f"OAT_COMPLEXITY_DONE=ok in {time.time()-t0:.0f}s")


def summarize(rows):
    out = {}
    for nob in NOBS:
        r = [x for x in rows if x["n_obst"] == nob]
        m = {k: round(float(np.mean([x[k] for x in r])), 3)
             for k in ["OATcond", "OATpin", "OATorac", "S", "cohT", "cohCond"]}
        best_oat = max(m["OATcond"], m["OATpin"])
        best_coh = max(m["cohT"], m["cohCond"])
        m["n_pins"] = int(np.mean([x["n_pins"] for x in r]))
        m["bestOAT_minus_bestCoh"] = round(best_oat - best_coh, 3)
        m["bestOAT_minus_S"] = round(best_oat - m["S"], 3)
        out[f"n_obst={nob}"] = m
    deltas = [out[f"n_obst={n}"]["bestOAT_minus_bestCoh"] for n in NOBS]
    out["_crossover"] = ("bestOAT_minus_bestCoh across n_obst=%s: %s. Denis's "
                         "hypothesis predicts this rises with complexity (coherence "
                         "leads at n_obst=1, OAT catches/overtakes by n_obst=3)."
                         % (NOBS, deltas))
    out["_crossover_trend_up"] = bool(deltas[-1] > deltas[0])
    return out


if __name__ == "__main__":
    main()
