"""One-shot gate-passage transfer — can a shared invariant trained on OTHER
embodiments let a held-out drone pass gates from n=1 demo?

North-star proxy: one-shot an IRL drone through a gate. Held-out body =
point_drag (realistic drone w/ inertia). Set A = arms {arm2,arm3,arm4}. The gate
CENTERS are observable (in obs), so scratch already sees where the gates are;
the shared invariant's job is to supply the cross-body EXECUTION knowledge (how
to weave through given dynamics) that one demo can't. Simple methods first
(scratch, coherence-pin, coherence-cond, OAT-cond) before VLA^2.

True one-shot: adaptation data is n distinct (scene,demo) pairs, n in {1,3,10},
one demo per scene. Shared priors are trained on the full set-A data; only the
per-body executor is adapted on the n demos.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
import gate_dataset as ds          # noqa: E402
import embodiments as emb          # noqa: E402
import dataset as tfd              # noqa: E402
import flow_embod as fe            # noqa: E402
import transfer_smoke as sm        # noqa: E402
import oat_transfer as ot          # noqa: E402
from pin import extract_mags       # noqa: E402

fe.OBS_DIM = ds.OBS_DIM
ot.OBS = ds.OBS_DIM

H = ds.H
SET_A = fe.SET_A
NG = [1, 2]                         # gates (racing difficulty)
BODIES_B = ["point_drag", "point"]
SEEDS = [0, 1, 2]
NS = [1, 3, 10]                    # adaptation demos; n=1 == one-shot
ITERS = 8000
N_ROLL = 8
N_EVAL = 100
OUT = os.path.join(HERE, "results", "gate_oneshot")


def success_rate(scenes, chs):
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

    for ng in NG:
        A_s, A_obs, A_ang, A_ch = ds.make_dataset(bodies, 200, 8,
                                                  np.random.default_rng(7), n_gates=ng)
        frame = ot.fit_frame(A_ch, A_ang, seed=0)
        S_A, _ = fe.freeze_frame(A_ch, A_ang)
        A_pool = np.concatenate([tfd.to_canonical(A_ch[b], A_ang[:, None])
                                 for b in SET_A], axis=1).reshape(-1, H, 2)
        mag_norm = extract_mags(A_pool, S_A).mean(axis=0)
        he_s, he_o, he_a, he_ch = ds.make_dataset(bodies, N_EVAL, 8,
                                                  np.random.default_rng(7777), n_gates=ng)
        y_shared = ot.shared_prefix_target(frame, A_ch, A_ang)
        # per-body demo ceilings (headroom check)
        ceil = {B: round(float(np.mean([ds.success(he_s[i], he_ch[B][i, d])
                for i in range(len(he_s)) for d in range(he_ch[B].shape[1])])), 3)
                for B in BODIES_B}
        print(f"[{time.time()-t0:.0f}s] n_gates={ng}: coh pins={len(S_A)}, OAT dim="
              f"{frame.m}, ceilings={ceil}", flush=True)

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
                for n in NS:
                    # true one-shot: n distinct scenes, one demo each
                    ad_s, ad_o, ad_a, ad_c = ds.make_dataset(
                        bodies, n, 1, np.random.default_rng(1234 + s), n_gates=ng)
                    obsB, chB, angB = sm.flat_body(ad_c, ad_o, ad_a, B, 1)
                    canB = tfd.to_canonical(chB, angB)
                    invB = frame.encode(canB)

                    pCon = ot.train_exec(obsB, chB, canB, invB, True, s, ITERS)
                    pS = ot.train_exec(obsB, chB, canB, None, False, s, ITERS)
                    pcohT = fe.train_executor(obsB, chB, angB, S_A, None, mag_norm, s, ITERS)
                    pcohC = fe.train_executor(obsB, chB, angB, [], S_A, mag_norm, s, ITERS)

                    row = {"n_gates": ng, "B": B, "seed": s, "n": n, "ceiling": ceil[B],
                           "OATcond": eval_oat(pCon, he_s, he_o, oat_cmd, True, 51),
                           "OATorac": eval_oat(pCon, he_s, he_o, inv_oracle, True, 52),
                           "S": eval_oat(pS, he_s, he_o, None, False, 54),
                           "cohT": eval_coh(pcohT, he_s, he_o, he_a, S_A, ct, cm, None, 55),
                           "cohCond": eval_coh(pcohC, he_s, he_o, he_a, [], None, None, coh_feats, 56)}
                    row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                    rows.append(row)
                    with open(rows_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[{time.time()-t0:.0f}s] ng{ng} {B} s{s} n{n}: "
                          f"OATcond={row['OATcond']} S={row['S']} cohT={row['cohT']} "
                          f"cohCond={row['cohCond']} (orac={row['OATorac']} ceil={ceil[B]})",
                          flush=True)

    verdict = summarize(rows)
    json.dump({"rows": rows, "verdict": verdict},
              open(os.path.join(OUT, "battery.json"), "w"), indent=2)
    print("VERDICT:", json.dumps(verdict, indent=2))
    print(f"GATE_ONESHOT_DONE=ok in {time.time()-t0:.0f}s")


def summarize(rows):
    out = {}
    for ng in NG:
        for B in BODIES_B:
            for n in NS:
                r = [x for x in rows if x["n_gates"] == ng and x["B"] == B and x["n"] == n]
                if not r:
                    continue
                m = {k: round(float(np.mean([x[k] for x in r])), 3)
                     for k in ["OATcond", "OATorac", "S", "cohT", "cohCond"]}
                best = max(m["OATcond"], m["cohT"], m["cohCond"])
                m["best_shared_minus_S"] = round(best - m["S"], 3)
                m["ceiling"] = r[0]["ceiling"]
                out[f"ng{ng}_{B}_n{n}"] = m
    # headline: one-shot (n=1) lift of best shared method over scratch, per body/gates
    out["_oneshot"] = {f"ng{ng}_{B}": out.get(f"ng{ng}_{B}_n1", {}).get("best_shared_minus_S")
                       for ng in NG for B in BODIES_B}
    return out


if __name__ == "__main__":
    main()
