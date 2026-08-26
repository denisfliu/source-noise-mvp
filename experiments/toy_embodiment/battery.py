"""Full transfer battery (Rung 1, Steps 2-4) + G-transfer / G-predict gates.

Grid: held-out body B in {point (drone analog, low coherence), arm4 (same
family, high coherence)} x seeds {0,1,2} x adaptation scenes n {5,10,25,50}.
Full eval: 100 held-out scenes x 8 rollouts. Writes results/transfer/rows.jsonl
incrementally (robust to disconnects) then battery.json + README + verdict.

G-transfer: at low n (5,10), pooled over seeds, T > S and T > Cond and T > Trand.
G-predict: mean transfer gain (T-S) is ordered by cross-body coherence c(B,setA)
           (arm4 shares more with the arms than point does -> different gain).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embodiments as emb
import mb_dataset as ds
import flow_embod as fe
import transfer_smoke as sm
import coherence_xembod as cx
from pin import extract_mags, extract_phases

H = ds.H
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "transfer_v3")
BODIES_B = ["point_phase0", "point_phase15", "point_phase30", "point_phase45"]  # controlled coherence sweep
SEEDS = [0, 1, 2]
NS = [10, 25]                                             # informative low-n regime
ITERS = 10000
N_ROLL = 8
THETAS = np.linspace(0.0, np.pi, 90)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "rows.jsonl")
    open(rows_path, "w").close()
    t0 = time.time()
    bodies = emb.make_bodies()

    # set-A training data (shared across everything)
    A_scenes, A_obs, A_angles, A_chunks = ds.make_dataset(bodies, 200, 8,
                                                          np.random.default_rng(7))
    S_A, diag = fe.freeze_frame(A_chunks, A_angles)
    A_pool = np.concatenate([fe.tfd.to_canonical(A_chunks[b], A_angles[:, None])
                             for b in fe.SET_A], axis=1).reshape(-1, H, 2)
    mag_norm = extract_mags(A_pool, S_A).mean(axis=0)
    json.dump({"S_A": [{**p, "axis": list(p["axis"])} for p in S_A], "diag": diag},
              open(os.path.join(OUT, "S_A.json"), "w"), indent=2)

    # held-out eval data (shared)
    he_scenes, he_obs, he_angles, he_chunks = ds.make_dataset(bodies, 100, 8,
                                                              np.random.default_rng(7777))
    # per-body divergence for G-predict = B's alignment to the set-A consensus on
    # the S_A bins (sensitive, unlike pooled concentration). Plus demo ceiling.
    coh = {B: cx.align_to_consensus(he_chunks, he_angles, fe.SET_A, B, S_A)
           for B in BODIES_B}
    ceil = {B: round(float(np.mean([ds.success(he_scenes[s], he_chunks[B][s, d])
            for s in range(len(he_scenes)) for d in range(he_chunks[B].shape[1])])), 3)
            for B in BODIES_B}
    print("alignment(B,setA):", json.dumps(coh), " ceiling:", json.dumps(ceil))

    # priors: shared (learned frame) + random frame, one per seed, reused across n/body
    priors, priors_r, rands = {}, {}, {}
    for s in SEEDS:
        priors[s] = fe.build_shared_prior(A_chunks, A_obs, A_angles, fe.SET_A, S_A, 100 + s)
        rands[s] = fe.rand_frame(s)
        priors_r[s] = fe.build_shared_prior(A_chunks, A_obs, A_angles, fe.SET_A,
                                            rands[s], 200 + s)

    rows = []
    for B in BODIES_B:
        heB0 = fe.tfd.to_canonical(he_chunks[B][:, 0], he_angles)
        for s in SEEDS:
            pt, pm, _ = fe.prior_predict(priors[s], he_obs, S_A)
            ptr, pmr, _ = fe.prior_predict(priors_r[s], he_obs, rands[s])
            ot = extract_phases(heB0, S_A)
            om = np.where([p.get("mag", False) for p in S_A],
                          extract_mags(heB0, S_A), np.nan)
            condfe = sm._feats_from_targets(pt, pm, S_A, mag_norm)
            for n in NS:
                ad_s, ad_o, ad_a, ad_c = ds.make_dataset(bodies, n, 8,
                                                         np.random.default_rng(1234 + s))
                obsB, chB, angB = sm.flat_body(ad_c, ad_o, ad_a, B, 8)
                pT = fe.train_executor(obsB, chB, angB, S_A, None, mag_norm, s, ITERS)
                pS = fe.train_executor(obsB, chB, angB, [], None, mag_norm, s, ITERS)
                pC = fe.train_executor(obsB, chB, angB, [], S_A, mag_norm, s, ITERS)
                pR = fe.train_executor(obsB, chB, angB, rands[s], None, mag_norm, s, ITERS)
                row = {"B": B, "seed": s, "n": n,
                       "T": sm.eval_arm(pT, he_scenes, he_obs, he_angles, S_A, pt, pm, None, N_ROLL, 50),
                       "Toracle": sm.eval_arm(pT, he_scenes, he_obs, he_angles, S_A, ot, om, None, N_ROLL, 51),
                       "S": sm.eval_arm(pS, he_scenes, he_obs, he_angles, [], None, None, None, N_ROLL, 52),
                       "Cond": sm.eval_arm(pC, he_scenes, he_obs, he_angles, [], None, None, condfe, N_ROLL, 53),
                       "Trand": sm.eval_arm(pR, he_scenes, he_obs, he_angles, rands[s], ptr, pmr, None, N_ROLL, 54)}
                row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                rows.append(row)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{time.time()-t0:.0f}s] {B} s{s} n{n}: "
                      f"T={row['T']} S={row['S']} Cond={row['Cond']} "
                      f"Trand={row['Trand']} (Torac={row['Toracle']})", flush=True)

    verdict = summarize(rows, coh, len(S_A))
    json.dump({"rows": rows, "alignment_to_setA": coh, "demo_ceiling": ceil,
               "n_pins": len(S_A), "verdict": verdict},
              open(os.path.join(OUT, "battery.json"), "w"), indent=2)
    write_readme(rows, coh, verdict, S_A)
    print("VERDICT:", json.dumps(verdict))
    print(f"BATTERY_DONE=ok in {time.time()-t0:.0f}s")


def summarize(rows, coh, n_pins):
    def pooled(B, n, key):
        vs = [r[key] for r in rows if r["B"] == B and r["n"] == n]
        return float(np.mean(vs)) if vs else float("nan")
    gt = {}
    for B in BODIES_B:
        low = NS
        T = np.mean([pooled(B, n, "T") for n in low])
        S = np.mean([pooled(B, n, "S") for n in low])
        C = np.mean([pooled(B, n, "Cond") for n in low])
        R = np.mean([pooled(B, n, "Trand") for n in low])
        gt[B] = {"T": round(float(T), 3), "S": round(float(S), 3),
                 "Cond": round(float(C), 3), "Trand": round(float(R), 3),
                 "T_gt_S": bool(T > S), "T_gt_Cond": bool(T > C),
                 "T_gt_Trand": bool(T > R), "gain_T_minus_S": round(float(T - S), 3)}
    g_transfer = all(gt[B]["T_gt_S"] and gt[B]["T_gt_Cond"] and gt[B]["T_gt_Trand"]
                     for B in BODIES_B)
    # G-predict: does the T-S gain track coherence across the body ladder?
    order_gain = sorted(BODIES_B, key=lambda b: gt[b]["gain_T_minus_S"])
    order_coh = sorted(BODIES_B, key=lambda b: coh[b])
    conc = tot = 0
    for i in range(len(BODIES_B)):
        for j in range(i + 1, len(BODIES_B)):
            bi, bj = BODIES_B[i], BODIES_B[j]
            dc = coh[bi] - coh[bj]
            dg = gt[bi]["gain_T_minus_S"] - gt[bj]["gain_T_minus_S"]
            if abs(dc) > 1e-9 and abs(dg) > 1e-9:
                tot += 1
                conc += 1 if (dc > 0) == (dg > 0) else 0
    concordance = round(conc / tot, 3) if tot else float("nan")
    return {"G_transfer_pass": bool(g_transfer), "per_body_lown": gt,
            "coherence_c": coh, "gain_order": order_gain,
            "coherence_order": order_coh,
            "gain_vs_coherence_concordance": concordance,
            "G_predict_note": "concordance = frac of body pairs where higher "
            "coherence -> higher transfer gain (1.0 = coherence predicts gain; "
            "0.0 = fully reversed; ~0.5 = unrelated). n=4 bodies."}


def write_readme(rows, coh, verdict, S_A):
    L = ["# toy_embodiment Steps 2-4 — cross-embodiment transfer", "",
         "Rung 1 of docs/cross_embodiment_plan.md. Frozen shared frame S_A "
         f"({len(S_A)} pins) + frozen scene->invariant prior trained on set A "
         "{arm2,arm3,arm4}; only the executor is adapted on held-out body B.",
         "Task-space actions (invariant linear, pin exact).", "",
         f"Cross-body coherence c(B, setA): {json.dumps(coh)}", "",
         "## Success on held-out scenes (100 scenes x 8 rollouts)", "",
         "| B | seed | n | T | Toracle | S | Cond | Trand |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['B']} | {r['seed']} | {r['n']} | **{r['T']}** | "
                 f"{r['Toracle']} | {r['S']} | {r['Cond']} | {r['Trand']} |")
    L += ["", "## G-transfer (pooled over n=10,25)", ""]
    for B, d in verdict["per_body_lown"].items():
        L.append(f"- **{B}**: T={d['T']} vs S={d['S']} / Cond={d['Cond']} / "
                 f"Trand={d['Trand']}  ->  T>S {d['T_gt_S']}, T>Cond "
                 f"{d['T_gt_Cond']}, T>Trand {d['T_gt_Trand']} (gain {d['gain_T_minus_S']})")
    L += ["", f"- **G_transfer_pass = {verdict['G_transfer_pass']}**",
          f"- G-predict (directional, n=2 bodies): gain order {verdict['gain_order']} "
          f"vs coherence order {verdict['coherence_order']}", "",
          "Reading: T = frozen arm-learned frame+prior + executor adapted on B's",
          "few demos; S = B from scratch on the same demos; Cond = same invariant",
          "conditioned not pinned; Trand = random-frame pin. T>S = transfer helps;",
          "T>Cond = the pin channel; T>Trand = the LEARNED frame specifically."]
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
