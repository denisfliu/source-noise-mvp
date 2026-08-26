"""Evaluation battery v2 — hybrid pins (D1/H). Gates G2-G4 + pre-registered
G3 bar (100 held-out scenes, 8 rollouts/scene/seed, 3 seeds, all-seeds
positive + pooled > binomial 95% half-width). v1 phase-only battery archived
in results/battery_phase_only/."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import flow  # noqa: E402
from dataset import ACT_SCALE, H, make_dataset, scene_structure, success, to_canonical  # noqa: E402
from pin import HYBRID_PINS, circular_dist, extract_mags, extract_phases  # noqa: E402

OUT = Path(__file__).parent / "results"
HELD_SEED = 7777
N_ROLL = 8
SEEDS = [0, 1, 2]


def load(arm, seed):
    with open(OUT / f"arm_{arm}_seed{seed}.pkl", "rb") as f:
        return pickle.load(f)


def rollouts(entry, obs, angles, targets, mags, feats, rng, n_roll=N_ROLL):
    M = obs.shape[0]
    obs_r = np.tile(obs, (n_roll, 1))
    ang_r = np.tile(angles, n_roll)
    tgt_r = np.tile(targets, (n_roll, 1)) if targets is not None else None
    mag_r = np.tile(mags, (n_roll, 1)) if mags is not None else None
    fe_r = np.tile(feats, (n_roll, 1)) if feats is not None else None
    out = flow.rollout(entry["params"], entry["arm"], obs_r, ang_r,
                       entry["pins"], tgt_r, fe_r, rng, mag_targets=mag_r)
    return out.reshape(n_roll, M, H, 2)


def main():
    rng_tr = np.random.default_rng(flow.DATA_SEED)
    tr_scenes, tr_obs, tr_chunks, tr_angles = make_dataset(
        flow.N_TRAIN_SCENES, flow.N_DEMOS, rng_tr)
    tr_canon = to_canonical(tr_chunks, tr_angles[:, None])
    rng_he = np.random.default_rng(HELD_SEED)
    he_scenes, he_obs, he_chunks, he_angles = make_dataset(100, flow.N_DEMOS, rng_he)
    he_canon = to_canonical(he_chunks, he_angles[:, None])
    mag_norm = extract_mags(tr_canon.reshape(-1, H, 2), HYBRID_PINS).mean(axis=0)

    results = {"success": {}, "probe": {}, "adherence": {}, "diversity": {},
               "leakage": {}, "prior": {}}

    # ---- priors ----
    priors = {}
    for arm in ("F", "Frand"):
        for s in SEEDS:
            pins = flow.arm_pins(arm, s)
            res, mg = flow.prior_targets(tr_canon, pins)
            pr = flow.train_prior(tr_obs, res, mg, pins, seed=100 + s)
            he_res, he_mg = flow.prior_targets(he_canon, pins)
            pt, pm, conf = flow.prior_predict(pr, he_obs, pins)
            errs, gated, mag_rel = [], [], []
            for k, pin in enumerate(pins):
                mult = 2.0 if pin["mode"] == "modpi" else 1.0
                true_ang = np.arctan2(he_res[:, k, 1], he_res[:, k, 0]) / mult
                mask = ~np.isnan(pt[:, k])
                d = circular_dist(pt[mask, k], true_ang[mask],
                                  mod_pi=(pin["mode"] == "modpi"))
                errs.append(round(float(np.degrees(d.mean())), 1))
                gated.append(round(float(mask.mean()), 2))
                if pin.get("mag"):
                    mrel = np.abs(pm[mask, k] - he_mg[mask, k]) / (he_mg[mask, k] + 1e-9)
                    mag_rel.append(round(float(np.nanmean(mrel)), 3))
            priors[(arm, s)] = pr
            results["prior"][f"{arm}_s{s}"] = {
                "heldout_circular_err_deg": errs,
                "pin_active_fraction": gated,
                "heldout_mag_rel_err": mag_rel}

    # ---- success table ----
    def success_rate(chs):
        return float(np.mean([[success(sc, chs[r, i]) for i, sc in
                               enumerate(he_scenes)] for r in range(chs.shape[0])]))

    orac_t = extract_phases(he_canon[:, 0], HYBRID_PINS)
    orac_m = extract_mags(he_canon[:, 0], HYBRID_PINS)
    orac_m = np.where([p.get("mag", False) for p in HYBRID_PINS], orac_m, np.nan)
    per_seed = {}
    for arm, seeds in (("A", SEEDS), ("F", SEEDS), ("Frand", SEEDS),
                       ("Cdisp", [0]), ("Bphase", [0])):
        for s in seeds:
            e = load(arm, s)
            rng = np.random.default_rng(50 + s)
            if arm == "A":
                modes = {"plain": (None, None, None)}
            elif arm == "Cdisp":
                inv = np.array([sc["target"] for sc in he_scenes]) * ACT_SCALE
                modes = {"oracle": (inv, None, None)}
            elif arm == "Bphase":
                fe = flow.features(he_canon[:, 0], HYBRID_PINS, mag_norm)
                modes = {"oracle": (None, None, fe)}
            else:
                pr = priors[(arm, s)]
                pt, pm, _ = flow.prior_predict(pr, he_obs, e["pins"])
                ot = extract_phases(he_canon[:, 0], e["pins"])
                om = extract_mags(he_canon[:, 0], e["pins"])
                om = np.where([p.get("mag", False) for p in e["pins"]], om, np.nan)
                modes = {"prior": (pt, pm, None), "oracle": (ot, om, None)}
            for mode, (tgt, mg, fe) in modes.items():
                ch = rollouts(e, he_obs, he_angles, tgt, mg, fe, rng)
                per_seed.setdefault(f"{arm}-{mode}", {})[s] = round(success_rate(ch), 4)
                if arm == "F" and mode == "oracle":
                    flat = ch.reshape(-1, H, 2)
                    rphi = extract_phases(to_canonical(flat, np.tile(he_angles, N_ROLL)),
                                          HYBRID_PINS)
                    rmag = extract_mags(to_canonical(flat, np.tile(he_angles, N_ROLL)),
                                        HYBRID_PINS)
                    cphi = np.tile(orac_t, (N_ROLL, 1))
                    cmag = np.tile(orac_m, (N_ROLL, 1))
                    adh_p = [round(float(np.degrees(circular_dist(
                        rphi[:, k], cphi[:, k],
                        mod_pi=(HYBRID_PINS[k]["mode"] == "modpi")).mean())), 1)
                        for k in range(len(HYBRID_PINS))]
                    adh_m = [round(float(np.nanmean(
                        np.abs(rmag[:, k] - cmag[:, k]) / (cmag[:, k] + 1e-9))), 3)
                        if HYBRID_PINS[k].get("mag") else None
                        for k in range(len(HYBRID_PINS))]
                    results["adherence"][f"F_s{s}"] = {
                        "phase_err_deg": adh_p, "mag_rel_err": adh_m}
    results["success"] = per_seed

    # ---- G3 ----
    fp = np.array([per_seed["F-prior"][s] for s in SEEDS])
    a = np.array([per_seed["A-plain"][s] for s in SEEDS])
    fr = np.array([per_seed["Frand-prior"][s] for s in SEEDS])
    n = 100 * N_ROLL * len(SEEDS)
    half_w = 1.96 * np.sqrt(a.mean() * (1 - a.mean()) / n) * 2
    g3 = {"F_prior_per_seed": fp.tolist(), "A_per_seed": a.tolist(),
          "Frand_prior_per_seed": fr.tolist(),
          "pooled": {"F_prior": round(float(fp.mean()), 4),
                     "A": round(float(a.mean()), 4),
                     "Frand_prior": round(float(fr.mean()), 4)},
          "ci_half_width": round(float(half_w), 4),
          "pass": bool(all(fp - a > 0) and (fp.mean() - a.mean()) > half_w
                       and all(fp - fr > 0)
                       and (fp.mean() - fr.mean()) > half_w)}
    results["g3"] = g3

    # ---- G2 wrong-structure probe: F vs Bphase ----
    wrong = (np.arange(100) + 50) % 100
    cmd_t, cmd_m = orac_t[wrong], orac_m[wrong]
    for arm, s in (("F", 0), ("F", 1), ("F", 2), ("Bphase", 0)):
        e = load(arm, s)
        rng = np.random.default_rng(80 + s)
        if arm == "F":
            ch = rollouts(e, he_obs, he_angles, cmd_t, cmd_m, None, rng, n_roll=4)
        else:
            fe = flow.features(he_canon[wrong, 0], HYBRID_PINS, mag_norm)
            ch = rollouts(e, he_obs, he_angles, None, None, fe, rng, n_roll=4)
        flat = ch.reshape(-1, H, 2)
        rphi = extract_phases(to_canonical(flat, np.tile(he_angles, 4)), HYBRID_PINS)
        rmag = extract_mags(to_canonical(flat, np.tile(he_angles, 4)), HYBRID_PINS)
        ct, cm = np.tile(cmd_t, (4, 1)), np.tile(cmd_m, (4, 1))
        errs = [round(float(np.degrees(circular_dist(
            rphi[:, k], ct[:, k],
            mod_pi=(HYBRID_PINS[k]["mode"] == "modpi")).mean())), 1)
            for k in range(len(HYBRID_PINS))]
        mag_errs = [round(float(np.nanmean(np.abs(rmag[:, k] - cm[:, k])
                                           / (cm[:, k] + 1e-9))), 3)
                    if HYBRID_PINS[k].get("mag") else None
                    for k in range(len(HYBRID_PINS))]
        results["probe"][f"{arm}_s{s}"] = {"phase_err_deg": errs,
                                           "mag_rel_err": mag_errs}

    # ---- diversity (symmetric blocked scenes; modpi pins keep side free) ----
    sym = [i for i, sc in enumerate(he_scenes)
           if scene_structure(sc)["blocked"]
           and scene_structure(sc)["forced_side"] == 0][:10]
    if sym:
        e = load("F", 0)
        pr = priors[("F", 0)]
        idx = np.array(sym)
        pt, pm, _ = flow.prior_predict(pr, he_obs[idx], e["pins"])
        obs_r = np.repeat(he_obs[idx], 40, axis=0)
        ang_r = np.repeat(he_angles[idx], 40)
        ch = flow.rollout(e["params"], "F", obs_r, ang_r, e["pins"],
                          np.repeat(pt, 40, axis=0), None,
                          np.random.default_rng(5),
                          mag_targets=np.repeat(pm, 40, axis=0))
        canon = to_canonical(ch, ang_r).reshape(len(sym), 40, H, 2)
        lat = np.cumsum(canon[..., 1], axis=-1)
        side = np.sign(lat[np.arange(len(sym))[:, None], np.arange(40)[None, :],
                           np.abs(lat).argmax(axis=-1)])
        results["diversity"] = {
            "n_symmetric_scenes": len(sym),
            "overall_frac_positive_side": round(float((side > 0).mean()), 3),
            "per_scene_frac": np.round((side > 0).mean(axis=1), 2).tolist()}

    # ---- leakage: pinned content -> chunk decoder R^2 (held-out) ----
    tr_f = flow.features(tr_canon.reshape(-1, H, 2), HYBRID_PINS, mag_norm)
    he_f = flow.features(he_canon.reshape(-1, H, 2), HYBRID_PINS, mag_norm)
    tr_y = tr_canon.reshape(-1, H * 2)
    he_y = he_canon.reshape(-1, H * 2)
    A_ls = np.concatenate([tr_f, np.ones((len(tr_f), 1))], axis=1)
    W, *_ = np.linalg.lstsq(A_ls, tr_y, rcond=None)
    pred = np.concatenate([he_f, np.ones((len(he_f), 1))], axis=1) @ W
    r2 = 1 - ((he_y - pred) ** 2).sum() / ((he_y - he_y.mean(0)) ** 2).sum()
    results["leakage"] = {"linear_decoder_heldout_r2": round(float(r2), 3)}

    OUT.joinpath("battery.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "success"},
                     indent=2))
    print("success table:", json.dumps(results["success"], indent=2))
    print(f"BATTERY_FINAL={'G3_PASS' if g3['pass'] else 'G3_FAIL'}")


if __name__ == "__main__":
    main()
