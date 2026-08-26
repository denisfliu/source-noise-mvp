"""Step 1 (Rung 1): cross-embodiment coherence discovery + the G-frame gate.

Runs BEFORE any executor training (per the plan's heatmap-first order). Emits:
  results/step1/synthetic.json   - estimator recovers a planted shared frame;
                                    a divergent body lowers coherence
  results/step1/gamma.npy, gamma2.npy, selection.json - shared frame over set A
  results/step1/divergence.json  - pairwise c(i,j); arm-vs-arm vs arm-vs-point
  results/step1/heatmaps.txt      - ascii gamma / gamma2
  results/step1/README.md         - summary + G-frame verdict

G-frame PASSES iff (1) synthetic recovery is exact (theta err <= 15 deg) AND a
divergent body lowers coherence; (2) select_structure finds a non-trivial shared
frame over set A (g1_pass, non-empty); (3) divergence ordering is sensible:
mean c(arm,arm) > mean c(arm,point) (the point robot = drone analog is the most
divergent from the arm family).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embodiments as emb
import mb_dataset as ds
import coherence_xembod as cx

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "step1")
SET_A = ["arm2", "arm3", "arm4"]
THETAS = np.linspace(0.0, np.pi, 90)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    # --- sanity: synthetic cross-body recovery -------------------------------
    syn = cx.synthetic_recovery(rng, thetas=THETAS)
    json.dump(syn, open(os.path.join(OUT, "synthetic.json"), "w"), indent=2)
    syn_ok = syn["theta_err_deg"] <= 15.0 and \
        syn["gamma_with_divergent_body"] < syn["gamma_at_planted"] - 0.1
    print("synthetic:", json.dumps(syn))

    # --- real multi-body dataset --------------------------------------------
    bodies = emb.make_bodies()
    scenes, obs, angles, chunks = ds.make_dataset(bodies, n_scenes=200,
                                                  n_demos=8, rng=rng)

    # per-body success ceiling (are the achieved demos actually solving it?)
    ceil = {}
    for b in bodies:
        ok = [ds.success(scenes[s], chunks[b][s, d])
              for s in range(len(scenes)) for d in range(chunks[b].shape[1])]
        ceil[b] = round(float(np.mean(ok)), 3)
    print("demo success ceiling per body:", ceil)

    # --- cross-body coherence over SET A (arms only) ------------------------
    g, g2 = cx.coherence_over(chunks, angles, SET_A, THETAS)
    np.save(os.path.join(OUT, "gamma.npy"), g)
    np.save(os.path.join(OUT, "gamma2.npy"), g2)
    sel = cx.tfc.select_structure(g, g2, THETAS)   # axes default = {0, pi/2}
    json.dump(sel, open(os.path.join(OUT, "selection.json"), "w"), indent=2)
    print("selection g1_pass:", sel["g1_pass"], "n_selected:", len(sel["selected"]))
    for s in sel["selected"]:
        print("   ", s)

    # --- pairwise divergence over ALL bodies --------------------------------
    all_bodies = ["arm2", "arm3", "arm4", "point"]
    cij = cx.pairwise_c(chunks, angles, all_bodies, THETAS, sel["selected"])
    arm_arm = [v for k, v in cij.items()
               if "point" not in k]
    arm_pt = [v for k, v in cij.items() if "point" in k]
    mean_arm_arm = round(float(np.mean(arm_arm)), 3) if arm_arm else float("nan")
    mean_arm_pt = round(float(np.mean(arm_pt)), 3) if arm_pt else float("nan")
    div = {"pairwise_c": cij, "mean_arm_arm": mean_arm_arm,
           "mean_arm_point": mean_arm_pt, "ceiling": ceil}
    json.dump(div, open(os.path.join(OUT, "divergence.json"), "w"), indent=2)
    print("pairwise c:", json.dumps(cij))
    print(f"mean c(arm,arm)={mean_arm_arm}  mean c(arm,point)={mean_arm_pt}")

    # --- heatmaps -----------------------------------------------------------
    hm = (cx.tfc.ascii_heatmap(g, THETAS, "gamma (cross-body, set A)") + "\n\n" +
          cx.tfc.ascii_heatmap(g2, THETAS, "gamma2 (mod-pi, cross-body, set A)"))
    open(os.path.join(OUT, "heatmaps.txt"), "w").write(hm)

    # --- G-frame verdict ----------------------------------------------------
    div_ok = (not np.isnan(mean_arm_arm) and not np.isnan(mean_arm_pt)
              and mean_arm_arm > mean_arm_pt)
    g_frame = bool(syn_ok and sel["g1_pass"] and len(sel["selected"]) > 0 and div_ok)
    verdict = {"synthetic_ok": bool(syn_ok), "frame_found": bool(sel["g1_pass"]),
               "n_selected": len(sel["selected"]),
               "divergence_ordering_ok": bool(div_ok),
               "mean_arm_arm": mean_arm_arm, "mean_arm_point": mean_arm_pt,
               "G_FRAME_PASS": g_frame}
    json.dump(verdict, open(os.path.join(OUT, "verdict.json"), "w"), indent=2)
    write_readme(syn, sel, div, ceil, verdict)
    print("G_FRAME_PASS=" + str(g_frame))
    print("STEP1_DONE=ok")


def write_readme(syn, sel, div, ceil, verdict):
    L = ["# toy_embodiment Step 1 — cross-embodiment coherence (G-frame)", "",
         "Rung 1 of docs/cross_embodiment_plan.md. Task-space (tip-delta) actions",
         "for all bodies (invariant linear, pin exact); embodiment = reach +",
         "radial-authority feasibility (embodiments.py). Coherence = phase",
         "agreement ACROSS BODIES doing the same scene.", "",
         "## Synthetic recovery (sanity)",
         f"- planted theta {syn['planted_theta_deg']}deg / omega {syn['planted_omega']}"
         f" -> recovered {syn['recovered_theta_deg']}deg (err {syn['theta_err_deg']}deg)",
         f"- coherence at planted bin {syn['gamma_at_planted']}; with a divergent"
         f" body added {syn['gamma_with_divergent_body']} (should drop)", "",
         "## Demo success ceiling per body (achieved tip paths solve the task?)",
         "| " + " | ".join(ceil) + " |", "|" + "---|" * len(ceil),
         "| " + " | ".join(str(ceil[b]) for b in ceil) + " |", "",
         "## Shared frame over set A {arm2,arm3,arm4}",
         f"- g1_pass = {sel['g1_pass']}; selected pins:"]
    for s in sel["selected"]:
        L.append(f"    - {s}")
    L += ["", "## Pairwise cross-body coherence c(i,j) on selected bins",
          "| pair | c |", "|---|---|"]
    for k, v in div["pairwise_c"].items():
        L.append(f"| {k} | {v} |")
    L += ["", f"- mean c(arm,arm) = {div['mean_arm_arm']}  |  "
          f"mean c(arm,point) = {div['mean_arm_point']}",
          "- Reading: the point robot (drone analog, unconstrained) should be the",
          "  most divergent from the arm family, so c(arm,arm) > c(arm,point).", "",
          "## G-FRAME VERDICT",
          f"- synthetic_ok: {verdict['synthetic_ok']}",
          f"- frame_found: {verdict['frame_found']} ({verdict['n_selected']} pins)",
          f"- divergence_ordering_ok: {verdict['divergence_ordering_ok']}",
          f"- **G_FRAME_PASS = {verdict['G_FRAME_PASS']}**", "",
          "See heatmaps.txt for the gamma/gamma2 grids; *.npy/*.json are the raw",
          "artifacts. Next: Steps 2-4 (front-half prior, per-body executors,",
          "freeze-and-adapt transfer) per docs/toy_embodiment_plan.md."]
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
