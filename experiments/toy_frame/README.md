# Learned-frame toy: results (2026-07-05, hybrid battery v2 — FINAL)

Two batteries were run. v1 (phase-only pins) is archived in
results/battery_phase_only/ and diagnosed below; v2 (hybrid pins, Denis-
approved Option H) is the headline. Dataset generator was fixed between
batteries (endpoint-vanishing bump + clearance enforcement: demo ceiling
1.00, endpoints exact; v1's ceiling was 0.81 — numbers across batteries are
not comparable). Hybrid set S_H frozen and pre-registered BEFORE arm
training: phases {(prog,0),(prog,1)} mod-2pi + {(lat,1),(lat,2)} mod-pi;
magnitudes at {(prog,0),(prog,1),(lat,1)} (rule: coherence>0.6, energy>=1%,
magnitude CV<0.15 — the CV gap in the data is 0.048/0.081 vs 0.226+).

## Gate verdicts (v2)

| gate | verdict | evidence |
|---|---|---|
| G1 | PASS | step1 re-run on fixed generator; axes/bins recovered |
| G2 | **PASS** | wrong-structure probe, F vs B-phase: phase 1.1-1.5 deg vs 26-37 deg; commanded MAGNITUDE executed at 1.6-2.5% rel err vs B-phase's 86-92% (B-phase ignores the magnitude features entirely) |
| G3 | **PASS** | table below: +17.3 pts over A, all seeds, 4.3x the pre-registered bar |
| G4 | PASS | side diversity 0.487 on symmetric scenes (mod-pi pins stay orientation-free under magnitude pinning); leakage R^2 0.209 (up from v1's 0.08 as expected, far from transcription) |

## G3: held-out no-oracle success (100 scenes x 8 rollouts x 3 seeds)

| arm | pooled success | per-seed |
|---|---|---|
| **F-prior (no oracle)** | **0.623** | 0.622 / 0.619 / 0.629 |
| F-oracle | 0.640 | 0.644 / 0.622 / 0.654 |
| A (floor) | 0.451 | 0.440 / 0.450 / 0.462 |
| F-rand-prior | 0.441 | 0.415 / 0.508 / 0.401 |
| F-rand-oracle | 0.448 | — |
| C-disp-oracle | 0.438 | — |
| B-phase-oracle | 0.558 | — |

Readings:
1. **The (b) claim, carried by the pre-registered pair:** F-prior beats A by
   +17.3 pts AND beats F-rand-prior (same architecture, same prior capacity,
   wrong frame) by +18.2 pts; bar was +-4.0. Geometric organization in the
   COHERENCE-LEARNED frame — not "any pin plus a planner" — improves
   no-oracle success on held-out scenes.
2. **The prior captures nearly all oracle value** (62.3 vs 64.0): the
   scene->structure map generalizes (phase err 0.3-7.8 deg held-out;
   magnitude 1.4-11% except clearance at ~25%, still sufficient).
3. **Discovered structure beats the hand-defined invariant on task success:**
   C-disp-oracle (43.8) ~ A. Endpoint displacement was never the binding
   constraint here; clearance amplitude is, and only the learned frame
   carries it.
4. B-phase-oracle at 55.8 shows conditioning uses the information too — but
   under contradiction (G2) it follows at 20-30x worse fidelity, and it
   needs the oracle at eval; the pin's prior mode needs only the scene.

## Why v1 failed and v2 passes (framing per Denis, 2026-07-05)

The paper's phase-only form restriction is not a neutral simplification that
happened to fail — in images, magnitude IS appearance, so phase/magnitude
maps exactly onto structure/appearance. The toy shows this split is
image-specific: in control, safety-critical structure is partly metric
(clearance amplitude), so structure straddles both components. v1's G3
failure mode (F-prior BELOW A, not merely at it) is the dropout economics
operating again: the always-on pin displaced obs->clearance learning while
the amplitude slot stayed Rayleigh-random. The hybrid pin closes exactly
that slot; the coherence criterion extends to complex coefficients with a
magnitude-agreement estimator (CV), which is Option H's bin selection in
miniature.

Wrinkle for completeness: under WRONG-structure commands the (prog,0)
magnitude (total displacement) is executed at only ~23% rel err while its
phase is exact — when the commanded metric content conflicts with obs, the
model compromises on magnitude before phase. Consistent with magnitude
carrying the obs-coupled content.

## Notes

- Magnitude overwrite breaks the Rayleigh marginal of the pinned coordinate
  (exact-mode trade-off, same as the displacement pin). Recorded, accepted.
- Pre-registered stats: n=2400/arm pooled; binomial 95% half-width ~4 pts.
- Reproduce: flow.py --arm {A,F,Frand,Cdisp,Bphase} --seed {0,1,2};
  evaluate.py. Step-1 artifacts in results/step1/.
