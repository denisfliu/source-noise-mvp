# Step 1 results: coherence maps + G1 (2026-07-05)

Synthetic recovery sanity: PASS (both estimators recover planted (u, S) to
within 1 degree at gamma ~1.0; the plain estimator correctly does NOT fire on
the sign-flipped plant).

## G1: PASS — but the raw selection rule over-pins

Dataset: 200 scenes x 8 demos, canonical frame. Coherence and energy at the
two planted axes (theta grid maxima land on-axis, so axes-restricted view):

| bin | g(prog) | g2(prog) | E%(prog) | g(lat) | g2(lat) | E%(lat) |
|---|---|---|---|---|---|---|
| 0 | 1.000 | — | 97.8 | 0.814 | — | 1.4 |
| 1 | 0.970 | 0.957 | 1.8 | 0.773 | **0.907** | **40.3** |
| 2 | 0.962 | 0.926 | 0.3 | 0.404 | 0.354 | 17.6 |
| 3 | 0.768 | 0.512 | ~0 | 0.361 | 0.314 | 24.8 |
| 4-9 | ~0.29 | 0.78-1.0 | ~0 | 0.35-0.64 | 0.32-0.91 | <8 |

Reading:
- Timing structure where planted: progress bins 1-2, mod-2pi, strongly
  coherent (0.96-0.97).
- Clearance structure where planted: lateral bin 1 = 40% of lateral energy,
  gamma2 0.907 vs gamma 0.773 — mod-pi coherence, exactly the bend-side
  bimodality signature. A mod-pi pin here carries obstacle geometry while
  preserving side diversity BY CONSTRUCTION.
- Style where planted: lateral bins 2-4 (the wiggle band) are energetic but
  incoherent — correctly excluded by any threshold.
- PITFALL the raw rule hits: progress bins 4-9 show gamma2 up to 0.997 while
  carrying ~0.0% energy — deterministic numerical dust of the smooth speed
  profile. Phase coherence of near-zero coefficients is real but contentless;
  the raw gamma>0.6 rule selects ~15 (axis, bin) pairs, most of them dust.

## Proposed amendment (needs sign-off before flow training)

Selection = coherence > 0.6 AND per-bin energy fraction >= 1% on that axis:

    S = { (prog, 0) mod-2pi trivial sign, (prog, 1), (prog, 2) mod-2pi,
          (lat, 1) mod-pi }            [4 pins; optionally (lat, 0) — see below]

(lat, 0) subtlety: DC-sign of the lateral field = net side. Structural for
forced-side scenes, style for symmetric scenes (gamma 0.814 is the mixture).
Pinning it unconditionally would collapse side diversity on symmetric scenes.
Proposal for Step 3: the scene prior predicts (cos, sin) scaled by the
predicted resultant length r (its own confidence); pin only where r > 0.6.
Confidence-gated pinning generalizes the fixed-S rule and falls out of the
same circular-mean training target.
