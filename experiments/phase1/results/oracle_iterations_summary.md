# Sim-state oracle iterations (D2/D5), 2026-07-05 — outcome and findings

Four oracle variants were tested against phase1_C_s42 checkpoints (5k, 12.5k),
2 trials x 10 spatial tasks each. All scored 0/20 task success; the
instrumented episodes localize WHY, and the why is informative.

| version | design | instrumented finding |
|---|---|---|
| v1 minimal | 7-dim cmd toward manipuland, gripper at dataset mean | far-field steering WORKS (EE homes 0.33->0.08 m, calibration-accurate); near object the demo prior overrides the pin; no grasp is ever commanded |
| v2 minimal2 | 6-dim cmd, gripper UNPINNED, bowl->goal switch on 3cm lift | vision DOES grasp (aperture closes, bowl lifts 3.2 cm, switch fires) — then drops: the oracle still commands DESCEND through the grasp/lift window |
| v3 +deadband | near-manipuland: gentle lift bias instead of descend | 0/20 — the calibration OFFSETS are state-dependent (measured at home states; wrong near objects), mis-translating small commands |
| v4 slope-only | offsets dropped from the inversion | 0/20 |

## Conclusions

1. **Far-field metric steering through the noise channel is PROVEN in the
   real sim loop** (v1/v2 approach phases; calibration R^2 0.87-0.97).
2. **Chunk displacement is the wrong command language for contact phases.**
   At grasp states the correct "invariant" is the demo-like maneuver
   (hold/close/lift), which no simple geometric oracle emits; the always-on
   pin binds hard enough that a wrong contact-phase command destroys the
   maneuver vision is executing. Not a channel failure — an oracle-design
   gap. (Echoes toy_frame: structure straddles more descriptors than a
   single hand-defined quantity.)
3. Scripted per-task oracles that fix this converge on "inject demo-like
   invariants per phase" — increasing oracle intelligence and weakening the
   fairness argument. Diminishing returns; iterations stopped at 4.

## Proposed path (D6, flagged in status_latest): learned invariant prior

The toy_frame result showed the working configuration is pin-from-PRIOR, not
pin-from-geometric-oracle: F-prior succeeded precisely where oracles are now
failing. LIBERO translation: train a small net p(invariant | image, state)
on demo frames (the Phase-2 prior head in miniature, per D4's "always-on
with a prior at inference" constraint), serve arm C with per-replan pins
from the prior => arm C becomes a self-contained policy, standard success
evals become valid (apples-to-apples with A), and H1 gets its success
comparison back — in the paper-faithful architecture.
