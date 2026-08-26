# toy_embodiment (Rung 1) — cross-embodiment transfer via a shared invariant frame

Rung 1 of `docs/cross_embodiment_plan.md`, testing aim (a) / H3 at toy scale
(2D, CPU). Built and run 2026-07-17. Self-contained summary for a reader who has
seen only `docs/REPRODUCTION_GUIDE.md`. Code: `experiments/toy_embodiment/`
(reuses `experiments/toy_frame/` coherence/pin/flow machinery).

## Question

Does a geometric command frame learned on a SET of embodiments transfer to a NEW
embodiment from few demos, when only the executor is re-learned? I.e. is the VLA
factorizable into a frozen, embodiment-shared front-half (perception -> geometric
invariant) and a small, embodiment-specific executor (invariant -> this body's
actions)?

## Design (Denis-approved fork)

All bodies act in **task-space (tip position-delta)** coordinates, so the shared
invariant (tip-trajectory phase/magnitude in the canonical task frame) stays
**linear** in every body's action space and the source-noise pin is carried
exactly. The embodiment difference is therefore reachability/feasibility, not the
action parameterization:
- `arm2/arm3/arm4` — planar arms, reach 1.8/2.0/2.2, lose radial authority near
  full extension (leading-order kinematics, `embodiments.py`).
- `point` — holonomic, unconstrained: the **drone analog**, deliberately the most
  divergent body.
Task = 2D reach with an obstacle detour (reuses `toy_frame`'s scene + ideal path).
Set A (front-half training) = {arm2, arm3, arm4}; held-out B = {point, arm4}.

## Step 1 — cross-embodiment coherence (G-frame): PASS

Coherence = phase agreement ACROSS BODIES doing the same scene (the `toy_frame`
estimator with "modalities" -> "embodiments").
- Synthetic recovery exact (planted axis recovered to 0.6 deg; adding a divergent
  body drops coherence 1.0 -> 0.69).
- Demo success ceilings show the constraint is real and graded: arm2 0.62,
  arm3 0.83, arm4 1.00, point 1.00.
- **Divergence ordering correct**: c(arm,arm)=0.93 > c(arm,point)=0.80 — arms
  share more structure with each other than with the point robot. (On the frozen
  S_A over the held-out set: c(arm4,setA)=0.933 > c(point,setA)=0.893.)
- Raw coherence>0.6 over-selected (22 "dust" pins); energy-floor + CV gating froze
  **S_A = 8 pins** (progress axis omega 0-1, lateral axis omega 1-6). Larger than
  `toy_frame`'s 4 — the lateral detour spreads energy across bins; worth tighter
  gating before scale (see Rung 2 pre-step).

## Steps 2-4 — freeze-and-adapt transfer (G-transfer): PASS

Freeze S_A + a scene->invariant prior trained on set A; adapt ONLY the executor on
held-out body B's few demos. Eval on 100 held-out scenes x 8 rollouts x 3 seeds.
Pooled at low n (<=10):

| held-out B | T (transfer) | S (scratch) | Cond | T-rand | verdict |
|---|---|---|---|---|---|
| point (drone analog) | **0.449** | 0.356 | 0.399 | 0.361 | T > all |
| arm4 (same family)   | **0.387** | 0.330 | 0.331 | 0.303 | T > all |

On BOTH bodies, T beats scratch (transfer helps), conditioning (the pin channel
beats a conditioned input), and random-frame (it is the LEARNED frame specifically).
**G_transfer_pass = True.** Signal is cleanest at n=10-25; n=5 is very noisy (some
single-seed configs collapse, e.g. arm4/s2/n5 T=0.095) — hence the multi-seed
pooling. **T-oracle sits ~0.10-0.15 above T everywhere**, confirming prior quality
is the success ceiling (same lever as the LIBERO result).

## G-predict — did NOT hold with 2 bodies (a hypothesis, reversed)

Naive expectation: higher coherence-with-training -> bigger transfer gain. Observed
the opposite: point (coherence 0.893) had gain +0.094 while arm4 (coherence 0.933)
had gain +0.057. With n=2 bodies this is directional only, but the reversal is a
plausible refinement: a body already SIMILAR to the training set is "easy" enough
that scratch does okay, so transfer adds less; a more DIVERGENT body benefits more
from the structured prior. Resolving this needs a body-ladder spanning the
coherence axis (Rung 2).

## Reading / takeaways

1. The cross-embodiment factorization works at toy scale: a coherence-learned,
   embodiment-shared frame + prior transfers to a new body (including a maximally
   divergent point robot) when only the executor is adapted — beating scratch,
   conditioning, and a random frame. First positive evidence for aim (a)/H3.
2. Success is prior-limited, not channel-limited (T-oracle >> T): a better
   invariant predictor is the lever, exactly as at LIBERO scale.
3. "Coherence predicts transfer gain" is not a simple monotonic law (may even
   reverse) — a real open question, not a settled gate.

## Caveats

Small study (2 bodies, 3 seeds, noisy n=5); leading-order kinematic embodiment
model (not full IK); S_A is 8 pins (loose). All addressed/scaled in Rung 2.

## Reproduce

`experiments/toy_embodiment/`: `step1_run.py` (G-frame, CPU ~1 min),
`battery.py` (transfer battery, CPU ~72 min for the full grid). Results in
`results/step1/` and `results/transfer/`.
