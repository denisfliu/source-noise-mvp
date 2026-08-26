# Toy validation: cross-embodiment transfer via a shared invariant frame (Rung 1)

Execution spec written 2026-07-17 with Denis, for a box-side Claude to run in
`~/code/source-noise-mvp`. **CPU-only** (autograd, like toy/ and toy_frame/);
does NOT need or touch the GPU. Read first: `docs/cross_embodiment_plan.md`
(the design + why), `docs/REPRODUCTION_GUIDE.md` (method + prior findings),
`experiments/toy_frame/` (the code to reuse: coherence.py, pin.py, flow.py,
dataset.py, evaluate.py).

Build under `experiments/toy_embodiment/` — do NOT modify toy/ or toy_frame/
(both are cited by status docs). Note provenance at the top of any reused file.
Use the DECISIONS NEEDED protocol in `docs/status_latest.md` for any fork, as
in the toy_frame run.

## What this tests

H3 / aim (a): a geometric command frame learned on a set of embodiments
transfers to a NEW embodiment from few demos, when only the executor is
re-learned. Concretely, the pre-registered claim (mirrors toy_frame's
F-prior/A/F-rand pair):

    T (frozen shared front-half + B-adapted executor)  >  S (B from scratch,
    same few demos)   AND   T > Cond (same invariant conditioned, not pinned)

on held-out scenes of a held-out body B. Plus a stronger, standalone result if
it holds: **cross-embodiment coherence predicts the transfer gain** across body
pairs (G-predict).

## Embodiments (the divergence axis)

2D reach-with-obstacle task (reuse toy_frame's scene: target at radius 1–2,
random bearing; one circular obstacle near the start→target line). Bodies:

- `arm2` — 2-link planar arm (joint-angle action space, link lengths e.g. 1.0/0.8)
- `arm3` — 3-link planar arm (0.7/0.6/0.5)
- `arm4` — 4-link planar arm (0.6/0.5/0.4/0.3)
- `point` — holonomic point robot (2D position-delta action space) = the DRONE
  analog: no joint structure, free workspace, deliberately the most divergent.

All bodies act over H=20 step chunks. Arms: forward kinematics = tip position
from summed joint angles (trivial; no dynamics — kinematic sim). Point robot:
tip = integrated position deltas. Each body reaches the target and clears the
obstacle within its own workspace/joint limits (arms may be forced into
genuinely different tip paths than the free point robot — that divergence is the
point, do NOT force identical paths).

**Set A (front-half training):** `{arm2, arm3, arm4}`. **Held-out B:** `point`
as the primary transfer target (max divergence). Secondary/ablation: hold out
`arm4` instead (same-family, expect easier transfer) to give G-predict two
points on the divergence axis.

## The shared invariant frame (embodiment-agnostic)

Express the invariant in the **object/goal-centric task frame**: the tip's
trajectory relative to the scene, canonicalized by target bearing and
scale-normalized by target distance (so "reach the target" is comparable across
bodies of different reach). This coordinate references the world/task, not any
body's joints, so it is defined identically for arms and the point robot.

The invariant is the coherence-discovered structure of this tip trajectory
(temporal phase + magnitude of learned projections), NOT a hand-defined
displacement. Carry forward the toy_frame lessons:
- energy floor (≥1% per-axis energy) against zero-energy "dust";
- mod-π (angle-doubled γ₂) variant for any sign-flipping/bimodal component;
- pin complex coefficients (phase AND magnitude) at magnitude-coherent bins
  (CV<0.15), phase-only elsewhere — metric content matters (toy_frame v1→v2).

## Step 1 — cross-embodiment coherence (offline, before executors)

Adapt toy_frame's coherence estimator with "modalities → embodiments". For a
projection direction u and temporal bin ω, per scene take each BODY's
representative tip trajectory (mean over that body's demos, in the shared frame);
coherence = phase concentration ACROSS BODIES, averaged over scenes:

    γ(u,ω)  = E_scenes | (1/|A|) Σ_{body∈A} exp(j·φ_body(u,ω)) |     (mod-2π)
    γ₂(u,ω) = E_scenes | (1/|A|) Σ_{body∈A} exp(j·2·φ_body(u,ω)) |   (mod-π)

Structure = high cross-body agreement; embodiment-private actuation = low.
Select the shared frame S_A by the toy_frame rule (coherence>0.6, energy≥1%,
CV<0.15 for magnitude bins). **Frame is discovered from SET A ONLY — body B never
participates** (honest novel-body test).

Also compute, for reporting, a pairwise coherence scalar `c(bodyi, bodyj)` (mean
selected-bin agreement) — this is the divergence number that G-predict tests.

## Step 2 — front-half: scene→invariant prior (shared, frozen after training)

Small MLP: scene state (target, obstacle) → the S_A invariant (cos/sin per phase
bin + magnitude per magnitude bin), trained on set-A demos' shared-frame
invariants (circular mean over each scene's bodies+demos). This is the toy-scale
analog of the frozen VL front-half. Frozen for all transfer evals. Report
held-out-scene prediction error (circular deg + magnitude %).

## Step 3 — executors (per body; the ONLY thing re-learned on B)

Per-body flow-matching MLP (reuse toy_frame flow.py): input = scene state (+ that
body's proprio if helpful), invariant pinned into source noise via pin.py in the
shared frame, output = that body's action chunk. Train A-body executors on full
A data. The invariant→action realization is where the embodiment lives.

## Step 4 — transfer protocol and arms

Freeze frame (S_A) + prior. On body B, sweep adaptation demo count
`n ∈ {5,10,25,50}` (few scenes); evaluate on 100 HELD-OUT B scenes × 8 rollouts
× 3 seeds, no oracle (prior supplies the invariant). Arms:

| arm | B executor | invariant at inference | tests |
|---|---|---|---|
| **T** | adapted on n demos, pinned on S_A | frozen prior | the method |
| S | trained from scratch on same n demos, no pin | — | does the shared frame help vs demos alone |
| Cond | adapted on n demos, invariant as CONDITIONING input | frozen prior | pin channel vs conditioning (arm-B analog) |
| T-rand | adapted, pinned on a RANDOM frame (matched dims) | its own prior | is it the LEARNED frame, or any pin |
| T-oracle | T but ground-truth invariant | oracle | upper bound / prior-quality gap |

## Metrics

1. Cross-embodiment coherence heatmaps (γ, γ₂) + pairwise c(i,j).
2. Frame recovery vs the planted scene-level structure (G-frame sanity).
3. Transfer success (task success = reach target + clear obstacle) on held-out B
   scenes, per arm, per demo count.
4. Prior held-out error (circular deg / magnitude %).
5. Diversity (style spread at fixed invariant) + leakage R² (bounded, not
   transcription) — as in toy_frame.
6. Adherence: realized-vs-commanded shared-frame invariant on B (does the pin
   bind in the new body).

## Gates (pre-register the stats bar: n=2400/arm pooled, ~±4 pts binomial half-width)

- **G-frame:** coherence recovers a non-trivial shared frame for set A, and
  c(arm,arm) > c(arm,point) (the estimator ranks divergence sensibly). Synthetic
  recovery exact. FAIL ⇒ estimator/dataset bug, stop and fix.
- **G-transfer:** T > S AND T > Cond on held-out B, all seeds, at low n
  (especially n=5,10). AND T > T-rand (the learned frame, not any pin).
- **G-predict (the standout if it holds):** the T−S transfer gain is larger for
  the same-family held-out body (arm4) than for the divergent one (point), and
  ordering matches the coherence numbers c(·). I.e. coherence predicts transfer.
- **G-nocollapse:** diversity preserved, leakage bounded.

Outcome logic:
- All pass ⇒ factorization holds at toy scale; write Rung 2 (robosuite arms +
  small diffusion policy, gsplat perception) spec.
- G-transfer passes but only for arm-family B, fails for point ⇒ the honest
  reading: transfer works within a morphology class, not across the arm/drone
  gap — report the coherence threshold where it breaks (still a real result).
- G-transfer fails even for arm-family B ⇒ the shared-frame executor isn't
  enough; diagnose (frame content? executor capacity? invariant not
  body-agnostic?) before Rung 2.

## Sanity sequence (before the arm sweep)

1. Synthetic cross-body coherence: plant a known shared (u,S) across synthetic
   multi-body fields; confirm the estimator recovers it and that adding a
   divergent body lowers c correctly.
2. Pin preservation (temporal, per pin.py): phase (and magnitude at hybrid bins)
   of x_t constant over t for the shared-frame invariant; v-phase on the φ-line
   (mod π). Reuse toy_frame's numerical check.
3. Identity mode (S=∅) executor matches an unpinned baseline loss curve
   bit-for-bit given the same seed (parity discipline).
4. Overfit one body on 10 scenes, oracle-pin inference: shape adherence tight
   even early.

## Implementation notes

- Same stack: autograd, CPU, 3×128 relu MLPs, ~10k Adam steps. Parallelize arms
  across cores. Full battery target: < a CPU-day including build/iterate; pure
  compute < ~2 h.
- Reuse toy_frame coherence.py/pin.py/flow.py/dataset.py by import or documented
  copy; the new pieces are the multi-body sim (FK + point robot), the cross-body
  coherence axis, the frozen front-half, and the freeze-and-adapt harness.
- Results: JSONs + `experiments/toy_embodiment/README.md` in the exact table
  style of toy_frame's README, Reading section written for someone who has read
  only the REPRODUCTION_GUIDE. Report circular quantities in degrees; state N,
  seeds, tolerances.
- Frozen held-out B scene split: fixed seed, pre-registered before any training.
