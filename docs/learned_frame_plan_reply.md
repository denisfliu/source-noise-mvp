# Reply to plan review (learned_frame_toy_plan.md) — 2026-07-05, from Denis via laptop-side Claude

All four points accepted. Decisions and amendments below; treat this as an addendum to
the plan. Green light to start on your proposed order.

## 1. Pin dimensionality: adopt k>1, heatmap-first order approved

Your diagnosis is right — the bimodal bend style makes the lateral field's phase flip
by π across demos of a scene, so the plain coherence estimator self-cancels off the
progress axis, and a single u* would pin timing while missing obstacle geometry.
Amendments:

- **Run two coherence estimators in Step 1:** the plain one, γ(u,ω) as specced, and a
  sign-flip-invariant variant using the angle-doubling trick from directional
  statistics — γ₂(u,ω) = E_scenes |(1/N) Σ_demos exp(j·2φ_demo(u,ω))| — which is
  invariant to the π-flip and should recover the clearance structure if it's there.
  This also matches the mechanism: the flow target carries phase only up to sign, so
  mod-π structure is exactly what the channel can bind.
- **Pre-registered selection rule:** structural set = {(u,ω)} pairs with γ (or γ₂)
  above the same 0.6 threshold, u drawn from {progress axis, lateral axis} after grid
  search confirms the maxima land near those axes (G1 checks this). Pin mod-2π phase
  where plain γ selects, mod-π phase (pin the doubled angle's line, let x_t supply
  orientation) where only γ₂ selects. If the maxima do NOT land near the planted axes,
  stop and bring the heatmap — that's a G1 conversation, not a judgment call.
- Deliverable unchanged: bring both heatmaps before any flow training.
- Preference between your two mitigations: agreed — don't redesign the task to make
  timing matter; accept the heatmap's verdict and pin what's coherent.

## 2. Arms table correction accepted

A and C-disp retrain on the toy_frame dataset (obstacle scenes). Same seeds discipline
as the rest (A: 3 seeds, C-disp: 1 is fine — it's a reference point, not a claim).
Note the confound you caught in the README so nobody re-introduces it.

## 3. Claim framing accepted

Add the line: the (b)-claim is carried by the PAIR (F-prior > A) AND
(F-prior > F-rand-prior) — same architecture, same parameter count, same prior-net
capacity, wrong frame — and neither comparison alone. F-rand's prior net must be
trained with identical budget/recipe on its own (random) frame's phases, so the
"you just added a planner" reading has nowhere to stand.

## 4. Execution details locked

- (a) Sanity #2 extended: assert v-phase lands on the φ-line (mod π) at ω ∈ S, not
  just x_t phase constancy.
- (b) RNG-consumption discipline in the arm-F loss path, Phase 1 parity style: S=∅
  must consume identical draws in identical order and match baseline bit-for-bit.
- (c) G3 statistical bar, pre-registered here: held-out scenes bumped 50 → **100**,
  8 rollouts per scene per seed, 3 seeds. "F-prior ≥ A" means the per-seed success
  difference is positive for all 3 seeds AND the pooled difference exceeds the
  binomial 95% CI half-width (~±4 pts at n=800 per arm at these rates). State this in
  the README before the first arm trains. If dataset-gen cost makes 100 scenes
  awkward, say so — n=50 with 16 rollouts is an acceptable fallback, note the
  scene-level clustering caveat either way.

## 5. Step budget (the 15k/20k call) — decision

Denis's steer, now understood on his side: **do not extend to 20k by default.** Rule:

- If the 25k curve point (~03:30 UTC) is within noise of 89% (10-trial protocol), the
  plateau is real → keep 15k, no restarts, sweep proceeds as launched.
- If it shows a material rise (say >3 pts), don't switch unilaterally — flag it back
  with the restart-cost accounting for C_s42 and let Denis make the call.

Additionally, and this matters to him: **add per-arm success-vs-steps curves to the
eval plan** using the existing 2500-step checkpoints (the eval-curve protocol you
already ran on the 30k baseline; 10 trials/task is fine for the curve). Rationale:
Denis wants faster iteration signal, and the curve gives the 5k-scale read without
shortening any run — and arm C converging faster than A at matched steps is itself
evidence for the "grounding improves learning" motivation, arguably more on-point
than the converged endpoint. Caveat to note in the README: intermediate checkpoints
sit mid-cosine-schedule, so curves are fair arm-vs-arm at matched steps but not
comparable to completed-schedule short runs.

For the toy_frame work this concern doesn't bite — everything is CPU-minutes; don't
shorten anything there for speed.

## Proposed order: approved

Dataset generator + both coherence estimators + synthetic-recovery sanity (Step 1
through G1), CPU-only, then bring the heatmaps before any flow training. Go.
