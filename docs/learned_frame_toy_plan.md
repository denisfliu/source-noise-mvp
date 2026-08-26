# Toy validation: learned geometric frame (coherence-discovered temporal-phase pin)

Plan written 2026-07-04 with Denis, for execution in `~/code/source-noise-mvp` on the
box. CPU-only (autograd), does not touch the running Phase 1 GPU sweep. Read first:
`experiments/toy/README.md` (esp. the dropout addendum), `docs/mvp_plan.md`,
the source paper's method section (subspace phase-invariant sources; summary below
is self-contained if the PDF isn't on the box).

## Why this experiment exists

The project's real motivation is not command-following. It is: **ground actions in a
geometric frame the model itself learns**, so that (a) tasks generalize once the
embodiment is learned and (b) learning improves because actions have geometric
identity rather than being arbitrary vectors.

The Phase 1 design (hand-defined displacement pin, sim-state oracle at eval) tests the
*delivery mechanism* — the noise channel binds harder than conditioning — but cannot
test the motivation, for a structural reason: because the pin is always-on (see the
dropout addendum), arm C is never evaluated without oracle information arm A doesn't
get. Every C measurement conflates "better-organized learning" with "was told where to
go." Arm B controls the channel claim, not the learning claim.

The source paper has **two** ideas, and the MVP so far transplanted only one:

1. **Structure is discovered, not hand-picked**: a low-rank subspace where Fourier
   phase *synchronizes across modalities of the same scene* (a coherence objective,
   no auxiliary encoder). Crucially this criterion is **external to the generation
   loss** — the flow loss enforces the invariant but never votes on what it is.
2. **Delivery**: pin the discovered invariant into the source noise so the regression
   target carries it (this is what Phase 1 transplanted, with a hand-defined stand-in).

This experiment transplants idea 1. The form restriction is the paper's real inductive
bias and we keep it: structure = **phase** (where/when, not how-much) of **learned
projections**; here, the temporal Fourier phase of projected action fields — literally
"the shape of the motion." The coherence criterion's modality set, adapted to actions:
**multiple demonstrations of the same scene** — structure is the shape all executions
share; style is the execution-private residual.

One organ the paper never needs: at re-rendering inference, structure comes free from
the input image. Action generation has no input action, so a **scene→structure prior**
must supply the pin (always-on, per the dropout finding — there is no unpinned mode).
The no-oracle eval through this prior is the honest test of motivation (b): both arms
consume only (scene), and any win for the pinned arm is organization, not obedience.

**Known-outcome guard:** the flow loss alone would learn transcription (a target
predictable from the source is trivially fit), i.e. "structure = everything" — the
action-space version of the paper's appearance leak. Separation of powers is the
design: coherence defines structure offline; flow training only enforces it.

## Environment and dataset (ground truth is planted, so frame recovery is checkable)

Extend the existing 2D point-robot toy (H=20 delta-action chunks, ACT_SCALE
normalization — keep it, the underfit failure without it is documented):

- **Scene** = target position (radius 1–2, angle uniform in [-180°,180°)) + one
  circular obstacle (radius ~0.25) placed on or near the straight start→target line
  with random lateral offset. Obs vector = (target_xy, obstacle_xy, obstacle_r).
- **Success** = final position within tol of target AND no timestep inside the
  obstacle disk.
- **Demos: N=8 per scene, M=200 train scenes + 50 held-out scenes.** Scripted
  generator with an explicit structure/style split (this is the planted ground truth):
  - *Structural (scene-determined, shared by all demos of a scene):* endpoint;
    detour clearance around the obstacle; progress-timing profile (e.g. slow-down
    near the obstacle) — i.e. the low-frequency shape of the progress coordinate.
  - *Style (demo-private):* bend side when both sides are clear (keep the bimodality
    — it is the diversity metric), lateral wiggle amplitude/phase, small timing jitter.
- **Canonical frame (default design choice):** rotate each chunk by the scene's target
  bearing so progress ≈ +x for every scene. Coherence learning and pinning operate in
  this frame (per-scene rotated u_s = R(θ_s)·u*; the subspace construction is per-
  sample-orthonormal, Gaussian calibration unaffected). The rotation uses only scene
  information, so it is legitimate at inference. Fallback if this feels too helpful:
  a cone dataset (targets within ±45° of +x, global frame) as an ablation.

## Step 1 — learn the frame offline (before any flow training)

For a chunk a ∈ R^{H×2} in the canonical frame, direction u ∈ R^2 (d=2 ⇒ u is a
single angle θ_u — grid search suffices, no gradients needed), scalar field
z = a·u ∈ R^H, rFFT over time → 11 bins, φ(u,ω) = phase of bin ω.

**Coherence** (intra-scene phase concentration, averaged over scenes):

    γ(u, ω) = E_scenes | (1/N) Σ_demos exp(j·φ_demo(u, ω)) |

γ ∈ [0,1]; 1 means every demo of a scene agrees on that phase (structure), ~1/√N
means demo-private (style). Deliverable: the γ(θ_u, ω) heatmap data (θ_u grid ×
11 bins) — this is the main diagnostic artifact.

Select u* = argmax aggregated coherence and the structural frequency set
S = {ω : γ(u*,ω) > 0.6} (fallback: top-4). **Gate G1 (frame recovery):** u* within
~15° of the planted progress axis and S contains the planted low frequencies (expect
roughly ω ∈ {0..3}). If G1 fails, fix the estimator/dataset before proceeding —
nothing downstream is interpretable.

## Step 2 — pin construction (paper's eq. 6, temporal version)

Training, per sample, from the sample's own action (analog of the paper using the
source image's phase): with ε ∼ N(0,I) over R^{H×2}, project ζ = ε·u*, rFFT, and at
frequencies in S only, keep the noise's magnitude and overwrite the phase:

    F(ζ̃)(ω) = |F(ζ)(ω)| · exp(j·φ_a(ω))   for ω ∈ S;  unchanged for ω ∉ S
    ε̃ = ε + u*·(ζ̃ − ζ)ᵀ-lift   (complement untouched)

Preservation (why this survives without linearity): ε̃ and a share phase at ω ∈ S, so
the interpolant x_t = t·ε̃ + (1−t)·a (openpi convention, same as the existing toy) is
a sum of same-phase complex numbers at those bins — phase held for all t — and the
flow target v = ε̃ − a carries it up to sign. Verify numerically in the sanity
sequence; do not take it on faith.

Notes:
- **Phase-only pins shape, not scale.** At the DC bin, phase = sign of the sum, so the
  endpoint *magnitude* is NOT pinned (unlike arm C's displacement pin, which fixes the
  full coefficient). Expected consequence: looser endpoint precision than arm C, tighter
  shape adherence. Optional hybrid ablation: full-coefficient pin at DC + phase-only
  elsewhere.
- **Always-on** (dropout addendum applies unchanged): the pin is applied to 100% of
  training samples. No dropout arm — we already know how that ends.

## Step 3 — scene prior (the no-oracle path)

Per training scene, the structural phase is well-defined by construction (demos share
it): target = circular mean over the scene's demos of exp(j·φ(u*,ω)), ω ∈ S. Train a
small MLP: obs → (cos, sin) per ω ∈ S (normalize output to the unit circle; MSE on the
circle handles wraparound). Report held-out-scene circular error — the prior's own
generalization is part of the result, not a nuisance.

## Arms and eval modes

| Arm | Training | Inference pin source | Role |
|---|---|---|---|
| A | no pin | — | floor (exists) |
| C-disp | displacement pin (existing arm C) | oracle | reference: hand-defined invariant |
| F | learned-frame phase pin (u*, S) | oracle (phase from a ground-truth demo of the eval scene) | mechanism at learned invariant |
| F-prior | same checkpoint as F | scene prior (no oracle) | **headline**: motivation (b) test |
| F-rand | phase pin with random u, random \|S\|-sized freq set | oracle / its own prior | control: does the *learned* frame matter, or any pin? |
| B-phase | same phase invariant as conditioning input | (input) | channel control at matched information (optional, run if time permits) |

3 seeds for A, F, F-rand; 1 for the rest. F-prior/F-oracle are eval modes of one
checkpoint, not separate trainings.

## Metrics

1. **Frame recovery** (G1 above) + the γ heatmap.
2. **Shape adherence**: circular distance between commanded and realized phases at
   ω ∈ S; plus a trajectory-shape distance (e.g. DTW or L2 after endpoint alignment)
   to a demo with the commanded structure.
3. **Wrong-structure probe**: command the phase of a *different* scene's structure;
   primary metric err-to-command (per the toy's established finding: adherence
   precision, not follow rate, is where channels separate). Compare F vs B-phase.
4. **No-oracle success on held-out scenes**: F-prior vs A vs F-rand-prior, all
   consuming only (scene). The (b) claim lives or dies here.
5. **Diversity**: at fixed pin, both bend modes present with reasonable spread across
   unpinned draws (existing metric). Also run an over-pinning control if diversity
   collapses: shrink S.
6. **Interpolation/geometry probe**: interpolate commanded phases (along the geodesic
   on the circle per bin) between two scenes' structures; realized trajectories should
   deform smoothly and monotonically (track a shape descriptor vs interpolation t; no
   mode jumps). Run the same interpolation through B-phase's conditioning input for
   contrast.
7. **Leakage check**: train a small decoder from pinned content (phases at S) alone →
   full chunk; report R². Near-perfect reconstruction ⇒ over-pinned (the appearance-
   leak analog); expect mid-range. Also confirm unpinned dims still drive style.

## Gates and what each outcome means

- **G1 frame recovery fails** → estimator/dataset bug; stop and fix.
- **G2 mechanism (F-oracle)**: wrong-structure err-to-command ≪ B-phase and in the
  ballpark of C-disp's separation. Fails ⇒ phase pins don't bind the way linear pins
  do at toy scale — investigate before any thought of LIBERO-scale phase pins.
- **G3 geometry (F-prior)**: ≥ A on held-out no-oracle success AND > F-rand-prior.
  This is the first evidence for motivation (b) anywhere in the project. If F-prior ≈
  A but F-oracle passes G2: the channel works but the coherence criterion captured the
  wrong content — iterate on the criterion (next candidate: scene↔action coherence,
  i.e. structure = the component of motion predictable from scene geometry), not on
  the mechanism.
- **G4 no collapse**: diversity preserved, leakage R² bounded.
- **All pass** ⇒ write a Phase 2 redesign proposal: replace (or precede) the VQ
  encoder with an offline coherence-learned frame at LIBERO scale — multi-demo
  per task/init as the "modalities," EE-canonical frame, temporal phase over the
  50-step chunk, prior p(phase | obs, text) as the Phase 2 prior head. The VQ route
  keeps discrete composability; the coherence route is the paper-faithful one. They
  can coexist; the toy result decides which leads.

## Sanity sequence (before the arm sweep)

1. Synthetic coherence check: plant a known (u, S) in generated fields, confirm the
   estimator recovers it exactly.
2. Numerical preservation check: for pinned ε̃ and action a, assert phase(x_t) at
   ω ∈ S is constant over t ∈ {0, .25, .5, .75, 1} and phase(v) matches up to sign.
3. Identity mode (S = ∅) training matches baseline loss curve bit-for-bit given the
   same seed (the parity discipline from Phase 1).
4. Overfit 10 scenes, then oracle-pin inference: shape adherence should be tight even
   this early (the overfit regime is where the pin is easiest to read).

## Implementation notes

- Same stack as the existing toy: autograd, CPU, 3×128 relu MLPs, ~10k Adam steps.
  Budget: dataset gen is trivial; each arm ≈ existing toy cost × modest factor for the
  larger scene set — the full battery should stay within a CPU-day.
- New code under `experiments/toy_frame/` (leave `experiments/toy/` untouched — it is
  cited by the status brief). Reuse toy_flow.py components by import or copy, don't
  fork silently: note provenance at the top of the file.
- Results: JSONs + a README.md in `experiments/toy_frame/` following the exact table
  style of `experiments/toy/README.md`, with a Reading section written for someone
  who has read only the status brief.
- Report circular quantities in degrees; state N, seeds, and tolerance choices in the
  README so the numbers are reproducible.
