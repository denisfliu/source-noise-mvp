# Rung 2 — cross-embodiment transfer at real-perception / small-model scale

Written 2026-07-18. Forward-looking spec (not yet executed). Scales Rung 1's
positive toy result (`findings/toy_embodiment_README.md`: G-frame + G-transfer
PASS) to real perception and real arm morphologies, using a small model so it
stays iterable off the low-compute target embodiment. Read
`docs/cross_embodiment_plan.md` (design) and the Rung 1 README first.

## What Rung 1 established (and what it left open)

- The factorization works: freeze a coherence-learned, embodiment-shared frame +
  prior; adapt only the executor on a new body's few demos; beat scratch,
  conditioning, and random-frame. Held on a maximally-divergent point robot.
- Success is prior-limited, not channel-limited (T-oracle >> T).
- OPEN: "coherence predicts transfer gain" did not hold with n=2 (even reversed) —
  needs a real body-ladder. And the whole thing must survive real perception and a
  real (not leading-order-kinematic) embodiment gap.

## Pre-step (cheap, do first, still CPU/toy): tighten S_A + add a body-ladder

1. **Tighten the frame gating.** Rung 1's S_A came out to 8 pins because the raw
   energy floor let the lateral detour spread across bins. Before Rung 2, re-freeze
   with (a) a per-axis energy-share floor that keeps only the top bins covering
   ~90% energy, and (b) stricter CV for magnitude pins. Target ~3-5 pins, matching
   `toy_frame`. Re-run the transfer battery once to confirm the result survives a
   tighter frame (it should strengthen T-rand separation).
2. **Body-ladder for G-predict.** Add intermediate bodies spanning the coherence
   axis (e.g. arms of graded reach + a "geared" point robot with anisotropic gains)
   so G-predict has >=4-5 bodies, not 2. Re-measure whether transfer gain tracks
   (or inversely tracks) coherence. This is the cleanest cheap test of the Rung 1
   reversal hypothesis and it stays on CPU.

## Architecture (unchanged from the plan; recap)

Frozen embodiment-shared front-half (perception -> object/goal-centric invariant)
+ per-embodiment executor (invariant pinned into its source noise -> this body's
actions). Only the executor is re-learned per body. **Cache the frozen front-half's
invariants over the dataset once** -> executor training is tiny and fast (the
low-compute-iteration unlock).

## Setup

- **Policy:** a Diffusion-Policy-scale denoiser (~tens of M params; the pin
  transplants cleanly — model-agnostic across DDPM and flow matching). Two orders
  of magnitude smaller than pi0; trains in ~tens of min on one GPU. GPU 1 on the
  box is free (GPU 0 runs the cosmos job).
- **Perception:** frozen small vision encoder (DINOv2-S / ResNet18 / frozen
  CLIP-ViT-S) on renders; train only the action head. First pass: privileged
  low-dim state (object/EE poses) to keep the loop fast (oracle-first discipline),
  add vision once the mechanism holds.
- **Embodiments:** real arm morphologies on identical tasks via **robosuite**
  (Panda / Sawyer / IIWA / UR5e / Jaco). Training set A = 3-4 arms; held-out B =
  the remaining arm(s), spanning the coherence axis for G-predict.
- **gsplat:** photorealistic observations, and — if it hosts >1 embodiment — the
  new-body host; it also naturally produces unpaired shared-geometry/different-
  appearance data for the coherence criterion. Slots in at the perception stage.
- **Invariant:** object/goal-centric EE-displacement in the task frame (linear in
  an EE-delta action space -> pin exact). Discovered by cross-embodiment coherence
  over set A (structure = what synchronizes across arms doing the same task).

## Protocol

1. Coherence over set-A arms -> shared frame S_A (tightened gating). Report
   c(B, setA) per held-out body.
2. Freeze S_A + train the scene->invariant prior on set A (readout on frozen
   perception features; language-conditioned if instructions vary). Cache invariants.
3. Freeze front-half. Adapt only the executor on held-out body B's few demos
   (sweep n). 
4. Eval on held-out tasks/scenes, no oracle: T (frozen front-half + adapted
   executor) vs S (scratch) vs Cond (conditioned) vs T-rand (random frame). Plus
   T-oracle upper bound.

## Gates

- **G-frame:** coherence recovers a non-trivial shared EE frame over set-A arms;
  c ranks morphology divergence sensibly.
- **G-transfer:** T > S, Cond, T-rand on held-out B, pooled, at low n. (The Rung 1
  gate, now at real-perception scale = G-scale.)
- **G-predict:** with the body-ladder, does transfer gain track coherence (or
  inversely)? Resolve the Rung 1 reversal.
- **G-scale:** the effect survives real perception (frozen encoder on gsplat/robosuite
  renders) before any OXE/VLA spend.

## Open decisions (need Denis / target specifics)

1. Ultimate target embodiment B — differs from pi0's arm in action space/DOF
   (humanoid/drone) or mainly kinematics? Sets whether the point-robot-style
   divergence or the robosuite-arm divergence is the real proxy.
2. gsplat capability — physics-controllable robot + rendering (self-generates
   demos) or rendering only (needs robosuite physics underneath)?
3. Whether to make the invariant readout language-conditioned now (fixes the
   D6 prior's "no instruction" limitation) or keep scene-only for the first pass.

## Then Rung 3

OXE (manipulators) as the diverse corpus + few-shot executor adaptation to a
held-out arm; on the real VLA this is "freeze VLM trunk, train action expert only"
— the cheap path that makes the low-compute target embodiment feasible.
