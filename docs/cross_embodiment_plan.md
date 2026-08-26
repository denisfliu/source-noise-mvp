# Cross-Embodiment Transfer via a Shared Invariant Frame — plan

**Written 2026-07-17 with Denis.** Forward-looking design doc (not yet executed).
Tests project aim (a) / hypothesis H3: a geometric command frame learned on a
diverse corpus transfers to a new embodiment from limited data, with the demos
*calibrating the invariant→actuation mapping* rather than teaching the movement.
Read `REPRODUCTION_GUIDE.md` first for the method and prior findings this builds on.

Design constraint that motivates the whole plan: the target embodiment has far
less compute for iteration than the π0/LIBERO setup, so the method must be
iterable with **small models**, reserving the large VLA for final confirmation.

---

## 1. The architecture: a VLA split at one seam

The VLA is decomposed into three parts; the freeze boundary is the point of the
whole design:

1. **Front-half — perception + intent (FROZEN, shared across embodiments):**
   the VL trunk plus a small **invariant readout head** on its (frozen)
   features. Input: images + instruction. Output: the geometric goal for the
   next motion chunk — the *invariant*, in an embodiment-agnostic coordinate.
   Making the readout a head on the frozen VL features (rather than a separate
   vision net as in the D6 prior) is near-free and gives it language
   conditioning for free — fixing the D6 prior's "never sees the instruction"
   limitation.
2. **The invariant — the API contract** between the halves. Pinned into the
   executor's SOURCE NOISE (not fed as a conditioning input — that is arm B,
   which follows ~11x more sloppily and can't be steered against the scene).
3. **Executor (LEARNED, embodiment-specific):** the flow-matching action expert
   that realizes the invariant as *this body's* actions. This is the ONLY part
   re-learned on a new embodiment.

Cross-embodiment adaptation = freeze parts 1–2, re-learn part 3 from few demos.

**Iteration unlock:** because the front-half is frozen, precompute and cache its
invariants over the whole dataset ONCE; then executor training is
"cached invariant + noise → actions" — a tiny model, no large forward pass,
many design iterations per day on modest hardware.

---

## 2. The invariant frame (the load-bearing design choice)

**Express the invariant in an embodiment-agnostic, task-relative, scale-
normalized coordinate.** Prefer **object/goal-centric** motion ("the manipulated
entity moves from here to there in world frame") over EE-centric motion.
Rationale: what is shared between an arm and a drone both delivering a package is
the *package's* trajectory, not any robot's EE or joints. EE-displacement is the
manipulator-only special case. Scale-normalize (e.g. by workspace extent or
object scale) so "reach the target" yields comparable invariants across bodies
of different size.

**Do not hand-assume universality — measure it (cross-embodiment coherence).**
Reuse the learned-frame coherence estimator (`experiments/toy_frame/`) with
"modalities" replaced by "embodiments":

    structure = the motion component that SYNCHRONIZES across different
    embodiments executing the same task; embodiment-specific actuation is the
    private residual ("appearance").

This returns a number: the size/energy of the shared subspace. Large for two
arms, small for arm-vs-drone. That number (a) defines the transferable invariant
without hand-picking it, and (b) *predicts* where transfer will work. So the
workspace-divergence worry becomes a reported quantity, not an assumption.

**Two distinctions to keep clean in analysis:**
- Shared *language* ≠ universally *reachable*: a common invariant frame does not
  mean every invariant is executable by every body (a drone can't wrist-twist).
  The executor realizes what its body can; coherence/transfer numbers expose
  where a body structurally can't follow. Report these separately.
- Carry forward the phase+magnitude lesson (toy_frame v1→v2): pins must carry
  metric/magnitude content, not shape/phase alone. Cross-embodiment coherence
  must select complex coefficients (CV-gated magnitude), same as Option H.

**Relation to OXE:** OXE is only trainable after mapping heterogeneous action
spaces into a unified representation — which IS this embodiment-agnostic
invariant frame. The shared-frame requirement is not extra work for transfer;
it is what multi-embodiment training already forces. (Caveat: OXE is almost all
manipulators → medium divergence, not the arm/drone extreme; and it is painful
to iterate on. OXE is the confirmation corpus, not the iteration loop.)

---

## 3. Embodiment ladder (by divergence; small → large)

**Rung 1 — 2D multi-morphology toy (CPU, planted ground truth, minutes/arm).**
Extend `experiments/toy_frame/` to `experiments/toy_embodiment/`. Same
reach-with-obstacle task, multiple bodies:
- 2-link, 3-link, 4-link planar arms (jointed manipulators: constrained
  workspace, joint limits, joint-space actions).
- A holonomic **point robot** (moves freely in the plane, no joint structure) —
  the **drone analog**, deliberately included to test the arm-vs-drone
  divergence in miniature and cheaply.
Shared invariant = object/goal-centric task-frame trajectory of the controlled
point. Planted structure/style split as in toy_frame so recovery is checkable.
This rung answers the core question — does the factorization hold, how few B
demos are needed, and does coherence correctly predict which body pairs transfer.

**Rung 2 — small real-perception policy + sim (one GPU, hours).**
Diffusion-Policy-scale denoiser (~tens of M params; the pin transplants cleanly
— model-agnostic across DDPM and flow matching). Multiple real arm morphologies
on identical tasks via **robosuite** (Panda / Sawyer / IIWA / UR5e / Jaco).
Frozen small vision encoder (DINOv2-S / ResNet18 / frozen CLIP-ViT-S). Use the
**gsplat** sim here for photorealistic observations and — if it supports more
than one embodiment — as the new-body host; it also naturally produces the
unpaired, shared-geometry/different-appearance data the coherence criterion
consumes. First pass: privileged low-dim state (object/EE poses) to keep the loop
fast (oracle-first discipline); add gsplat vision only once the mechanism holds.

**Rung 3 — OXE confirmation (expensive, few runs).**
OXE manipulators as the diverse corpus; freeze the front-half; few-shot adapt the
executor to a held-out arm. On the real VLA this is the "freeze VLM trunk, train
action expert only" path — cheaper than full training, which is what makes the
low-compute target embodiment feasible for a handful of runs.

---

## 4. Protocol (per rung) and pre-registered comparisons

1. Train the front-half (+ coherence-discovered invariant frame) and executors
   on an embodiment SET A (>=2 bodies, to make coherence meaningful).
2. Report cross-embodiment coherence over A (the shared-subspace size) — this is
   a result in itself and predicts step 4.
3. Freeze the front-half. On embodiment B (held out), adapt ONLY the executor
   from few demos (sweep demo count: e.g. 5/10/25/50).
4. **Headline comparison, held-out B scenes, no oracle:**
   - **T (transfer):** frozen shared front-half + B-adapted executor (the method).
   - **S (scratch):** B policy trained from scratch on the same few demos.
   - **Cond:** same invariant as a conditioning input, not pinned (the arm-B
     control at this scale).
   The (a) claim is carried by the PAIR: T > S (transfer helps) AND
   T > Cond (the pin channel, not just the extra information, is what helps) —
   mirroring the toy_frame F-prior/A/F-rand pre-registered pair.
5. Diagnostics: does coherence-over-A predict the T−S gap across body pairs?
   (High shared coherence → large transfer gain; low → small.) Diversity +
   leakage monitored as in toy_frame.

---

## 5. Gates

- **G-frame:** cross-embodiment coherence recovers a non-trivial shared subspace
  for same-family bodies (arms), and a correctly SMALLER one for the point-robot
  vs arms — i.e. the estimator measures divergence sensibly. (Rung 1.)
- **G-transfer:** T > S and T > Cond on held-out B scenes at low demo counts,
  all seeds, pre-registered bar. (Each rung.)
- **G-predict:** the coherence number monotonically predicts the transfer gain
  across body pairs. If it does, coherence is a usable a-priori transfer
  predictor — a strong standalone contribution.
- **G-scale:** the effect survives Rung 2 (real perception, real arm morphology)
  before any OXE/VLA spend.

---

## 6. Open decisions (need Denis / target specifics)

1. **What is the ultimate target embodiment B**, and does it differ from the π0
   embodiment in action space/DOF (arm→humanoid/drone) or mainly kinematics? If
   radically different, the object/goal-centric invariant is mandatory and the
   point-robot rung is the relevant proxy; if same-family, EE-displacement
   suffices and Rung 2 (robosuite arms) is the closer proxy.
2. **What does the gsplat sim provide** — a physics-controllable robot with
   gsplat rendering on top (can generate its own demos), or rendering of
   reconstructed scenes only (needs an external physics sim underneath)? Decides
   whether Rung 2 self-generates data or borrows robosuite physics.
3. **Framing vs hierarchical control** (pre-empt the reviewer): the low level is
   a *generative* flow policy (multimodal, contact-rich, body-appropriate), the
   interface is a *discovered* geometric invariant (coherence), and — via the
   always-on-pin finding — geometry is baked into the generative process, not
   tracked after the fact. State this deliberately.

---

## 7. Why this is the compute-constrained iteration answer

The expensive object (the VLA front-half) is trained/used once and cached; every
method iteration trains only a small executor on cached invariants. The
embodiment ladder starts on CPU with planted ground truth and only reaches the
VLA at the confirmation rung. So the slow, low-compute target embodiment costs a
handful of executor-adaptation runs, never a full retrain — and the design loop
that precedes it runs entirely on small models.
