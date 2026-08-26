# Source-Noise Action Steering — Reproduction & Reimplementation Guide

**Purpose.** This is the single authoritative document for understanding and
re-implementing the project from scratch. It is written so that someone with no
prior context (human or AI) can rebuild the method, the experiments, and the
results, and — just as importantly — avoid the dead ends we already ruled out.

It supersedes the forward-looking plans as the description of *what was actually
done*. Where `docs/mvp_plan.md` describes the original 4-arm plan (A/B/C/D ×
placement-success as the headline), note that the plan **changed** as findings
came in; this guide reflects the final design. Read this first, then use the
other files as depth references (map at the end).

---

## 1. The one-paragraph idea

Flow-matching policies (and diffusion policies) generate an action chunk by
starting from Gaussian noise `ε` and integrating a learned velocity field to a
clean action `a`. Normally that source noise is meaningless. **We write a
movement command into the noise** — specifically a low-dimensional *invariant*
of the action chunk (e.g. "where does this 50-step motion end up") — in a way
that the flow interpolant preserves at every noise level, so the *training loss
itself* penalizes any deviation from the command. This moves control out of a
conditioning branch (which the network is free to under-use) and into the
regression target (which the loss must fit). The idea is transplanted from an
ICLR'26 paper that does this for image re-rendering by pinning the Fourier
**phase** of a learned subspace of the source noise ("subspace phase-invariant
sources"); we adapt it to robot **actions**. Testbed: π0 (openpi) on LIBERO.

**Two motivating hopes** (worth stating because they shape what to measure):
(a) *generalization* — a policy grounded in a geometric command should transfer
to novel scenes/embodiments once the command-to-motion mapping is learned;
(b) *better learning* — grounding actions in geometry should improve sample
efficiency by handing the model the "where" instead of making it infer it.
A third, which became the strongest angle: **steerability** for interpretable
human-facing control (command the robot in physical units; it obeys).

---

## 2. The core method (enough math to reimplement)

### 2.1 The invariant

An action chunk is `a ∈ R^{H×D}` (H timesteps, D action dims; LIBERO/π0: H=50,
D=32 padded, 7 real dims = 3 translation + 3 rotation + 1 gripper). Actions are
per-step **deltas**. Define the chunk invariant as the per-dim sum:

    L(a) = Σ_{t=1..H} a[t]  ∈ R^D      ("net displacement of the chunk")

**L must be LINEAR in the action representation.** This is the single most
important design constraint — it is what makes the pin survive the flow
interpolant (below). Sums of deltas are linear; absolute end-poses through
forward kinematics are not. Only pin linear functionals.

### 2.2 The pin construction (source-noise overwrite)

Let `U ∈ R^{(H·D)×k}` be a fixed orthonormal basis for the "pinned subspace"
(for the displacement invariant, the per-dim constant/mean directions). Given
Gaussian `ε` and a target invariant `m`:

    ε̃ = ε + U (m̂ − Uᵀε)

i.e. overwrite the projection of the noise onto the pinned subspace with the
(normalized) command `m̂`, leaving the orthogonal complement untouched.

**Normalization (`m̂`).** z-normalize the invariant against dataset statistics
so its marginals ≈ N(0,1). This is the analog of the paper's Gaussian/Rayleigh
calibration: the pinned noise must look statistically like ordinary Gaussian
noise except for the information it carries, so the network can't trivially
detect-and-discount the special coordinates. (Exact magnitude overwrite in the
hybrid/phase variants does break the marginal of the pinned coordinate — an
accepted trade-off, see §4.2.)

### 2.3 Why the loss enforces it (the load-bearing property)

openpi's flow head uses interpolant `x_t = t·ε + (1−t)·a` and velocity target
`v = ε − a`. If we pin `L(ε̃) = L(a)`, then by linearity:

    L(x_t) = t·L(ε̃) + (1−t)·L(a) = L(a)   for ALL t
    L(v)   = L(ε̃) − L(a) = 0

So the invariant is carried, unchanged, to every noise level, and the velocity
target has zero invariant-component. Any predicted velocity `v̂` with `L(v̂)≠0`
is directly penalized by the flow MSE loss. Obedience is *optimized*, not hoped
for. (Contrast: a conditioning input leaves `v` independent of the command, so
the loss is silent on whether the command is followed — see §5, finding 6.)

### 2.4 Inference

At test time there is no `a` to pin from. Sample fresh `ε`, overwrite its
pinned subspace with the commanded `m̂`, integrate the sampler as usual (π0/π0.5
use a deterministic ~10-step Euler ODE — noise enters only at init, so the pin
is preserved without per-step recalibration; a stochastic sampler would require
recalibrating every injected noise). The command `m̂` must be in the SAME
normalized action space used in training (apply the run's `norm_stats.json`).
The command source is either an oracle or the learned prior (§6).

---

## 3. Testbed 1 — the mechanism toy (`experiments/toy/`)

**Goal:** prove the mechanism end-to-end, cheaply, before touching a VLA.

**Setup:** 2D point-robot reach. H=20 delta-action chunks, bimodal path style
(left/right bend). Invariant = chunk displacement (sum of deltas). Tiny
flow-matching MLPs (3×128 ReLU), ~10k Adam steps, CPU, `autograd` (no torch
needed). **Critical detail:** actions must be normalized to O(1) (`ACT_SCALE`);
with raw small deltas the flow model badly underfits (endpoint err ~0.5 → ~0.02
after scaling). The real pipeline's q01/q99 normalization plays this role.

**Arms:** A = no invariant (baseline); B = invariant as conditioning input;
C = invariant in source noise (the method); D = both.

**Results that matter:**
- **C follows contradictory commands ~26× tighter than B** (err-to-command
  0.027 vs 0.70). Both "follow" (follow rate 1.0), but C executes *precisely*;
  B executes *approximately*. → The separating metric is **error-to-command
  under a wrong-invariant probe**, not binary follow rate.
- Diversity survives the pin (both bend modes present at fixed invariant): the
  unpinned noise dims still drive style. No mode collapse.
- C also beat the *unconditioned* baseline A on endpoint error (0.018 vs 0.038)
  — early hint of hope (b), later found NOT to hold at scale (§5).

**The dropout finding (decisive design lesson).** Arms P/Q trained with the pin
applied to only 80% of samples (CFG-style), Q adding a "pin present" flag.
Both KILL the channel: err-to-command 90× worse, follow rate 0.0. Mechanism:
with dropout the model must learn the obs→action decode for the unpinned
samples; once learned, that decode is correct for ALL samples, so no gradient
pressure maintains the pin-read. **Consequences that constrain everything
downstream:**
1. The pin must be **always-on** in training. There is no free dual-mode
   (steerable + plain) checkpoint from one model.
2. Therefore inference ALWAYS needs a command source (oracle or learned prior).
3. The pin is read because it is the *cheaper* feature (a linear read off `x_t`)
   than deriving the answer from observation — not because obs is insufficient.
   This "gradient economics" explains later results (§5, §7).

---

## 4. Testbed 2 — the learned-frame toy (`experiments/toy_frame/`)

**Goal:** the original method uses a *hand-defined* invariant. The paper's real
contribution is that structure is **discovered** by a coherence criterion that
sits OUTSIDE the generation loss. This toy transplants that idea and tests hope
(b) honestly (no oracle at eval).

### 4.1 Discovering the frame (coherence, done offline BEFORE flow training)

Data: 2D point-robot with **obstacles**, N=8 demos per scene, with a *planted*
structure/style split (structure = endpoint + obstacle clearance + progress
timing, shared across a scene's demos; style = bend side, wiggle — demo-private)
so recovery is checkable. Canonicalize each chunk by the scene's target bearing.

For a projection direction `u` and temporal frequency bin `ω`, define coherence
as intra-scene phase concentration across demos:

    γ(u,ω)  = E_scenes | (1/N) Σ_demos exp(j·φ_demo(u,ω)) |          (mod-2π)
    γ₂(u,ω) = E_scenes | (1/N) Σ_demos exp(j·2·φ_demo(u,ω)) |        (mod-π)

γ ∈ [0,1]: 1 = all demos agree (structure); ~1/√N = demo-private (style).
The **γ₂ (angle-doubled) variant is essential** for bimodal styles: when a field
flips sign across demos (bend-left vs bend-right), plain γ self-cancels but γ₂
survives — and this matches the mechanism, since the flow target carries phase
only up to sign anyway.

**Selection rule (pre-register before training):** pin bin `(u,ω)` iff
coherence > 0.6 AND per-axis energy ≥ 1% (the **energy floor** — without it you
select ~15 "dust" bins with γ up to 0.997 at ~0% energy, which is meaningless).
For magnitude-pinnable bins additionally require magnitude CV < 0.15 (§4.2).

### 4.2 The critical finding: phase-only FAILS; you need magnitude too

**v1 (phase-only pins, faithful to the paper): G3 FAILED — F-prior scored
*below* baseline A.** Diagnosis: the paper pins phase only because in *images*
magnitude IS appearance (the thing to discard). In *control* that split is
wrong: safety-critical structure is partly **metric** — obstacle clearance is an
*amplitude*, not a shape. Phase-only pins "shape, not scale," so they cannot
carry clearance, and (dropout economics again) the always-on pin displaced the
obs→clearance learning while the amplitude slot stayed random → worse than not
pinning.

**v2 = "Option H" (hybrid pins): all gates PASS.** Pin the full complex
coefficient (phase AND magnitude) at magnitude-coherent bins, phase-only
elsewhere. The coherence criterion extends to magnitude via the CV estimator.
Also: fix the dataset generator first (v1 had an endpoint-biasing bump; ceiling
0.81 → 1.00) so the success ceiling is clean.

### 4.3 Results (100 held-out scenes × 8 rollouts × 3 seeds; bar ±4 pts)

| arm | no-oracle success | note |
|---|---|---|
| **F-prior** (learned frame, prior supplies pin, no oracle) | **62.3%** | the method |
| F-oracle (learned frame, ground-truth pin) | 64.0% | prior captures nearly all oracle value |
| A (no pin) | 45.1% | floor |
| F-rand-prior (same arch, RANDOM frame) | 44.1% | the control that matters |
| C-disp-oracle (hand-defined displacement pin) | 43.8% | ≈ A |

**Readings:**
1. Hope (b) supported here: F-prior beats A by +17.3 and beats F-rand-prior
   (same architecture/capacity, wrong frame) by +18.2. The win is the
   *coherence-learned geometric organization*, not "any pin + a planner."
2. The prior generalizes (captures 62.3 of 64.0 oracle value).
3. **Discovered structure beats the hand-defined invariant** (C-disp ≈ A):
   endpoint displacement wasn't the binding constraint; clearance amplitude was,
   and only the learned frame carried it.
4. G4: diversity preserved (0.487 on symmetric scenes; mod-π pins stay
   orientation-free), leakage R²=0.209 (up from v1's 0.08 but far from
   transcription — the pin carries a genuine summary, not the whole trajectory).

**Separation of powers (do not violate):** the coherence criterion that defines
structure must be INDEPENDENT of the flow loss. If the flow loss alone defines
what goes in the noise, the optimum is transcription (write the whole action in
→ perfect autoencoder, useless at inference). This is the action-space analog of
the paper's appearance leak. Leakage R² is the monitor.

---

## 5. Testbed 3 — real model (π0 on LIBERO, `experiments/phase1/`)

### 5.1 openpi integration (see `docs/openpi_integration.md` for exact sites)

- openpi commit `15a9616`, PyTorch path. `PI0Pytorch.forward` and
  `sample_actions`/`Policy.infer` already accept caller-supplied `noise`, and
  use exactly `x_t = t·noise + (1−t)·actions`, `u_t = noise − actions` — the
  convention the math assumes.
- **Two hooks, both ~20-line patches, arms selected purely by env vars:**
  - `patches/openpi_arm_c_training.patch` — training hook: sample noise
    explicitly, pin `extract_invariant(actions)` via the SourceConstructor, pass
    `noise=` to the model. `SNMVP_PIN_ALPHA` (0 = off/baseline, 1.0 = hard pin,
    fractional = soft-pin ablation), `SNMVP_PINNED_DIMS` (restrict to real
    actuated dims; use 7 for LIBERO).
  - `patches/openpi_arm_b_conditioning.patch` (applies on top) — inject the same
    z-normalized invariant into the trailing padding dims of the 32-dim proprio
    state, read through the existing `state_proj`. **Parameter count EXACTLY
    unchanged** vs C → the fairest possible B-vs-C comparison. `SNMVP_COND_STATS`.
- **Consistency requirement:** the invariant must be computed in NORMALIZED
  action space (post q01/q99) at both training and inference; a commanded
  invariant at inference must be normalized with the same `norm_stats.json`.
- **Parity discipline (do this before trusting anything):** under
  `torch.use_deterministic_algorithms`, a patched run with α=0 must produce
  bit-identical losses to the pristine baseline. We verified 60/60 bit-exact.
  (Baseline is NOT run-to-run deterministic without the shim; inject the shim
  into both arms identically.)

### 5.2 The final design (revised from the original 4-arm plan)

A (baseline, no channel) ×3 seeds; C (source-noise pin, α=1, 7 dims) ×3 seeds;
B (conditioning control) ×1 seed. D (both) dropped. All: `pi0_libero`, 15k
steps, cosine decay rescaled to 15k, batch 32, single GPU/run, checkpoints every
2500. B demoted to a *control* answering "is C's advantage the channel or the
extra information?" — not a headline arm.

### 5.3 Established checks

- **Phase 0 exit gate PASSED:** 30k baseline = 94.6% LIBERO-Spatial (50-trial,
  500 episodes) vs ~96% community ref. Infra/recipe/eval validated end-to-end.
- **Channel alive & calibrated:** C executes contradictory commands ~16× tighter
  than chance; oracle commands at ~2.3% of dataset scale; follow rate 16/16 at
  every checkpoint, seed-stable. Command→realized-EE-displacement is **affine,
  R²=0.87–0.97, ~3–4.7 mm per normalized unit** → "move 20 cm left" is a
  literally computable command.

### 5.4 The headline result — a coupling spectrum

Same invariant, same learned prior as command source, same data; the ONLY
variable across B/C is *where the signal enters*.

| arm | adherence (oracle / contradictory / negated; scale ~123) | follow rate | success (canonical / held-out) |
|---|---|---|---|
| A (no channel) | — | — | **90.3% / 89.7%** |
| B (conditioning) | 6.15 / **82.7** / **138.8** | 0.625 | 76% / 73% (1 seed) |
| C (source noise) | 3.07 / **7.82** / 9.99 | 1.000 | 49.7% / 53.3% |

**What it means:**
1. The source-noise channel binds **~11× tighter than conditioning** under
   contradiction, unboundedly tighter under negation. B's contradictory/negated
   error EXCEEDS its own plain control — conditioning doesn't weakly follow a
   bad command, it *actively follows the scene instead*.
2. **Both non-A channels capture the model** (command-less stock eval: B 0–4%,
   C 0%). "Conditioning is ignorable" is FALSE at 3B scale. The difference is
   coupling *strength*, not whether coupling happens. (This killed the project's
   original binary thesis and replaced it with the better spectrum framing.)
3. **Coupling strength = an obedience↔success operating point.** Loose (B):
   recovers most of A's success from a noisy prior because vision overrides bad
   commands, but is unsteerable against the scene. Tight (C): fully steerable,
   but pays a ~40-pt "obedience tax" faithfully executing the prior's ~25%
   errors. This frontier IS the Phase 1 result.
4. **No placement generalization gap for any arm** → LIBERO-Spatial placement
   variation does not discriminate; the plan's original H1 (C beats B/A on
   held-out *success*) is empirically void at this scale. The mechanism claim
   lives in **adherence/steering**, not success.

**Honest caveats:** B is n=1 (control). C's success ceiling is set by *prior
quality* (~25% err), NOT the channel — oracle-command adherence is 2.4% of
scale, so a better command source raises C's success directly. The tax is a
prior-quality statement, not a channel limit.

---

## 6. The learned invariant prior (the command source that works)

**Why it exists:** always-pinned C needs a command at every inference step.
Sim-state geometric oracles were tried (4 variants) and ALL scored 0/20 on task
success despite perfect far-field steering — because chunk displacement is the
wrong command language at **contact** (at grasp states the correct "invariant"
is the maneuver itself, hold/close/lift; the hard-binding pin faithfully
executes a wrong geometric command and destroys the grasp vision was
performing). Scripting around this erodes fairness. So: replace the oracle with
a learned prior — exactly the toy_frame F-prior configuration at scale.

**What it is:** a small CNN+MLP, `p(invariant | base image, wrist image, raw
state)`, ~3 MB, trains in minutes. Predicts the 7-dim invariant in the exact
normalized space the pin uses, so its output drops straight into the noise.
(`scripts/train_invariant_prior.py`, saved `invariant_prior.pt`.)

**How it's trained (labels are free):** for any demo frame, the next H=50
recorded actions ARE the label. Pipeline: take next-50 actions (first 7 dims) →
convert dims 0–5 to deltas w.r.t. current state, gripper absolute (replicating
the openpi transform) → normalize by `norm_stats.json` → **sum over 50 steps**
→ standardize by `invariant_stats.json`. Two CNN towers (base + wrist cam,
5 stride-2 convs → 128 each) + state MLP (8→64) → concat → 3-layer head → 7.
6000 steps, batch 96, AdamW lr 1e-3 cosine, Huber loss. **Validation splits by
EPISODE (5%), not by frame**, so val frames come from unseen demos (no leakage).
Final held-out MAE per dim `[7.9,11.4,7.6,3.4,1.0,4.9,13.8]` vs stds 34–50
(≈20–30% rel err).

**Result:** self-contained C (checkpoint + prior + calibrated noise, zero
sim-state reads) completes pick-and-place at ~50%, up from the command-less 0%.
Served via `serve_snmvp_policy.py --prior`.

**Known limitation:** sees images+state but NOT the instruction → predicts *the*
motion the scene typically implies. Fine on LIBERO (scene implies task); making
it language-conditioned is the Phase 2 upgrade.

---

## 7. Consolidated findings — what worked, what didn't, what's open

**Solid / reproduced:**
- The pin mechanism works at toy and 3B scale; steering is calibrated in mm.
- Source-noise binds far harder than conditioning (26× toy, ~11× real).
- Pin must be always-on (dropout kills it); inference needs a command source.
- Structure can be *discovered* (coherence) and discovered beats hand-defined.
- Pins must carry **magnitude**, not phase alone (control ≠ images).
- A learned prior is the working command source; makes C self-contained.

**Did NOT hold / negative results (report these, don't bury them):**
- Hope (b) at scale: pinning did NOT improve LIBERO success (−40 pts vs A). It
  helps only where perception is the bottleneck; LIBERO vision is near-ceiling,
  so the noisy command can only subtract. (It DID help in the toy, where the
  policy was the bottleneck.)
- Held-out placement success discriminates NOTHING at this scale — the original
  H1 metric is void. Report as a negative finding about the benchmark.
- Geometric sim-oracles fail at contact; chunk displacement is the wrong command
  language there. Pin authority is state-dependent (strong free-space, weak at
  contact).

**Open / next:**
- Steerability-focused experiments (deprioritize task success): a soft-pin α<1
  or subset-of-dims knob that traces the B↔C coupling frontier — the HRI dial
  for "how much authority the human has."
- prior-v2 to shrink the tax: temporal context, bigger net, confidence-gated α
  (drop pin authority where the prior is unsure — matches the contact finding),
  per-replan residual prediction.
- Phase 2 (learned composable movement codes) hard constraints, all forced by
  data: codes must be always-on with a prior at inference (dropout); must carry
  metric/magnitude content (toy_frame v1); should be time-localized so contact
  phases get their own code / less pin authority (oracle finding). Coherence-
  discovery stays live ("structure = coherent complex content"); VQ-on-residual
  is the layered alternative.
- Aim (a) (novel-scene/embodiment transfer) is untested — its natural home is
  where vision ISN'T already sufficient (few-shot, new embodiment), i.e. Phase
  2/3, not LIBERO-Spatial.

---

## 8. Non-obvious pitfalls (things that already cost us time)

1. **Normalize actions to O(1)** or the flow model underfits badly (toy).
2. **Only pin LINEAR functionals** of the action rep (delta sums, not FK poses),
   or the interpolant won't carry the invariant.
3. **Energy floor on coherence selection** — zero-energy bins show spurious
   near-1 coherence ("dust"). Pin only bins with real energy.
4. **Use the mod-π (angle-doubled) coherence** for any bimodal/sign-flipping
   style, else you cancel real structure.
5. **Pin magnitude, not just phase**, for anything metric/safety-critical.
6. **Coherence criterion must be external to the flow loss** (separation of
   powers) or you get transcription. Monitor leakage R².
7. **Always-on pin only** — no CFG-style dropout, no "pin present" flag.
8. **Match the prior's inputs to the serving client exactly** (same image spec,
   same raw state dims) — train on the raw dataset, not through the openpi
   transform stack, to avoid a silent train/serve mismatch.
9. **Validate the prior by EPISODE split**, not frame split.
10. **Parity-check the training patch** (α=0 bit-identical to baseline) before
    drawing any conclusion.
11. **Command-less eval of an always-pinned model is 0% by construction** — it
    is an artifact, not the arm's result. Always eval with a command source.

---

## 9. File map (in the archive / on the box `~/code/source-noise-mvp`)

- `docs/status_latest.md` — canonical running brief (findings + numbers).
- `docs/mvp_plan.md` — ORIGINAL plan (contains since-revised choices; read as
  history + rationale, not as the final design).
- `docs/openpi_integration.md` — exact integration sites & env vars.
- `docs/learned_frame_toy_plan.md` (+ `_plan_reply`, `decisions_2026-07-05`) —
  the coherence-discovery experiment design & decisions.
- `experiments/toy/README.md` — mechanism toy + dropout finding + results JSONs.
- `experiments/toy_frame/README.md` — discovery + hybrid-pin win; `*.py` = the
  coherence estimator / pin / flow / dataset / eval; `results/step1/` = frame
  recovery (γ heatmaps).
- `experiments/phase1/results/phase1_results.md` — the coupling-spectrum table.
- `experiments/phase1/results/oracle_iterations_summary.md` — why oracles were
  replaced by the prior.
- `experiments/phase1/results/*.json`, `evals/` — raw probe & success evidence.
- `experiments/phase1/invariant_prior.pt` + `scripts/train_invariant_prior.py`,
  `serve_snmvp_policy.py`, `wrong_invariant_probe.py`, `libero_eval_client.py`,
  `calibrate_invariant_map.py`, `compute_invariant_stats.py`.
- `patches/` — the two openpi patches (arms C and B).
- `unpaired_rerendering_subspace.pdf` — the source paper.

## 10. Minimal reproduction order

1. Read the paper + this guide + `status_latest.md`.
2. Toy (`experiments/toy/`): `pip install autograd`; reproduce the 26× gap and
   the dropout kill. Confirms the mechanism and the always-on constraint.
3. Toy-frame (`experiments/toy_frame/`): reproduce frame recovery (step1), then
   the hybrid-pin +17-pt no-oracle win. Confirms discovery + magnitude lesson.
4. Real model: openpi @ `15a9616`, apply both patches; parity-check (α=0
   bit-exact); Phase 0 gate (baseline ≈ ref); train A×3/C×3/B×1 at 15k;
   train the invariant prior; probe adherence (`wrong_invariant_probe.py`) and
   eval success with the prior (`serve_snmvp_policy.py --prior`). Reproduces the
   coupling spectrum.

---

## 11. Forward direction — cross-embodiment transfer (planned)

Aim (a) / H3 is untested and its natural home is where vision ISN'T already
sufficient (few-shot, new embodiment). The design is in
`docs/cross_embodiment_plan.md`. One-paragraph summary:

Split the VLA at one seam into a FROZEN, embodiment-shared front-half (VL trunk
+ an invariant readout head on its frozen features → the geometric goal, as a
language-conditioned readout) and a LEARNED, embodiment-specific executor (the
flow head that realizes the pinned invariant as this body's actions). Adapting
to a new body re-learns only the executor from few demos; the invariant is the
API contract, pinned into the executor's source noise. The invariant must live
in an embodiment-agnostic, object/goal-centric, scale-normalized frame — and
rather than hand-assuming that frame is universal, use **cross-embodiment
coherence** (the toy_frame estimator with "modalities" → "embodiments") to
*discover and measure* how much structure a set of bodies shares; that number
predicts where transfer will work. Iterate cheaply by caching the frozen
front-half's invariants once, then training only the small executor. Embodiment
ladder by divergence: 2D multi-morphology toy including a holonomic point robot
as a drone analog (CPU, planted) → small diffusion-policy + robosuite arms /
gsplat perception (one GPU) → OXE + VLA confirmation. Note: OXE's need for a
unified action space IS the shared-frame problem, so the frame doubles as a
principled unified action representation.

**Rung 1 RESULT (2026-07-17, `experiments/toy_embodiment/`, see
`findings/toy_embodiment_README.md`):** G-frame PASS (cross-body coherence
recovers the shared frame; arms cohere more with each other than with the point
robot) and **G-transfer PASS** — freeze the coherence-learned frame + prior on
set A {arm2,arm3,arm4}, adapt only the executor on a held-out body's few demos,
and T beats scratch/conditioning/random-frame on BOTH the same-family arm4 and
the maximally-divergent point robot (pooled, low n). First positive evidence for
aim (a)/H3. Success is prior-limited (T-oracle >> T), as at LIBERO scale.
G-predict did NOT hold with n=2 (lower-coherence body had higher transfer gain —
reversed); needs a body-ladder. Design choice: task-space actions for all bodies
(invariant linear, pin exact); embodiment = reach/feasibility. Next: Rung 2 at
real-perception/small-model scale — spec in `docs/rung2_plan.md`.
