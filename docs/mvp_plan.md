# Source-noise action steering: MVP experiment plan

Staged demonstration that geometric/movement invariants carried in the *source noise* of a flow-matching action head are followed more reliably — and generalize better — than the same invariants fed through a conditioning input. Phase 1 validates the mechanism with a hand-defined invariant; Phase 2 replaces it with jointly learned movement modes; Phase 3 (preview) applies the result to the low-data embodiment-adaptation problem.

Adapted from the ICLR '26 submission on subspace phase-invariant sources for image re-rendering. Core idea transplanted: move the control signal out of the conditioning branch and into the source distribution q(x_T | c), so the denoising loss itself penalizes deviation from control.

---

## Hypotheses

- **H1 (mechanism):** Pinning an invariant into the flow-matching source noise yields (a) higher adherence to the invariant and (b) better success on held-out object placements than feeding the identical invariant as a conditioning input, at equal data and parameters.
- **H2 (discovery):** A hard-bottlenecked segment encoder + autoregressive prior, trained jointly with the flow loss through the noise construction, discovers reusable movement modes; editing the code sequence steers execution.
- **H3 (adaptation, out of MVP scope):** Modes learned on a diverse corpus transfer to a new embodiment from limited single-task data, with the demos serving to calibrate the code→actuation mapping rather than to teach movement.

## Stack

| Component | Choice | Notes |
|---|---|---|
| Base model | π0 via openpi (`pi0_libero` config) for iteration; π0.5 (`pi05_libero`) for confirmation runs | Both have flow-matching action heads — SourceConstructor is identical, only the config changes. π0.5's LIBERO baseline is near-ceiling, which hides arm differences; π0 leaves headroom. π0-FAST is autoregressive (no flow head) and incompatible with the method. |
| Training path | openpi PyTorch branch | Noise sampling easier to intercept than JAX; validated on LIBERO for finetune + inference |
| Benchmark | LIBERO (Spatial for Phase 1; 90-task suite for Phase 2) | Spatial varies object placements — directly tests placement generalization |
| Compute floor | Full finetune >70 GB/GPU (A100-80GB/H100); LoRA >22.5 GB (JAX path only for now) | From openpi README; verify current support before committing |

---

## Phase 0 — Infrastructure (~1–2 weeks)

1. Reproduce the π0 LIBERO fine-tune and eval numbers on the PyTorch path (reference against community-reported π0-LIBERO results; the `pi05_libero` checkpoint is the sanity upper bound). This is the go/no-go for the whole plan: if the baseline doesn't reproduce, nothing downstream is interpretable.
2. Locate where the flow head samples ε and where the interpolant x_t = t·ε + (1−t)·a₀ and target are formed. Wrap noise sampling behind a `SourceConstructor` interface: input (ε, invariant m) → calibrated ε̃. Identity by default.
3. Instrument eval: per-episode logging of EE trajectories, initial object poses, and success, so adherence metrics are computable offline.
4. Build the invariant extractor for training data: for each action chunk, compute the pinned quantity (below) from the demo.

**Exit criteria:** baseline LIBERO-Spatial success within ~2 points of reference; noise hook verified by checking identity-mode training matches baseline loss curves.

---

## Phase 1 — Hand-defined invariant (mechanism test, ~3–5 weeks)

### Invariant definition

Chunk-level EE displacement: the summed action deltas over the chunk, L(a) = Σ_t δ_t ∈ R^6 (+ terminal gripper state). Rationale:

- LIBERO/π0 actions are per-step EE deltas, so the chunk displacement is a **linear** functional of the chunk — required for the invariant to survive the interpolant.
- With current EE pose known, displacement pins the chunk's endpoint pose: the "where does this motion end up" quantity.
- Linearity gives the carried-target property for free: set L(ε̃) = L(a₀); then L(x_t) = t·L(ε̃) + (1−t)·L(a₀) = L(a₀) for all t, and the flow target v = ε̃ − a₀ satisfies L(v) = 0. Any predicted velocity with L(v̂) ≠ 0 is directly penalized, and the implied denoised action always satisfies the pin.

### Noise construction

ε̃ = ε + U(m̂ − U⊤ε), where U ∈ R^{(H·d)×k} is a fixed orthonormal basis for the pinned subspace (k = 7), and m̂ is the invariant **z-normalized against dataset statistics** so its marginals match N(0,1). Statistics matching is the analog of the paper's Rayleigh-magnitude preservation: the calibrated noise should be indistinguishable from Gaussian except in the information it carries, so the network can't trivially detect and discount the special coordinates. Orthogonal complement untouched.

### Arms

| Arm | Invariant path | Purpose |
|---|---|---|
| A | none (baseline fine-tune) | floor |
| B | conditioning input (extra proprio-style token) | branch-carried control |
| C | source noise (construction above) | target-carried control |
| D | both | check for interference/synergy |

Same data, same steps, same seeds (≥3 per arm). At eval, the invariant comes from a **sim-state oracle** (ground-truth object pose → desired displacement via scripted geometry). Oracle first, deliberately: it isolates the mechanism from perception quality. A learned invariant predictor is a Phase 3 concern.

### Metrics

1. **Task success on held-out placements** — LIBERO-Spatial with initial-state seeds excluded from training data. Primary H1 metric.
2. **Adherence error** — |realized chunk displacement − commanded invariant|, distribution over episodes.
3. **Wrong-invariant probe** — command a displacement contradicting the scene (e.g., toward an empty region). Primary metric: **error-to-command** (how precisely the contradictory command is executed), with binary follow rate as a secondary check. Toy-scale validation (experiments/toy) showed both arms can "follow" a contradictory command, but the source-noise arm executes it ~26x more precisely — adherence precision, not follow rate, is where target-carried control separates from branch-carried.
4. **Residual diversity** — at fixed invariant and observation, trajectory spread across noise draws (unpinned dims). Confirms the model resamples style rather than memorizing; collapse to a single trajectory means the pin leaked into everything.
5. **Sampler check** — π0.5 uses a deterministic ODE sampler (~10 Euler steps), which preserves the invariant; verify no stochastic-sampler code path injects uncalibrated noise mid-rollout.

### Go/no-go gate

Proceed to Phase 2 iff: C > B on held-out placement success (statistically, across seeds) **and** wrong-invariant follow rate for C ≥ ~80% while B's is materially lower. If C ≈ B, diagnose before abandoning: chunk length (H=50 may dilute a 7-dim pin — try pinning per-sub-chunk), normalization scheme, timestep sampling distribution (invariant salience varies with t-weighting).

### Ablations (run only after the gate passes)

- Pin dimensionality k (endpoint only vs +2 via-points; watch H1 metric vs diversity trade-off)
- Normalization: z-norm vs quantile-norm vs none (none is the leakage-detector control)
- Pin strength: hard overwrite vs convex blend α·m̂ + (1−α)·U⊤ε (a soft-pin knob analogous to phase-mixing)

---

## Phase 2 — Learned movement modes (~5–8 weeks)

### Architecture additions

- **Segmenter:** fixed-length sub-chunks to start (H_seg ≈ 10 steps); keyframe-based segmentation as an ablation.
- **Encoder g:** small transformer, input = action sub-chunk (optionally + proprio), output = VQ code. Codebook K = 64, code dim 8 as starting point. The bottleneck is the load-bearing design element: wide enough for "reach-left-ish, slow," too narrow to transcribe the trajectory.
- **Writer:** code embedding written into a fixed orthonormal subspace of the sub-chunk's noise, statistics-normalized as in Phase 1.
- **Prior head p(code_t | obs, prompt, codes_<t>):** autoregressive over the chunk's code sequence; supplies codes at test time (sample = diversity, argmax = canonical, override = steering).

### Losses

Flow-matching loss (gradients flow through the differentiable noise construction into g) + VQ commitment/codebook loss + prior cross-entropy. Single joint training run on the LIBERO-90 multi-task suite — mode diversity requires task diversity; do **not** attempt discovery on a single task.

### Collapse and leakage diagnostics (monitored throughout training)

- Codebook perplexity (collapse → perplexity ~1)
- Reconstruction-from-code-alone probe: train a small decoder from code → sub-chunk offline; if reconstruction is near-perfect, the bottleneck is too wide (leakage)
- Code-usage entropy per task: codes should be *reused across* tasks; a code that fires for exactly one task is a task ID, not a movement mode

### Evaluations

1. **Wrong-code probe** (the Phase 1 probe, at code level): condition on a code contradicting the prior's prediction; measure whether execution follows the code. If it follows the observation instead, the channel died and steering is fiction.
2. **Steering demo:** on a successful task, edit one code in the sequence (e.g., swap an approach-direction code) and show the execution changes accordingly while the task still succeeds where geometrically possible.
3. **Recombination:** compose code sequences never observed for a given task; measure execution fidelity to the composed sequence.
4. **No-regression check:** prior-argmax rollouts should match or exceed Phase 1 arm C on standard LIBERO success.

### Go/no-go gate

Wrong-code follow rate ≥ ~70%, codebook perplexity ≥ ~K/4, and at least one clean steering demo. Deliverable: a short internal writeup + rollout videos of code-edited executions.

---

## Phase 3 — Low-data embodiment adaptation (preview, not in MVP)

Sketch, to be planned properly after Phase 2: freeze g and the codebook from the LIBERO-90 run; fine-tune the denoiser on a held-out task family (sim proxy for "new embodiment") with few demos, relabeled by the frozen encoder; test whether code-steered movements outside the demo distribution execute correctly. If yes in sim, port to the real TRI dataset — which is the actual target, and where the open question of a perception-based invariant predictor (replacing the sim oracle) must be solved.

---

## Risks

| Risk | Symptom | Mitigation |
|---|---|---|
| Denoiser ignores pinned dims | Wrong-invariant probe follows vision | Increase pin salience: check t-sampling weighting; stats-normalize so pin isn't suppressed by normalization; shorten chunk |
| Encoder leaks full trajectory (Phase 2) | Near-zero flow loss + failed prior; reconstruction probe near-perfect | Shrink codebook/code dim; add code dropout |
| Codebook collapse | Perplexity ~1 | Standard VQ remedies: EMA codebook, restarts, lower commitment weight |
| Nonlinearity breaks the carried invariant | Adherence fine at t≈0, poor at rollout | Only pin quantities linear in the action representation (delta sums, not absolute poses through FK) |
| Time misalignment for via-points | Adherence good for endpoint, poor for intermediate pins | Endpoint-only in Phase 1; progress-normalized segmentation in Phase 2 |
| openpi PyTorch path missing a needed feature | e.g., no LoRA → full finetune only | Budget A100-80GB/H100 nodes; confirm current feature matrix at kickoff |
| Baseline doesn't reproduce | Phase 0 exit criteria fail | Stop; fix infra before any conclusion is drawn |

## Rough resourcing

Phase 1 is ~12 training runs (4 arms × 3 seeds) of a LIBERO fine-tune; at full fine-tune scale this wants a small pool of 80GB GPUs for ~2–3 weeks of queue time. Phase 2 is one large joint run plus ablations on LIBERO-90 — comparable or larger. Oracle invariants, probes, and metrics are eval-side and cheap.

## Immediate next steps

1. Confirm openpi PyTorch feature status and GPU allocation.
2. Phase 0 items 1–2 (baseline repro; noise hook).
3. Decide the exact held-out placement split for LIBERO-Spatial before any training, and freeze it.
