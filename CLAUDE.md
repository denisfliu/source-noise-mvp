# Project: source-noise action steering MVP

## What this is

Testing whether a geometric invariant carried in the *source noise* of a
flow-matching VLA action head is followed more reliably than the same
invariant fed as a conditioning input. Idea adapted from an ICLR'26 paper on
subspace phase-invariant diffusion sources for image re-rendering: move
control out of the conditioning branch and into the source distribution
q(x_T|c), so the denoising loss itself penalizes deviation from control.

Invariant (Phase 1): chunk displacement L(a) = sum_t a_t (linear in the
chunk, so L(x_t) = L(a_0) at every noise level and the flow target satisfies
L(v)=0 — control lives in the regression target).

## State of play (all verified before this handoff)

- `src/snmvp/` — noise calibration library. 9 unit tests pass (numpy;
  torch mirrors run when torch present). Core property: exact carried
  invariant under pi0's interpolant `x_t = t*noise + (1-t)*actions`.
- `experiments/toy/` — CPU toy validation DONE, mechanism confirmed:
  pinned-noise arm executes contradictory commands at ~1% error vs ~26x
  worse through a conditioning branch; diversity preserved. See its README.
- `patches/openpi_arm_c_training.patch` — applies cleanly to openpi commit
  `15a9616`; gates on env var `SNMVP_PIN_ALPHA` (0 = inert baseline path).
- Inference needs NO patch: `Policy.infer(obs, noise=...)` already threads
  caller noise; build it with `snmvp.openpi_adapter.make_calibrated_noise`.
- Full experiment plan + go/no-go gates: `docs/mvp_plan.md`.
  Integration details + sanity sequence: `docs/openpi_integration.md`.

## Hard constraints

- This EC2 box runs OTHER WORK WITH PRIORITY. Before launching anything on
  GPU: `nvidia-smi` — if either GPU has significant memory in use by
  processes that aren't ours, do not launch; report and wait for Denis.
- Touch nothing outside `~/code` (plus `~/.cache/openpi` which openpi uses
  for checkpoint downloads). No system packages, no sudo, venv-local only.
- Never mark a gate as passed on partial evidence. Failed gates stop the
  line: diagnose per the plan's mitigation table, or escalate to Denis.

## Task order

1. `bash scripts/setup_ec2.sh` (idempotent; clones openpi, uv env, tests).
2. `bash scripts/box_phase0.sh` (applies patch, cheap checks, Blackwell
   smoke test). Fix what breaks. Known likely issues + fixes are printed by
   the script and detailed in docs/openpi_integration.md (torch/Blackwell:
   install cu128+ torch in the venv only; patch drift: pin openpi to
   15a9616 or re-derive the ~10-line hook).
3. Parity gate: two short `pi0_libero` PyTorch runs, identical seed, one
   with SNMVP_PIN_ALPHA=0 vs unpatched baseline — logged losses must match
   exactly. Proves the patch is inert when off. (~50 steps is enough.)
4. Overfit probe: SNMVP_PIN_ALPHA=1.0, overfit ~10 LIBERO episodes a few
   hundred steps. Then wrong-invariant probe: serve the policy, command an
   invariant contradicting the scene via `make_calibrated_noise` +
   `policy.infer(example, noise=...)`. PRIMARY metric: error-to-command
   (realized chunk sum vs commanded, normalized units), not binary follow
   rate. Expected: the overfit model follows the noise. If it follows
   vision instead, the channel is dead — stop and diagnose (chunk-sum
   salience, normalization space mismatch, pinned_dims asymmetry).
5. Only after 3+4 pass: Phase 1 arms per docs/mvp_plan.md (A/B/C/D, 3 seeds,
   LIBERO-Spatial held-out placements). Arm B requires a small conditioning-
   token addition — design it minimally, keep parameter counts comparable,
   and write the diff to `patches/` before training with it.

## Known technical footguns

- Invariants must be in NORMALIZED action units (post q01/q99) everywhere.
  The oracle path: extract invariants from already-normalized chunks. Do not
  implement physical-unit conversion casually (H-dependent offset; see
  `snmvp/openpi_adapter.py` docstring).
- Training pin currently covers all 32 padded motor dims; inference adapter
  pins only the leading real dims. Before any headline run, restrict the
  training pin with `pinned_dims` to the actuated dims for symmetry.
- Deterministic Euler sampler only (openpi default ~10 steps). Any
  stochastic sampling would erode the pin.
- Record every run: config, seed, SNMVP_PIN_ALPHA, openpi commit, metrics →
  `experiments/phase1/results/` as JSON, mirroring experiments/toy format.

## Reporting

Denis is migrating decision-making here, but flag rather than decide:
GPU-memory-driven architecture compromises (LoRA / frozen trunk), any gate
failure, and anything requiring >a few hours of GPU time while other
workloads are present.
