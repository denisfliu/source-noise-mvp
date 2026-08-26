# NORTH STAR — read before doing ANYTHING else (added 2026-08-05, Denis + Claude)

**The hypothesis this project exists to test:** an action can be FACTORED into two
components — a *coarse* component carried in the flow model's input (source) noise, and the
residual detail produced by denoising. Coarse actions are hypothesized to be easy to predict;
given the coarse command, the robot figures out the rest through denoising. If true, this
buys (a) steerability and interpretability of the action head, and (b) — the generalization
target — competence on **low-data-regime tasks**, because the hard-to-learn part of the
action is supplied rather than learned. "Distilling the idea of movement" is this
factorization at work: a commandable movement vocabulary the system composes for new tasks.

**Non-negotiable on the command source: one-hot / keyword task encodings are unacceptable in
the long run** (Denis, repeatedly). They enumerate tasks instead of understanding them — no
paraphrase robustness (MiniLM/one-hot findings), no unseen-task generalization (−1.9 R² on
held-out LIBERO), and no composition, so the movement half of this north star is unreachable
through them. One-hot is permitted only as a debugging scaffold to isolate other components;
the destination command source is semantically grounded (VLM-grounded) language.

## RECORD BOARD (a REFERENCE for orientation, not a requirement — update when results land)

**Best known closed-loop config** (2026-08-05 evening, strict `gate_success`, corrected center
render, pending video confirmation): **39/40 — LEFT 9/10 · RIGHT 10/10 · CFL 10/10 · CFR 10/10**,
and it is the SIMPLEST system tried: **NO clock, no VLM in the command loop**:
- **RRR pin flow** `gate_both_pin_rrr/4999` + **no-progress prior** c = MLP([state, task-onehot])
  `rung3/noprog_prior_rrr4.pt` (builder `make_progress_prior4.py` NOPROG=1), server
  `serve_gate_pin_prog4.py` (auto-detects clockless priors).
- **The progress clock is RETIRED**: it caused the right settle gap (clock 1-3/10 vs clockless
  10/10) and its historical 0→60% unlock was compensation for the contaminated-label prior.
  A-priori duration knowledge is a hack (Denis); if future tasks need phase, use observational
  signals (milestone latching, action-magnitude stop, state history) — never a wall clock.
- **Controls**: scratch π0 transits 20/20 but completes 0/20; VLM-feature command maps 0/5
  everywhere even decontaminated (offline metrics excellent — the chasm is real). VLM features
  are for TASK SELECTION (they ground it, cos 0.92-1.0); state for geometry; pin for movement.
- **The right gate was a LABEL BUG, not an architecture gap**: the old binary labeler folded
  center-from-right demos into the "right" task; the mixture-trained prior flew the center
  route (0/10). Correct labels alone: 0/10 -> 10/10 transit. Beware the same bug elsewhere:
  `gate_ctx_common.load_eps` labels binarily — `local/gate_nav_aug` originals (aug flow
  training) and ALL VLM-line feature caches/axes inherit it; rebuild before reuse.
- **Scope caveat (why the VLM line exists):** LEFT scene works — 2/2 strict full-task success
  under the authoritative scorer (`rung3/gate_success.py` -> falsify posthoc + safety-YAML
  aperture/goal-box). RIGHT scene fails — **0/10 validated 2026-08-05**: the flights head to
  the correct gate (side selection isn't the failure there) but a consistent ~1 m +x aiming
  bias sends every one past the opening (closest approach 0.65-0.85 m). NOTE this config's
  one-hot command source is a SCAFFOLD only (see non-negotiable above) — the open problem is
  BOTH the right-gate aim AND replacing one-hot with a grounded command source; the VLM line
  is judged against that combined goal, not against the left-gate record.
- **2026-08-05 late, REVISED after video veto + clearance audit: the +x aim bias is
  COMMAND-SIDE, but the oracle runs were VETOED — every compound/right "success" clipped the
  gate** (min clearance 0.001-0.085 m vs demos' 0.28-0.38; cause: oracle aimed at safety-AABB
  mid-height z=1.0 while the physical opening demos transit is z~1.5). Transit-level results
  stand (routes correct, both gates latched); clean-tier executability under exact commands is
  UNPROVEN pending a demo-derived-aim rerun. **Scoring rule upgraded: strict success = transit
  judge + `gate_clearance.py` (gate-cloud distance, body 0.18 m) + human video** — the transit
  judge's aperture AABB spans the posts, not the hoop (region-box bug class). Record-board
  audit under the new rule: LEFT/RIGHT/CFL 10/10 clearance-clean; **CFR only 4/10 clean**
  (6 grazes 0.11-0.16 m) — CFR line downgraded. Factorial eliminations unaffected (momentum/
  splice/flow-variant/render all still excluded; splice verified to zero seam velocity). The
  oracle is a scaffold like one-hot: the open problem is learning its three ingredients (gate
  selection, carry-through aim, route topology) FROM DEMOS AND PERCEPTION ONLY. **Never train
  product models on SIM GROUND TRUTH (Denis, 2026-08-05): no oracle/waypoint labels, and no
  scene-YAML anchors either — the gate-locator (trained on YAML anchors) is a REPRESENTATION
  PROBE, not a deployable component.** Sim ground truth is for environment construction,
  scoring, and clearly-labeled diagnostics; deployable supervision must exist outside the sim
  (demos — e.g., gate location is inferable from where demo trajectories pass through — own
  observations, generic pretrained perception). Oracle distillates (`oracle_distill_prior.pt`,
  route learned 4/5, endgame blurred) are capacity diagnostics under the same rule.
- Other known flaws: crosses the gate off-center (faithful to off-center demos); descends at
  the end (no hold/stop — the aug flow `gate_aug_pin_rrr/4999` has G2-proven hover/stop
  vocabulary; combining them is an obvious open experiment).
- **FEATURE SOURCE OF THE COMMAND PATH IS A TEMPORARY SCAFFOLD (Denis, 2026-08-11).** A VLM
  feature cache is stamped to a checkpoint: fine-tuning the VLA moves the VLM as a side effect
  (18-22% of embedding dims beyond 3 sigma between two of our own LoRA checkpoints), so a prior
  fit under checkpoint A and served under B consumes a representation it never learned against.
  That bug cost the enumeration-free command source 0/10 closed-loop at offline c-R2 0.94;
  re-pairing it with its own checkpoint gave LEFT 10/10 clearance-clean with NO retraining.
  Interim rule: the command path reads a **frozen** encoder, and priors record `feat_ckpt`
  (`experiments/rung3/pin_basis.py`) which servers check. **The plan is to move to fine-tuned
  features via JOINT TRAINING** — command head inside the flow train loop, saved into the flow's
  checkpoint so the pairing cannot be wrong by construction; detached first, coupled (backprop
  into the VLM, i.e. making the representation predict `c` instead of hunting for a subspace that
  happens to be predictable) as the follow-up. Plan: `docs/command_source_design.md`. Ordering
  rule that holds either way: choose U from action statistics -> train the flow with oracle
  c = U^T a -> only then extract features and fit the prior.

## Operating rules

1. **Orient against the record board before continuing any line** — the newest log entries
   are the frontier of activity, not necessarily of results. The board is a reference point,
   not a gate.
2. **Success is checked by `falsify.safety.posthoc`** (directional gate-plane transit +
   compositional phase latching, geometry from the published scene YAMLs
   `falsify-pi/configs/scenes/*.yaml` `gate_region`) — never by ad-hoc scorers — and
   **confirmed by human video review**. Scalar metrics filter; they do not declare success.
3. **Whole-trajectory metrics only** for follow/execution claims — pinned-coordinate error
   is confounded by passthrough (toy_multicont, 2026-08-04).
4. **Statistics, two-tier (Denis, 2026-08-06):** exploratory screens run **5 seeds**
   (fast turnaround); record-board claims and final method comparisons require **≥10**
   (protocol noise ±5-6 pts). A 5-seed result is a lead, never a claim.
5. **Good code is the priority. Development time is irrelevant and is never a
   consideration or a talking point.**

# Project knowledge map — READ THIS FIRST (added 2026-08)

Source of truth for what we've learned and how to work on this box. The sections *below this map*
are the original Phase-1 onboarding (Jul 2026) and are partly superseded — treat them as historical
context for the toy + parity/overfit gates.

- **Findings index** — `experiments/FINDINGS_INDEX.md`: 109 one-line findings across every experiment
  (toy → robosuite → pi0/LIBERO → drone-gate), each with a pointer. **Grep this before re-deriving
  anything.**
- **Research log** — `docs/RESEARCH_LOG.md`: the dense chronological log; the `MEM:<n>` line pointers
  in the findings index resolve here. Newest work is at the bottom. Append to it (with an absolute
  date) when a result lands; add a one-line entry to the findings index with a pointer.
- **Experiment ledger** — `experiments/FACTORING_ARC.md`: long-form writeup of the pin construction +
  toy/Panda/cross-embodiment arc (§1–5).

## Working on this box
- 2× ~98 GB GPUs. `nvidia-smi` before launching; the box is sometimes shared — don't stomp other procs.
- Long jobs: `setsid bash script </dev/null >PERSISTENT_DIR/log 2>&1 & disown`, logging to a **persistent
  dir, not `/tmp`** (it gets cleaned). Verify the log grows before trusting the launch. Avoid a single
  ssh call that both backgrounds a job and then sleeps/echoes — the output gets swallowed; launch, then
  verify in a separate call.
- Envs: openpi (jax/pi0) at `~/code/openpi/.venv`; standalone gsplat renderer at `/tmp/tv`
  (torch 2.11 + cu128, gsplat 1.5.3). The falsify `.venv` is a **dead symlink** — don't use it; the
  render chain was re-ported into `/tmp/tv`.
- JAX grabs all visible GPUs: scope with `CUDA_VISIBLE_DEVICES`; CPU-only JAX = `JAX_PLATFORMS=cpu
  CUDA_VISIBLE_DEVICES=-1`. Serve pin with env-gated `patch_pi0_pin.py`; `policy.infer(obs, noise=...)`.

## Engineering preferences (Denis)
- Commit messages: **never** auto-add an agent name as co-author.
- Technical decisions: weight quality, simplicity, robustness, scalability, and long-term
  maintainability over development cost.
- Bug fixes: first **reproduce the bug end-to-end** as the end user would experience it.
- Fix things that are clearly off even when tangential; hold a high bar on test failures/flakiness.

---

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
