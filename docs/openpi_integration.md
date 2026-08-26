# Integrating SourceConstructor into openpi (PyTorch path)

Verified against openpi commit `15a9616` (July 2026). A ready-made diff is at
`patches/openpi_arm_c_training.patch` (applies cleanly to that commit; if it
drifts, the notes below locate the sites again).

## Findings from the source (pi0_pytorch.py)

- `PI0Pytorch.forward(observation, actions, noise=None, time=None)` already
  accepts caller noise; internally `x_t = t*noise + (1-t)*actions` and
  `u_t = noise - actions` — exactly the convention snmvp's math assumes.
- `sample_actions(device, observation, noise=None, num_steps=10)` likewise
  accepts caller noise, and `Policy.infer(obs, noise=...)` already threads a
  numpy `(H, D)` array through batching/device placement.
- Noise is sampled by `sample_noise` as float32 standard normal.
- Actions are the model's padded motor dim (D=32) in normalized units by the
  time the loss sees them.

## The two hooks

1. **Training (needs the patch):** `scripts/train_pytorch.py` calls
   `losses = model(observation, actions)` in the train loop. The patch wraps
   it: sample noise explicitly, pin `extract_invariant(actions)` (the sum
   over the chunk) via `SourceConstructor`, pass `noise=` to the model.
   Toggled by env var `SNMVP_PIN_ALPHA` (0 = arm A/B baseline path, 1.0 =
   arm C hard pin; fractional = soft pin ablation). DDP-wrapped models are
   handled (`model.module`). Requires `snmvp` installed in the venv
   (`uv pip install -e ~/code/source-noise-mvp`).

2. **Inference (no patch needed):** build calibrated noise caller-side with
   `snmvp.openpi_adapter.make_calibrated_noise(...)` and call
   `policy.infer(example, noise=noise)`. The commanded invariant must be in
   the model's NORMALIZED action units — extract it from normalized demo
   chunks (oracle path), or implement the physical-to-normalized affine map
   against your run's norm_stats.json (see `normalize_invariant` docstring
   for the H-dependent subtlety before trusting physical-unit commands).

Padding note: the training pin covers all 32 padded dims (pinning a constant-
zero dim is a no-op in expectation and consistent between train and test);
the inference adapter pins only the leading `len(invariant)` dims. For strict
symmetry, restrict the training pin with `pinned_dims` to the real actuated
dims of your robot — worth doing before any headline run.

## Consistency requirements

- The invariant must be computed in **normalized action space** (post
  q01/q99 normalization) at both sites. Training actions are already
  normalized when the loss sees them; a commanded invariant at inference must
  be normalized with the same `norm_stats.json` before pinning.
- π0/π0.5 both use a deterministic Euler ODE sampler (~10 steps): noise enters only
  at initialization, so no per-step recalibration is needed. If you switch to
  any stochastic sampler, every injected noise must be recalibrated or the
  pin decays.
- Arm B (conditioning baseline): inject the same normalized invariant as an
  extra input token (proprio-style) instead of into the noise. Keep tokenizer
  changes minimal so parameter counts stay comparable.

  Implemented (2026-07-04) in `patches/openpi_arm_b_conditioning.patch`
  (applies ON TOP of the arm C patch): the z-normalized invariant is written
  into the trailing 7 dims of the 32-dim padded proprio state (LIBERO uses
  only the leading 8, so those dims are constant zero otherwise). The state
  token feeds the action expert through the existing `state_proj` linear —
  parameter count is EXACTLY unchanged, which is the fairest B-vs-C
  comparison available. Enabled by `SNMVP_COND_STATS=/path/to/
  invariant_stats.json` (generate with `scripts/compute_invariant_stats.py`;
  stats must come from the same data subset the run trains on); inert when
  unset. Arm D = `SNMVP_COND_STATS=... SNMVP_PIN_ALPHA=1.0`. Eval-side, apply
  `snmvp.conditioning.inject_invariant_state` to the normalized state with
  the SAME stats file before `sample_actions`/`policy.infer`.

## Blackwell notes

- openpi's pinned torch may predate Blackwell (sm_100/sm_120). Check
  `python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"`
  and if kernels fail, override with a cu128+ build (torch >= 2.7) inside the
  project venv only: `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`.
  Watch for the `transformers` patch step in openpi's README (it copies files
  into the venv's transformers install — venv-local, but note their warning
  about uv's hardlink cache).
- Two GPUs: openpi PyTorch supports single-node DDP via torchrun
  (`--nproc_per_node=2`). Full π0.5 fine-tune wants >70 GB per GPU; if the
  Blackwell cards are B200-class this is fine, if workstation-class check
  memory and consider freezing the VLM trunk and training the action expert
  only (sufficient for the Phase 1 mechanism test).

## Sanity sequence (before any real training)

1. `python tests/test_source_constructor.py` in the training venv (runs the
   torch mirror tests).
2. One training step with `alpha=0.0` — loss must match an unpatched step
   bit-for-bit given the same seed.
3. One training step with `alpha=1.0` — assert `carried_residual(noise,
   actions)` is ~0 for the sampled batch inside the loss function.
4. Overfit 100 steps on 10 episodes, then run inference with an oracle
   invariant and confirm adherence error is small; then the wrong-invariant
   probe on the same overfit model (it should follow the noise even this
   early — the overfit regime is where the pin is easiest to read).
