# Extraction guide — pulling this box's state to another machine (2026-08-13)

Written for a machine that will SSH into this box (`ubuntu@<this-host>`). Everything research-
critical lives in five places; sizes and pull commands below. The RESEARCH_LOG (docs/) and
FINDINGS_INDEX (experiments/) are the narrative; this file is the physical inventory.

## 1. The repo itself — `~/code/source-noise-mvp` (small, pull all of it)

    rsync -av ubuntu@HOST:code/source-noise-mvp/ ./source-noise-mvp/

Uncommitted by convention (single `initial import` commit; Denis's flow doesn't use commits —
`git status` shows the live state). Contents of note beyond the obvious:

- `experiments/rung3/` — all analysis/serving/eval code for the drone-gate line. Every script has
  a header docstring stating what it measures and which finding it produced.
  Key instruments (each is a standing measurement, re-runnable):
  `manifold_tail_probe.py` (restoring-field / covariate analysis),
  `feature_separation_probe.py` (task separability of a head's own features, by phase),
  `residual_bimodality_audit.py` (mode-averaging falsification test),
  `mh_basis_audit.py` (multi-horizon basis builder + go/no-go audit),
  `sim_real_c_probe.py` (real-vs-sim command consistency at matched states),
  `confirm_vlm_rrr.py` (LIBERO RRR-from-VLA 2x2 confirmation).
- `experiments/rung3/viz/` — the artifact-page builders (cloudviewer/gridviewer WebGL viewers,
  trajectory-grid builder) + the decimated scene clouds they render. `build_traj_grid.py`
  regenerates the 6-scene grid page from `/home/ubuntu/ctxrun` score files.
- `scripts/run_joint_arm*.sh, run_center_addon.sh, run_eval10.sh` — the arm pipeline
  (train -> readout gate -> 6-cell closed-loop eval). `run_joint_arm3.sh` is current
  (all six cells, staggered clients, --seed support); arm/arm2 are kept for provenance of
  older runs only.
- `patches/openpi_joint_gen_head_full.patch` — THE COMPLETE DIFF of our openpi working tree
  against upstream commit `15a9616` (see §2). 549 lines; includes the pin, the joint command
  head (MSE/logmag/generative CFM), C2 flow-detach routing, freeze-VLM config, zero-pad fix,
  and the norm-stats/asset plumbing.

## 2. openpi — `~/code/openpi` (env + working tree)

Clone upstream at `15a9616`, then apply the patch from §1:

    git clone <openpi> && cd openpi && git checkout 15a9616
    git apply source-noise-mvp/patches/openpi_joint_gen_head_full.patch

The venv (`~/code/openpi/.venv`) is box-specific (cu128 torch — see the uv gotcha below).
Env flags the patch adds (all inert unless set): SNMVP_PIN_U, SNMVP_HEAD, SNMVP_HEAD_DETACH,
SNMVP_HEAD_LAM, SNMVP_HEAD_STATE, SNMVP_HEAD_LOGMAG, SNMVP_HEAD_GEN, SNMVP_FLOW_DETACH,
SNMVP_ZERO_PAD_ACTIONS, SNMVP_GEN_STEPS/SNMVP_GEN_SAMPLES (serve-side).
Configs added: `pi0_gate_full` (non-LoRA twin for raw pi0_base loading),
`pi0_gate_freezevlm` (C1 frozen-VLM arm; reuses pi0_gate's norm stats explicitly).

## 3. Checkpoints — `~/code/openpi/checkpoints/pi0_gate*` (~6 GB per arm)

One dir per arm (`gate_pin_joint_<name>/4999/params`). The routing/generative families referenced
in the log: b2lam03 (+s7 seed rep), b2long, b1x/b1s/b1long, c2, c1 (freezevlm), mh16, b2logmag,
gen1, gen1lam1, gen16 (+s7), gen1det. Pull only what you need — each is ~6 GB:

    rsync -av ubuntu@HOST:code/openpi/checkpoints/pi0_gate/gate_pin_joint_c2/ ...

Base model comes from `~/.cache/openpi` (auto-downloads on first use; no need to copy).

## 4. Run artifacts — `/home/ubuntu/ctxrun` (scores, trajectories, videos, CLOGs)

Flat dir, name-keyed: `traj_<arm>_<side>_<t>.npy` (Nx3+ mocap positions),
`overlay_<arm>_..mp4` (review videos), `arm_<name>_scores.txt` / `ctr_<name>_scores.txt`
(judge + clearance dumps the grid builder parses), `clog_<name>.npy` (per-replan [pos(3), c(K)]
command logs — the calibration/forensic instruments read these). ~a few GB total; rsync the
whole dir if in doubt. `ctxrun/invalid/` holds quarantined runs (scene-selection bug) — do not
use for analysis.

## 5. Data + renderer

- Demos: `experiments/rung3/data_gate_synth` (200 eps, 4 tasks), `data_gate_real` (101 eps),
  `data_libero_multi` (in-repo, pulled with §1).
- Norm stats / serving assets: `~/hf_bundle/gate-drone-pi0` (small; rsync it).
- Splat scenes: `~/code/falsify-pi/data/gate_scenes_export` (large; only needed to RENDER —
  scoring geometry is derived via `gate_clearance.py`/YAMLs in `~/code/falsify-pi/configs`).
  NOTE 2026-08-12/13 fixes: `configs/safety/right_and_center.yaml` gate_1 corners were corrected
  on this box — rsync `falsify-pi/configs` too, don't use a stale copy.
- Renderer env: `/tmp/tv` (torch 2.11 + cu128 + gsplat 1.5.3). /tmp is volatile — the restore
  copy lives at `~/tv_env_backup/`; restore to the SAME `/tmp/tv` path (hardcoded shebangs).

## Known gotchas (the expensive ones)

- Bare `uv run` inside openpi RE-SYNCS and downgrades the cu128 torch: always `UV_NO_SYNC=1`
  (or call `.venv/bin/python` directly, as every script here does).
- JAX grabs all GPUs: scope with `CUDA_VISIBLE_DEVICES`; CPU-only = `JAX_PLATFORMS=cpu`.
- The eval clients and servers must not start simultaneously per GPU (handshake+compile race) —
  the current scripts stagger; don't "simplify" that away.
- Rollouts are non-reproducible run-to-run (0.63 m divergence) AND training runs carry ~±5
  strict-point seed variance: >=10 rollouts for any comparison, seed replication for deltas <5.
- Success = `gate_success.py` (falsify posthoc judge) + `gate_clearance.py` (0.18 m body) +
  human video review. Scalars filter; they never declare.
