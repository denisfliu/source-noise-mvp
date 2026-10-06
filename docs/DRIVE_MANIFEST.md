# Drive upload manifest: "Dynamism of a Trajectory" (source-noise pin, quadrotor)

Everything a collaborator needs to reproduce the paper's results, as it sits on **manaan**
(`SOE-50TJK74.stanford.edu`). Paths are absolute on that box. Sizes measured 2026-10-05.

Suggested Drive layout: one top-level folder with the six sub-folders below (`1_code` ... `6_dataset`).
Model weights and splats barely compress; upload them as-is (or `tar` without compression).

| Folder | Contents | Size | Needed for |
|---|---|---|---|
| `1_code` | three source trees | ~0.2 GB | everything |
| `2_splats` | two Gaussian-splat scenes + one overlay image | 10.1 GB | every simulation result |
| `3_models_paper` | ours + π0, trained on 100 real demos | 11.6 GB | main-text results |
| `4_models_appendix` | mixed-data checkpoints | 17.4 GB | Appendix D (more data, flawed sketch) |
| `5_small_assets` | command basis, σ map, norm stats | < 1 MB | serving any checkpoint |
| `6_dataset` (optional) | training demonstrations | 6.3 GB real / 29 GB real+synth | retraining only |

---

## 1. Code (`1_code`)

All three are git repos. The cleanest hand-off is the GitHub remote plus the patch files listed; a
plain folder copy (excluding `.venv`, `checkpoints`, large data) also works.

**a. `source-noise-mvp`** (this repo) — github.com/denisfliu/source-noise-mvp, branch `master`.
Method library, servers, sim rollouts, scorers, figures, logs.
- Copy: `git archive` of `master` (160 MB of tracked files), **plus** the untracked
  `docs/paper/` and this file. Do **not** copy `experiments/rung3/data_gate_real` (6.3 GB) or
  `experiments/rung3/data_gate_synth3` (16 GB) here; the dataset goes in `6_dataset`.
- Entry points used for the paper:
  - `scripts/run_realonly_apc25.sh` — serves a checkpoint and flies the demonstrated-task cells.
  - `experiments/rung3/serve_gate_pin_joint.py` — our policy server (head + flow + σ map).
  - `experiments/rung3/serve_gate_plain_sketch.py` — π0 with an injected sketch / SDEdit / velocity projection.
  - `experiments/rung3/gate_rollout_batch.py` — closed-loop simulation in the splat.
  - Scoring: `gate_success.py` (transit judge), `gate_clearance.py` (18 cm contact radius),
    `route_contact.py`, `score_ood_cells.py`, `rescore_tables.py`.
  - Figures: `experiments/rung3/viz/build_*.py`.
  - History: `docs/RESEARCH_LOG.md`, `experiments/FINDINGS_INDEX.md`.

**b. `openpi-snmvp`** (`/home/dfliu/code/openpi-snmvp`) — Physical-Intelligence/openpi at commit
`15a9616` with our changes uncommitted in the working tree (pin/head/σ training and serving, the
`pi0_gate` config). Rebuild with:

    git clone https://github.com/Physical-Intelligence/openpi.git openpi-snmvp && cd openpi-snmvp
    git checkout 15a9616
    git apply <source-noise-mvp>/patches/openpi_snmvp_working_tree_2026-09-20.patch

(The patch is in repo (a); `patches/README_openpi_snmvp.md` explains the snapshots.)

**c. `falsify-pi`** (`/home/dfliu/code/falsify-pi`) — github.com/denisfliu/falsify-pi at `280d160`.
Provides the posthoc judge and the scene / safety YAMLs (gate apertures). Our 2026-09-25 judge fix
(re-measured apertures, direction labelled by the aperture normal) is uncommitted because that tree has
other people's changes; apply it with:

    git checkout 280d160 && git apply <source-noise-mvp>/patches/falsify_pi_judge_on_280d160.patch
    cp <source-noise-mvp>/patches/test_directional_transit_normal.py tests/

**Environments** (rebuild, don't upload): openpi `uv` env at `~/code/openpi/.venv` (JAX, serving);
gsplat renderer env at `~/code/tv` (torch 2.11 + cu128, gsplat 1.5.3). See `CLAUDE.md` in (a).

## 2. Gaussian splats (`2_splats`)

Only the two captured rooms are needed; the figure-eight, center-gate and arbitrary-gate layouts are
built from these at load time by `experiments/rung3/gsplat_scene_edit.py`.

| File | Size |
|---|---|
| `/home/dfliu/code/falsify/data/gate_scenes_export/left_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt` | 4.4 GB |
| `/home/dfliu/code/falsify/data/gate_scenes_export/right_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_144353/nerfstudio_models/step-000029999.ckpt` | 5.7 GB |
| `/home/dfliu/code/falsify/configs/embodiments/assets/carl_wrist_overlay_pinhole_rgb.png` (strut overlay on the downward view) | small |

Keep the directory structure under `gate_scenes_export/` (the rollout script has these paths
hard-coded near the top of `gate_rollout_batch.py`). The world-to-splat transforms are in that script too.

## 3. Paper models (`3_models_paper`)

Each is an orbax checkpoint directory (`params/`, `assets/`, `_CHECKPOINT_METADATA`); copy the whole
`4999/` folder.

| Checkpoint | Paper label | Size |
|---|---|---|
| `/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_realonly/4999` | ours | 5.8 GB |
| `/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch_real/4999` | π0 | 5.8 GB |

## 4. Appendix models (`4_models_appendix`)

| Checkpoint | Used for | Size |
|---|---|---|
| `.../pi0_gate3/gate_pin_joint_gmsig3/4999` | ours, more data; flawed-sketch figure | 5.8 GB |
| `.../pi0_gate3/gate_scratch3/4999` | π0, more data | 5.8 GB |
| `.../pi0_gate3/gate_pin_joint_xswap/4999` | orbit / figure-eight rows of the more-data table | 5.8 GB |

## 5. Small assets (`5_small_assets`)

| File | What |
|---|---|
| `experiments/rung3/pin_U_mh16.npy` | the command basis U (K = 16) |
| `experiments/rung3/sigma_map_realonly.json` | head uncertainty → served σ map for ours |
| `/home/dfliu/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json` | action/state normalization used by every server |
| `experiments/rung3/sketch_*.json` | the sketched routes (orbit, figure-eight, 13 arbitrary poses, flawed sketches) |

The first two and the sketches are already in repo (a); listed here so they are not missed.

## 6. Dataset (`6_dataset`, optional)

| Path | What | Size |
|---|---|---|
| `experiments/rung3/data_gate_real/` | the 100 real demonstrations (per-episode `.npz`) behind the paper models | 6.3 GB |
| `~/.cache/huggingface/lerobot/local/gate_nav3/` | LeRobot dataset, real + synthetic, used to train the mixed-data models | 29 GB |

---

## Quick start for the collaborator

1. Rebuild the three code trees (section 1) and the two envs.
2. Put the splats back at the paths in section 2 (or edit the `CK=` lines in `gate_rollout_batch.py`).
3. Put the checkpoints under `openpi-snmvp/checkpoints/pi0_gate3/`.
4. Run `bash scripts/run_realonly_apc25.sh realonly 25` for ours on the demonstrated tasks; scores land
   in `~/ctxrun/arm_realonly_apc25_scores.txt`. The π0 cell and the sketch cells have sibling scripts
   in `scripts/` (see `RESEARCH_LOG.md` for the exact command behind each paper number).

Paths in the scripts are absolute to `/home/dfliu`; a collaborator on another machine will need to
edit them (mostly `RD=`, `RUN=`, `CKROOT=`, `HFB=` and the splat `CK=` lines).
