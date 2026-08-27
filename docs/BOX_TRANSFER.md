# Box transfer — restoring the local-4090 state on the EC2 box (written 2026-08-25)

This repo is the source of truth for the local-4090 arc (2026-08-19 → now). Everything
code-shaped is committed here; the two external trees we modified travel as full patches in
`patches/`; large data and checkpoints travel by regeneration or rsync per the manifest below.
Read `docs/RESEARCH_LOG.md` (bottom) and `experiments/FINDINGS_INDEX.md` for where the science
stands; `docs/LOCAL_CONTINUATION.md` for the ops details behind these steps.

## 1. This repo

```
git clone git@github.com:denisfliu/source-noise-mvp.git
```
(private; pushed from the local 4090 — the `~/source-noise-mvp-*.bundle` file is a fallback
if the box lacks GitHub SSH access)

Carries: `experiments/rung3/` (joint head, serve, judges, probes, viz builders, basis .npy,
sigma maps, adherence caches), `scripts/` (train/eval/regen chains), `docs/`, `patches/`,
`src/snmvp/`, and `assets/openpi/` — a mirror of the openpi norm-stats assets
(`pi0_gate{,2,3}`), which pin the checkpoint↔normalization pairing and MUST be copied into
the openpi tree (step 2) before serving.

NOT in git (see §4): `experiments/rung3/data_gate_synth3/` (16G, regenerable),
`experiments/rung3/data_gate_real/` (6.3G, re-extractable), `vlm_feat_gate_prefix_local.npz`
(re-cacheable).

## 2. openpi fork

```
git clone https://github.com/Physical-Intelligence/openpi openpi-snmvp
cd openpi-snmvp && git checkout 15a9616
git apply <repo>/patches/openpi_snmvp_local_15a9616.patch
cp -r <repo>/assets/openpi/pi0_gate* assets/
# env: uv sync per openpi README; on Blackwell install cu128 torch in-venv only
```

The patch carries: all SNMVP env-gated training/serve hooks in `models/pi0.py` (pin,
SNMVP_HEAD, SNMVP_HEAD_GMM + FiLM diet + NLL, SNMVP_HEAD_COND_DROP per-channel,
SNMVP_PIN_NOISE/RAND/COND σ-conditioning, SNMVP_GMM_LANG_CFG guidance, `snmvp_sigma`
inference threading), `policies/policy.py` (infer `snmvp_sigma` kwarg), `training/config.py`
(`pi0_gate`, `pi0_gate2`, `pi0_gate3`), `training/checkpoints.py` (orbax async_save port for
orbax 0.11.1), data_loader/weight_loaders/train_pytorch tweaks.

## 3. falsify-pi (data generation)

```
cd falsify-pi && git checkout 280d160
git apply <repo>/patches/falsify_pi_local_280d160.patch
```

The patch carries: the four course YAMLs (anchor ladders, two-class `jitter_m`
gate-sphere/corridor-tube waypoints, return berths, CFR standoff), `plan_course_variants.py`
(`--start-jitter/--start-mean/--snap-kt` — `--snap-kt none` is what defeats the kT=10.0
rush-and-park bug), `waypoints.py`/`perturbations.py` (per-waypoint jitter radius), and the
`tools/gate_pi0/` scripts. The gsplat render env is separate (torch 2.11 + cu128 +
gsplat 1.5.3 + ninja binary; see `docs/LOCAL_CONTINUATION.md` — on the box, rebuild in a
persistent dir, not /tmp).

## 4. Data / checkpoints manifest (large, not in git)

| artifact | size | how to restore |
|---|---|---|
| `experiments/rung3/data_gate_synth3/` | 16G | regenerate: `scripts/regen2_phaseA_plan.sh` → `scripts/regen_phaseB_render.sh` (needs falsify-pi + gsplat env + scene splats) → episodes land as npz |
| `experiments/rung3/data_gate_real/` | 6.3G | re-extract from the HF bundle: `experiments/rung3/gate_extract_raw.py` (REPO env → `~/hf_bundle/gate-drone-pi0`) |
| LeRobot `local/gate_nav3` | ~30G | rebuild from synth3+real: `experiments/rung3/build_gate_nav3.py` (then copy norm stats per §2) |
| checkpoints `pi0_gate3/gate_pin_joint_gmsig3` | 5.8G | **rsync — flagship (40/40 route-clean)**; retrain otherwise: `scripts/run_gmsig3` recipe (SNMVP_HEAD_GMM=1, σ-cond, gate_nav3) |
| checkpoints `pi0_gate3/gate_pin_joint_gmsig4` | 5.8G | rsync — CFG/null-language arm (per-channel COND_DROP); retrainable same way |
| checkpoints `pi0_gate3/gate_scratch3` | 5.8G | scratch control; retrainable |
| old-data arms (`pi0_gate/gate_pin_joint_*`) | 5.8G ea | historical (claims withdrawn 2026-08-25, route-clean rule) — do not transfer |
| `~/ctxrun/traj_*.npy`, clogs, scores | ~1G | rollout evidence for logged claims — rsync if the audit trail should survive the machine |
| gsplat scene assets + `~/hf_bundle/gate-drone-pi0` | — | HF: re-download (`tools/gate_pi0/upload_gsplat_hf.py` documents the repo ids) |

Checkpoint transfer: `rsync -a --info=progress2 ~/code/openpi-snmvp/checkpoints/pi0_gate3/ box:...`
— gmsig3 and gmsig4 are the two that matter; everything else is rebuildable from this repo +
data regen in ~1 GPU-day each.

## 5. Sanity sequence on the box (in order, cheap → expensive)

1. `pytest src/snmvp` (numpy invariant tests).
2. Judge self-check: `experiments/rung3/gate_success.py` on a few rsynced `traj_armgmsig3_*`
   — must reproduce 10/10 route-clean left/right.
3. Serve smoke: `serve_gate_pin_joint.py` on rsynced gmsig3 + `assets` norm stats; one
   rollout per atomic cell (headless client per `scripts/run_sixcell_eval_local.sh`,
   adjust GPU fractions — the box has 2×98G, drop the 0.45 squeeze).
4. Only then: new training. Ordering rule stands: U from action statistics → flow trained
   with oracle c = U^T a → features → prior.
