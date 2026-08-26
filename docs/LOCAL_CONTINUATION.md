# Local continuation guide — 4090 machine (written 2026-08-19)

For the agent kicking off experiments on this machine (hostname SOE-50TJK74, single RTX 4090
24 GB, driver 580.173.02). The box (EC2, 2× ~98 GB) is gone; this is the successor environment.
Read `docs/status_latest.md` for where the research stood at box cutoff, `CLAUDE.md` for the
north star and operating rules (they still apply — seed policy, success scoring, no sim ground
truth in product models). This file is only the machine/bootstrap layer.

## Environments (never bare `uv run` — it downgrades torch; call venv pythons directly)

| What | How |
|---|---|
| openpi + SNMVP patch (train/serve new arms) | `PYTHONPATH=~/code/openpi-snmvp/src ~/code/openpi/.venv/bin/python` |
| openpi stock 16affa3 (Denis's live checkout, serve old ckpts) | `~/code/openpi/.venv/bin/python` — do NOT modify this tree |
| falsify sim/renderer (gsplat 0.1.13, torch 2.1.2+cu121) | `~/code/falsify/.venv/bin/python` |
| plain analysis (numpy/torch-cpu+cuda) | `python3` (miniforge base) |

`~/code/openpi-snmvp` is a git worktree of `~/code/openpi` pinned at upstream `15a9616` with
`patches/openpi_joint_gen_head_full.patch` applied (pin, joint MSE/logmag/GEN heads, C2
flow-detach, FiLM head, zero-pad fix, all `pi0_gate*` configs) **plus one local fix**: the
data loader passes `local_files_only=True` for `local/*` repos (this machine has no HF token;
anonymous hub probes of the `local/` namespace 401 instead of 404). PYTHONPATH beats the venv's
editable install, so the one venv serves both trees. Single GPU: no jobq parallelism; JAX grabs
the whole card — `nvidia-smi` first, one training or one serving job at a time
(`XLA_PYTHON_CLIENT_MEM_FRACTION` if colocating a small eval client).

## Data (all local — nothing needs the box)

- Source of truth: `~/code/falsify/data/no_3pov_v3/gate_scenes_all_no_3pov` — LeRobot **v3.0**,
  300 eps = 100 real (50 L + 50 R teleop, variable length) + 200 synth (50 L + 50 R @241
  frames, 100 center @301), 4 task strings. **Images are legacy BGR bytes + fisheye** (its
  PROVENANCE.md is authoritative, incl. the frame-count synth/real classifier and the
  episode-53 exception).
- Training dataset `local/gate_nav` (LeRobot v2, RGB-corrected): built by
  `experiments/rung3/gate_v3_to_lerobot.py` (recovered box script, adapted) into
  `~/.cache/huggingface/lerobot/local/gate_nav` (symlink → `~/Documents/datasets`). Also writes
  `rung3/gate_{synth,real}_eps.json`. Launched 2026-08-19, log: `rung3/gate_convert.log`
  (`GATE_CONVERT_DONE` = finished; expect synth=200 real=100).
- Raw npz mirrors for the rung3 analysis scripts: run `gate_extract_raw.py` twice —
  `EPS=gate_synth_eps.json OUT=data_gate_synth` and `EPS=gate_real_eps.json OUT=data_gate_real`
  (needs `HF_HUB_OFFLINE=` unset? no — it loads via lerobot; if the hub 401 bites, add
  `local_files_only=True` to its `LeRobotDataset(...)` call the same way as the loader fix).
  The repo's existing `data_gate_*/meta.json` are the BOX's copies (kept for provenance of old
  findings); regeneration overwrites them — episode ORDER may differ from the box's, so old
  episode-indexed artifacts must not be assumed aligned.
- Norm stats / serving assets: `~/hf_bundle/gate-drone-pi0` (restored from backup) and
  `~/code/falsify/local/assets/gate_nav/norm_stats.json`.

## Checkpoints

- Local (pre-box generation, from the HF bucket): `~/code/falsify/local/checkpoints/`
  {gate_both_pin, gate_both_scratch, gate_synth_scratch}. Smoke-verified:
  `cd ~/code/falsify && ~/code/openpi/.venv/bin/python local/smoke_gate.py
  --ckpt local/checkpoints/gate_both_pin --norm local/assets/gate_nav` → `SMOKE_OK`.
- Box-era arms (b2lam03, C2, mh16, gen*, genfilm) are LOST; retrain from `pi0_base`
  (cached in `~/.cache/openpi`). Remember `--save-interval=5000` (6 GB/ckpt disk trap)
  and ~166 GB free on this disk.

## Rebuild chain for a new arm (the order rule: U → flow → features → prior)

1. **Basis (RRR, K=5)**: recipe = `rung3/tmp_scripts_rescue/make_u_rrr_gate.py` — OLS(VLM
   prefix features → normalized H=50 chunk), U = top-K eigvecs of Cov(Ŷ); box paths inside
   (`/home/ubuntu/hf_bundle`, `data_gate_synth`) need the local equivalents. `refit_rrr_basis.py`
   is the checkpoint-robustness check (principal angles; box finding: RRR ≈ PCA within 0.2° on
   4/5 dirs, so the basis does not inherit VLM drift). Stamp with `rung3/pin_basis.py`
   (priors record basis sha256 + `feat_ckpt`; servers refuse mismatches — keep this).
2. **Flow**: train via the `pi0_gate*` configs with `SNMVP_PIN_U` (+ `SNMVP_ZERO_PAD_ACTIONS=1`,
   canonical recipe per RESEARCH_LOG 2026-08-11: no tail weighting). Head arms:
   `SNMVP_HEAD` (+`SNMVP_HEAD_GEN`, `SNMVP_HEAD_FILM`, `SNMVP_FLOW_DETACH` for C2-routing).
   Chain scripts: `scripts/run_joint_arm3.sh` (staggered eval clients — keep the stagger),
   `scripts/run_center_addon.sh`, `scripts/run_eval10.sh`.
3. **Serve/eval**: falsify renderer + `serve_gate_pin*.py`; success = falsify posthoc judge +
   `gate_clearance.py` (0.18 m body) + human video, APC=50, ≥10 rollouts, seed replication for
   deltas <5 strict points (all per CLAUDE.md).

## The next planned arm (from the 2026-08-19 toy result, `experiments/toy_cmdhead/`)

GMM/MDN command head with genfilm's information diet (FiLM channels: state, language pool,
image pool), NLL loss, serve = argmax-mode with π-hysteresis; MSE-twin control. Rationale: toy
shows GMM==CFM distributionally, argmax serve is valid + jitter-free (k-mean smoothing
re-averages modes), and explicit π(o) makes posterior calibration an offline readout. The toy
also says the calibration lottery is feature-side, so pair the GMM head WITH the FiLM diet,
not concat. `SNMVP_HEAD_GEN`/`SNMVP_HEAD_FILM` in `openpi-snmvp/src/openpi/models/pi0.py` are
the patterns to extend (a `SNMVP_HEAD_GMM` sibling).

## Bring-up completed 2026-08-19 evening (this section supersedes the plan items above)

- Renderer venv: `~/code/tv` (PERSISTENT successor of the box's /tmp/tv): py3.10 +
  torch 2.1.2+cu121 + gsplat 1.5.3 prebuilt wheel from docs.gsplat.studio/whl/pt21cu121 (cp310 is
  the only linux wheel) + openpi_client/imageio/packaging. Rasterization smoke-passed.
- `~/code/falsify-pi` is a SYMLINK to `~/code/falsify` (gate_success.py / gsplat_scene_edit.py
  expanduser it). Literal /home/ubuntu paths rewritten to /home/dfliu in gate_rollout_batch,
  gate_clearance, joint_head, refit_rrr_basis.
- Second worktree venv-compat fix (same class as the loader fix): checkpoints.py
  CallbackHandler.async_save ported from 16affa3 (venv orbax 0.11.1 lacks the 15a9616 API;
  bites at the FIRST checkpoint save, not at import).
- npz mirrors + basis DONE: layout matches the box convention (CFL 0-49/CFR 50-99/L 100-149/
  R 150-199); pin_U_gate_rrr_k5.npy sha256 ac49ae6b16bc..., RRR~=PCA reproduced. Feature cache:
  vlm_feat_gate_prefix_local.npz (stamped to gate_both_pin).
- Chain script for this machine: `scripts/run_joint_arm_local.sh` (single GPU, six cells,
  120 s stagger, server at XLA fraction 0.45 to colocate with render clients). Run dir ~/ctxrun.
- zsh gotcha: `env PYTHONPATH=~/...` does NOT tilde-expand (arguments to env are not
  assignments) — silently imports the STOCK tree. Use $HOME. Prefix assignments
  (`PYTHONPATH=~/... python`) are fine.
- Long jobs MUST be `setsid ... </dev/null > log & disown` — session-tracked background tasks
  get killed by the harness after ~5-30 min.

## Known local gotchas

- `~/.cache/huggingface/lerobot` is a symlink to `~/Documents/datasets` (recreated 2026-08-19).
- No HF token on this machine → `local/*` lerobot repos need `local_files_only=True` (loader
  in the worktree already fixed; fix other call sites as hit).
- This venv's lerobot is older than the box's: `save_episode(task=...)` not per-frame task,
  `LEROBOT_HOME` not `HF_LEROBOT_HOME` (see `gate_v3_to_lerobot.py` header).
- falsify checkpoints' serving contract: RGB 224², two cams, 7-D EE-delta, replan ~8
  (`~/code/falsify/tools/gate_pi0/README.md`) — but box findings moved to APC=50 for the pin
  line; the falsify-side client must execute 50 steps/chunk for parity with box results.
- Kernel/nvidia-module skew caused a driverless boot 2026-08-14→19; if `nvidia-smi` dies after
  a future kernel update, check `linux-modules-nvidia-580-open-$(uname -r)` is installed.
