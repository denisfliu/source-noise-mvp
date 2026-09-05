# Hardware agent guide — serving gate policies from manaan

For a person or an agent who has to start, check, or fix the policy server during a flight session.
Read `docs/HARDWARE_RUNBOOK.md` for the flight procedure; this file is the reference behind it.
Everything here was verified on 2026-09-04 (dry client end to end, 133 ms per replan).

## Machines

| Name | Hostname / address | Role |
|---|---|---|
| **manaan** | `SOE-50TJK74.stanford.edu`, 171.64.160.64 | this box: RTX 4090 (24 GB), all checkpoints, the policy servers, the judge |
| drone workstation | lab machine on the ROS 2 domain | `dronevla2.0` branch `gate-pin`; runs `run_policy.py gate`; mocap (VRPN) and cameras arrive here |
| VOXL2 (onboard) | on the drone | PX4 + `drone_scripts/control.py`; manual arm/takeoff/land; RC mode switch = kill switch |

The workstation's client resolves `--policy_host manaan` to `SOE-50TJK74.stanford.edu`
(`dronevla2.0/policy_nodes/common.py`, `POLICY_HOSTS`). If DNS fails use the IP, or an ssh tunnel
(`ssh -N -L 8900:127.0.0.1:8900 dfliu@171.64.160.64`, then `--policy_host 127.0.0.1`).

## Paths on manaan

| What | Path |
|---|---|
| this repo | `~/code/source-noise-mvp` |
| serving code | `~/code/source-noise-mvp/experiments/rung3/` (`serve_gate_pin_joint.py`, `serve_gate_plain.py`, `joint_head.py`, `sketch_prompt.py`) |
| python for servers | `~/code/openpi/.venv/bin/python` with `PYTHONPATH=~/code/openpi-snmvp/src` (the modified openpi; its diff is `patches/openpi_snmvp_working_tree_*.patch`) |
| python for scoring/clearance | `~/code/tv/bin/python` (torch + gsplat) |
| checkpoints | `~/code/openpi-snmvp/checkpoints/pi0_gate3/<arm>/4999` |
| norm stats | `~/hf_bundle/gate-drone-pi0/assets/gate_nav` (7 real dims, padded to 32 by `gate_ctx_common.pad_norm_stats`) |
| command basis | `experiments/rung3/pin_U_mh16.npy` (1600 x 16) |
| trust maps | `experiments/rung3/sigma_map_<arm>.json` |
| sketches | `experiments/rung3/sketch_<name>.json` |
| per-flight command logs | `~/gate_flights/clog_<tag>.npy` (server) |
| judge geometry | `~/code/falsify-pi/configs/safety/*.yaml`; scene clouds `experiments/rung3/viz/scene_cloud_*.npz` |

## Arms (rows of `docs/real_experiments.tsv`)

| `hw_serve.sh` arm | Paper name | Checkpoint dir | Server | Extra |
|---|---|---|---|---|
| `baseline` | pi0 baseline | `gate_scratch3` | `serve_gate_plain.py` | none (model draws its own noise) |
| `ours` | full method | `gate_pin_joint_xswap` | `serve_gate_pin_joint.py` | pin env + `sigma_map_xswap.json` |
| `noswap` | w/o sim-real swap | `gate_pin_joint_gmsig3` | `serve_gate_pin_joint.py` | pin env + `sigma_map_gmsig3.json` |

Seed-7 replicas exist (`gate_pin_joint_xswaps7`, `gate_scratch3s7`, sigma maps `*s7.json`) and the sim ablation
arms (`gate_pin_joint_nosig`, `gate_pin_joint_synthonly`, `gate_pin_joint_gmsig4`); add them to `hw_serve.sh`'s
`case` block if they are to be flown. A pin checkpoint must be served with **its own** sigma map and with the
pin environment; the baseline must be served by `serve_gate_plain.py` (feeding it pinned noise puts it 4 to 6
sigma off its training distribution).

## The pin environment, variable by variable

Set by `hw_serve.sh` for `ours` / `noswap`; the meaning matters when something is off.

| Variable | Value | Meaning |
|---|---|---|
| `SNMVP_HEAD=1`, `SNMVP_PIN_U=<U>` | required before openpi import | builds the command-head submodules the checkpoint contains; without them the load fails with "this checkpoint has no snmvp_q" |
| `SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1` | training-time flags | must match the checkpoint (all pin arms here are GMM heads trained joint) |
| `SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1` | training-time flags | sigma-conditioned flow; `snmvp_sigma` becomes a valid inference input |
| `SNMVP_SIGMA_MAP=sigma_map_<arm>.json` | per arm | maps the head's own uncertainty to the trust input; the sim results used it; omit and the flow is served at full trust |
| `SNMVP_ZERO_PAD_ACTIONS=1` | always | action dims 7..31 are zero padding |
| `CLOG=<path.npy>` | per run | per-replan log rows `[pos(3), c(16), mixture weights, sigma*, alpha, sigma_serve, phase]`; rewritten every replan |
| `SNMVP_PIN_PROMPT=sketch_<name>.json` | sketch rows | the server carries the sketch; it activates within the entry radius of the drawn line and swaps the prompt to the sketch's `prompt_after` |
| `SNMVP_GMM_HYST=0.2` | default | mixture-component hysteresis per `snmvp_trial`; why each flight needs a unique `--trial` |
| `SNMVP_NOISE_SEED` | default 0 | residual-noise stream; change for rollout-seed replicates |
| `SNMVP_INTENT_WS`, `SNMVP_INTENT_BIND` | off by default | the intent cockpit; in gate mode the server **blocks** until a human approves — do not enable unattended |
| `SNMVP_PIN_OFF`, `SNMVP_PIN_DECODE_ONLY`, `SNMVP_PIN_ADVICE`, `SNMVP_PIN_REASON` | diagnostics | never set for a flight |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | `hw_serve.sh` default | JAX allocates on demand; if another GPU process must coexist use `=true` with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.42` |

## Sketches

| name | drawing | fly as `--task` | notes |
|---|---|---|---|
| `cmpl_denis` | hand-drawn L->C | `compound_left` | sim 5/5 clean on both xswap seeds |
| `cmpl_min4v2`, `cmpl_min4sv2` | redrawn 4-click L->C, sigma 0 / 0.5 | `compound_left` | prefer over `cmpl_min4` / `cmpl_min4s`, whose line shaves the centre post |
| `cmpr_denis_r2` | redrawn hand-drawn R->C | `compound_right` | |
| `tempo06`, `tempo10`, `tempo15` | 4-point route at 0.6 / 1.0 / 1.5x pace | `right` | clearance degrades off-pace in sim |
| `orbit`, `fig8` | 1.5 loops around the right gate; figure-eight | `right` | wide margins; the judge reports these as failures by design (they cross the plane) |

Sketch geometry is the first suspect for a graze: `experiments/rung3/sketch_geom.py`'s `evaluate(points, scene)`
returns the line's own clearance to the gate cloud and where it pierces each aperture.

## Start, verify, stop

```bash
cd ~/code/source-noise-mvp
bash scripts/hw_serve.sh ours --tag ours_$(date +%H%M)         # foreground; ~1 min load; prints the client command
bash scripts/hw_status.sh                                       # in another shell: pid, port, checkpoint, sketch, GPU
bash scripts/hw_status.sh --test 8900 left                      # dry client: 6 replans, chunk (50, 7), latency line
bash scripts/hw_status.sh --kill 8900                           # stop the server on that port (by pid)
```
"Ready" is the log line `[serve_gate_pin_joint] ready on ws://0.0.0.0:8900` (or `[serve_gate_plain] ready`).
The first request compiles (about 8 s); run the `--test` once before the pilot lifts off. After that a replan is
about 135 ms on this box (flow 80 ms, command head 55 ms) plus the network.

## The wire contract (what the client must send, what the server returns)

Request keys: `observation/image`, `observation/wrist_image` (uint8 224x224x3 **RGB**, built as native ->
cv2 256 bilinear -> BGR-to-RGB -> PIL 224 bicubic, no undistortion, no overlay), `observation/state`
(float32[7] = x, y, z mocap metres, mocap yaw radians, 0, 0, 0), `prompt` (an exact training string),
`snmvp_trial` (unique per flight), `reset` (first replan), `progress` (ignored). Reply: `actions` float32
(50, 32); columns 0..6 are per-step mocap deltas dx, dy, dz, dyaw and zeros; execute 8 steps at 10 Hz, then
replan. The client integrates `pose + cumsum(deltas)` and publishes absolute mocap setpoints.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `hw_serve.sh` exits at once with "Address already in use" / probe bind error | a server already holds the port (possibly the wrong checkpoint) | `hw_status.sh`, then `hw_status.sh --kill <port>`; never leave a stale server on 8900 |
| load fails: "this checkpoint has no snmvp_q ... SNMVP_HEAD/SNMVP_PIN_U were unset" | pin checkpoint started without the pin env, or via `serve_gate_plain.py` | use `hw_serve.sh ours/noswap` |
| baseline through the pin server, or pin arm through `serve_gate_plain.py` | wrong server for the checkpoint | baseline -> `serve_gate_plain.py`; pin arms -> `serve_gate_pin_joint.py` |
| CUDA out of memory at load, or the renderer/another process OOMs | two GPU processes preallocating | one server at a time; or `XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.42` for each |
| first request takes 8-10 s, drone holds position | JIT compile | expected once per server; warm with `hw_status.sh --test` before flying |
| every replan takes seconds | the compiled head is not in use (old `joint_head.py`) or GPU contention | `git pull`; `nvidia-smi`; expect 133 ms locally |
| client prints "server not up ... retrying" forever | port/host wrong, server still loading, firewall | `hw_status.sh` shows listening or not; `nc -zv manaan 8900` from the workstation; check `--bind 0.0.0.0` |
| client: "unexpected action shape" | wrong server (returns other than (50, >=7)) | check the server script in `hw_status.sh` |
| flights go through the gate but the wrong way / mirror image | yaw sign or y sign broken on the client | the node's state must be mocap yaw (not negated); `control.py` negates y, z, yaw to NED; do not "fix" both |
| flights consistently miss or graze although sim was clean | image chain changed (BGR, letterbox, 255 px), or fisheye rectification added | run `policy_nodes/tests/test_gate_obs.py` on the workstation; never undistort; frames stay raw fisheye |
| policy ignores the sketch | drone never came within the sketch's `enter_radius` of the line; wrong task prompt; sketch env not set | `hw_status.sh` shows the sketch; check `[sketch] ... points -> steps` in the server log; the client flies `compound_left` etc. |
| server log shows `sigma_serve=-1` or trust always 0 | sigma map missing | `SNMVP_SIGMA_MAP` set by `hw_serve.sh`; check the file exists |
| server appears hung inside a request | intent cockpit enabled in gate mode with no operator | never set `SNMVP_INTENT_WS` for unattended flights; kill and restart |
| node stops with "PILOT: switch out of OFFBOARD" | three consecutive timeouts (`--infer_timeout`) | pilot takes over; check server, network; restart node with a new `--trial` |
| scoring says every flight failed | judge geometry not registered to the room's mocap | wand walk (R0); compare a mocap track of a gate post to `scene_cloud_*.npz` |

## Adding an arm or a sketch

Arm: add a `case` line in `scripts/hw_serve.sh` (checkpoint dir, sigma map) and a row in the table above.
Sketch: draw in the Sketchpad (`experiments/rung3/viz/sketchpad.html`, exports `sketch_<name>.json`), check it
with `sketch_geom.evaluate`, fly it in sim first (`scripts/run_xswap_sketches3.sh` is the template), then add
it to the sketch table. Keep `prompt_after` an exact training string.

## Never

Change the image chain on either side. Enable fisheye rectification. Fly with the intent cockpit in gate
mode unattended. Serve a pin checkpoint without its own sigma map. Run two servers on one port. Kill servers
by process-name pattern from a shell whose command line contains that name (kill by pid via `hw_status.sh`).
