# Hardware runbook — gate-drone flights with the pi0 baseline and the source-noise pin

Written 2026-09-04 for the collaborator flying the campaign in `docs/real_experiments.tsv`.
Two machines are involved:

| Machine | What runs | Repo / branch |
|---|---|---|
| **manaan** (this box, `SOE-50TJK74.stanford.edu`, 171.64.160.64) | the policy server, one process per arm | `source-noise-mvp` (this repo), `scripts/hw_serve.sh` |
| Drone workstation (ROS 2, mocap, cameras) | the flight node that talks to the server and publishes setpoints | `dronevla2.0`, branch **`gate-pin`**, `python run_policy.py gate ...` |

The VOXL2 onboard stack (`drone_scripts/control.sh`), the RC pilot, arming, takeoff, landing and the
OFFBOARD switch are unchanged from every previous flight. The RC mode switch is the kill switch: leaving
OFFBOARD returns the aircraft to the pilot instantly. Nothing on the server side can stop the drone.

## 0. Pre-flight, once per session (Phase R0 of `docs/REAL_EXPERIMENT_PLAN.md`)

1. Mocap registration: wand-walk the gate posts and confirm they land on the scene cloud
   (`experiments/rung3/viz/scene_cloud_*.npz`) — the judge geometry is only meaningful if this holds.
2. Network: from the workstation, `nc -zv manaan 8900` after starting a server (or set up an ssh tunnel:
   `ssh -N -L 8900:127.0.0.1:8900 manaan` and use `--policy_host 127.0.0.1`).
3. Dry run WITHOUT the drone: on the workstation, in the `dronevla2.0` root,
   `python tools/gate_dry_client.py --host manaan --port 8900 --task left --episode <a data_gate_real ep_XXXX.npz>`
   must print six replans with `chunk (50, 7)` and a latency line. This exercises the exact observation
   code the flight node uses (256 bilinear -> RGB -> 224 bicubic, 7-float state, exact prompt).
4. Latency: the flight budget is a **< 400 ms** round trip sustained. See "Latency" below.
5. `python policy_nodes/tests/test_gate_obs.py` passes on the workstation's python (needs numpy, opencv, pillow).

## 1. Start a server (GPU box)

```bash
cd ~/code/source-noise-mvp
bash scripts/hw_serve.sh baseline                  # pi0 baseline, gate_scratch3
bash scripts/hw_serve.sh ours                      # source-noise pin, gate_pin_joint_xswap + sigma_map_xswap
bash scripts/hw_serve.sh noswap                    # w/o sim-real swap, gate_pin_joint_gmsig3 + sigma_map_gmsig3
bash scripts/hw_serve.sh ours --sketch cmpl_denis  # + a sketch command carried by the server
```
Options: `--port 8900` (default), `--bind 0.0.0.0` (default; `127.0.0.1` when tunnelling), `--tag <name>`
(names the per-replan command log `~/gate_flights/clog_<tag>.npy`). The script prints the matching client
command. One server per terminal; it runs in the foreground and Ctrl-C stops it. Model load takes about
a minute; the first request compiles for about 10 s — send the dry client once before the pilot lifts off.

Sketch names: `cmpl_denis` (hand-drawn L->C), `cmpl_min4` (4-click L->C, sigma 0), `cmpl_min4s` (4-click,
sigma 0.5), `cmpl_min4v2` / `cmpl_min4sv2` (redrawn 4-click, recommended over the originals — see
RESEARCH_LOG 2026-09-02), `cmpr_denis_r2` (hand-drawn R->C, redrawn), `tempo06` / `tempo10` / `tempo15`,
`orbit`, `fig8`. Tempo/orbit/fig8 are flown as the **right** task; compounds as **compound_left** /
**compound_right**. The sketch activates when the drone comes within its entry radius of the drawn line.

## 2. Fly a trial (drone workstation)

```bash
cd ~/dronevla2.0 && git checkout gate-pin
python run_policy.py gate --task left --policy_host manaan --policy_port 8900 --trial ours_left_01
```
`--task` is one of `left | right | center_from_left | center_from_right | compound_left | compound_right`
(the exact training strings live in `policy_nodes/gate_obs.py`). Give every trial a unique `--trial`: it keys
the server's per-flight state and names the log files. Useful flags: `--fence XMIN XMAX YMIN YMAX` (lateral
mocap fence, the node stops publishing outside it), `--infer_timeout 3 --max_timeouts 3` (after three
consecutive server timeouts the node stops publishing and prints PILOT: TAKE OVER), `--dry_run` (everything
except publishing), `--apc 8` (steps per replan, the sim protocol).

Sequence: server up and warmed -> node started, it waits for pose + both cameras -> pilot takes off in
MANUAL/POSITION -> pilot flips to OFFBOARD -> node's setpoints take effect -> flip back to abort or when done
-> Ctrl-C the node (it saves its logs on exit).

## 3. What each trial leaves behind

Workstation `~/gate_flights/`: `traj_<trial>.npy` (N x 3 mocap positions at every executed step) and
`<trial>.jsonl` (per replan: pose, latency, net displacement, max speed). GPU box `~/gate_flights/`:
`clog_<tag>.npy` (per replan: position, the 16-dim command, mixture weights, sigma). Also record the onboard
video and the mocap bag as the plan requires.

## 4. Score

Copy the `traj_*.npy` files to the GPU box and run the same judge the sim rows use:
```bash
bash scripts/hw_score.sh left  ~/gate_flights/traj_ours_left_*.npy
bash scripts/hw_score.sh compound_left ~/gate_flights/traj_ours_cmpl_*.npy
```
Success = route-clean transit (correct gates, correct direction, goal reached) AND minimum clearance
>= 0.18 m to the gate cloud. Report both counts per cell, as in the paper tables.

## 5. The table, as commands

| Row | Server (GPU box) | Client `--task` | Trials |
|---|---|---|---|
| pi0 baseline, four atomics | `hw_serve.sh baseline` | `left`, `right`, `center_from_left`, `center_from_right` | 5 each |
| Ours, four atomics | `hw_serve.sh ours` | same four | 5 each |
| w/o sim-real swap, four atomics | `hw_serve.sh noswap` | same four | 5 each |
| Ours, hand-drawn compound | `hw_serve.sh ours --sketch cmpl_denis` | `compound_left` | 5 |
| Ours, 4-click compound | `hw_serve.sh ours --sketch cmpl_min4v2` (or `cmpl_min4sv2`) | `compound_left` | 5 |
| Ours, tempo 0.6 / 1.0 / 1.5 | `hw_serve.sh ours --sketch tempo06` etc. | `right` | 2 each |
| Ours, orbit | `hw_serve.sh ours --sketch orbit` (wide margins) | `right` | 2 |
| Ours, figure-eight | `hw_serve.sh ours --sketch fig8` | `right` | 2 |

Fly the baseline arm first: it establishes the room and shakes out the ops. Run cells in the pre-registered
order R1 -> R5 and stop the campaign on any contact.

## Latency (measured 2026-09-04, dry client on the same box as the server)

| Server | Per replan, median | First call |
|---|---|---|
| xswap pin server, after the head-forward jit fix | 133 ms | about 8 s (compile) |
| xswap pin server, before the fix | 2.3 s | 10.6 s |

The fix (a compiled GMM-head forward in `experiments/rung3/joint_head.py`) was checked against the eager path
on real frames: same mixture component on 6/6, served command within 0.2 % of a command-std. Wi-Fi adds its own
round trip; re-measure with the dry client from the workstation before flying. Always send one dry-client
request after starting a server so the compile happens on the ground.
