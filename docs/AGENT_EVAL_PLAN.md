# Agent-in-the-loop evaluation plan (drafted 2026-09-22)

The claim under test: an LLM agent (Sonnet) planning in a small vocabulary of movement primitives, executed
through the source-noise pin, completes tasks the policy was never trained on, and does so better than the same
agent driving the same primitives as raw waypoints. Two arms, four tasks, the same agent, brief, primitives,
budget and start pose in every cell; only the execution backend changes.

Decisions taken (Denis, 2026-09-22): two arms only; the primitives are shared between the arms; the agent is
told the policy was trained on the left-gate and right-gate tasks; the orbit is composed from primitives, not a
sketch, and counts if the drone completes a circle around the gate; sim first, then the drone.

## Arms

| Arm | Checkpoint / mode | What a move becomes |
|---|---|---|
| **ours** | `gate_pin_joint_realonly` (100 real demos, pin + head, no swap) | the move's command written into the source; the flow denoises the rest at the agent's sigma; the policy also proposes on every replan, so the agent can APPROVE |
| **waypoints** | same server, `SNMVP_PIN_DECODE_ONLY=1` | the decoded command track U c executed as the chunk, no denoising; sigma is inert; there is no policy proposal, so every decision is an authored move |

Both arms run the real-only checkpoint: the head's goal-box miss (RESEARCH_LOG 2026-09-22) does not enter,
because the agent authors every leg that matters and the real-only pin executes authored routes at the mixed
pin's fidelity (orbit and figure-eight 5/5 in sim). What differs between the arms is only what happens inside
a chunk: the flow bends the move around what the camera sees (sigma > 0) and shapes its kinematics; the
waypoint track is open-loop for 5 s.

## Primitives (identical in both arms)

`--dx --dy --dz` in room metres, `--yaw` in degrees (positive = left), `--sigma` in [0, 1]; clamps 2 m / 1 m /
45 deg per chunk; adaptive execution (an override runs at 0.45 m/s and returns after at most 25 steps, an
approve runs the full 50-step chunk). In the waypoint arm `--sigma` is accepted and ignored, and `approve` is
refused with a message, so a brief written for ours runs unchanged.

## Tasks

| # | Task | Start | Budget | Success (from the trajectory) |
|---|---|---|---|---|
| 1 | find the mannequin, hover in front of it | (0, 0, 1.5) heading -pi | 14 | within 2.5 m of (7.4, -0.2) and heading within 30 deg of the bearing to it for the final 20 steps (Denis: looking at it, somewhat centred, pretty close); never below 0.8 m; no contact (clearance >= 0.18 m to gates) |
| 2 | left gate, then centre gate, hover over the penguin | (0, 0, 1.5) heading 0 | 14 | compound judge (both gates in order, route-clean, goal dwell) AND clearance-clean |
| 3 | fly one full circle around the centre gate | (0, 0, 1.5) heading 0 | 20 | the bearing about the centre gate's midpoint spans 360 deg over the flight, the radius never drops below 0.3 m (no pass through the aperture) or above 3 m, clearance-clean, altitude 0.8-2.0 m |
| 4 | left gate, then find the mannequin | (0, 0, 1.5) heading 0 | 24 | left-gate transit route-clean and clearance-clean, then task 1's criterion |

Task 3's judge: cumulative signed angle of (x, y) about the gate anchor over the flight; success if it reaches
360 deg in one direction and the radius stayed inside the band while it did. Direction is the agent's choice.

## Briefs

One brief per task, shared by both arms except for a single backend paragraph:

- ours: "The policy proposes a plan at every decision; approve it when its motion is right. It was trained on
  two tasks in this room: through the left gate and through the right gate to the stuffed animal. It knows
  nothing else. A sigma of 0.3-0.5 lets it bend your move around what the camera sees; use it near gates."
- waypoints: "Your move is flown exactly as written, with no correction from the camera. The policy that was
  trained on the left and right gate tasks is not driving; there is nothing to approve. sigma has no effect."

Everything else is the current mannequin/penguin brief structure: coordinate frame, L/R marks, phased strategy,
distance scale, decision-log replay, calibration sheet. Task 4's brief has two phases: phase one crosses the
left gate (in ours, "approve the left-gate proposal unless a post fills the frame or the crossing is off
centre"; in waypoints, "author the crossing: line up on the middle of the opening, then one straight move
through"); phase two is task 1's sweep-commit-approach, entered when the crossbar passes overhead in the
downward frame.

## Protocol

- Screen: 5 trials per cell (4 tasks x 2 arms = 40 flights). Claim tier: 10 per cell for any quoted comparison.
- A fresh Sonnet agent per trial (no memory across trials), per-trial mailbox, identical brief text per task.
- Per trial: success, decisions used, path length, min clearance, override rate, median reviewer seconds,
  and the mannequin/orbit metrics above. Every cell gets a cloud page; every flight a video.
- Order: run ours first for task 1 (already 1/1 on the mixed pin, 0 trials on real-only) to confirm the
  real-only backend flies the search; then the full matrix.
- Hardware: the same four tasks on the drone through `dronevla2.0 tools/agent_cli.py` once the setpoint
  tracking fix is verified (`--agent_settle_s`), with the workstation briefs in
  `docs/HARDWARE_AGENT_BRIEFS.md` extended for tasks 3 and 4.

## What has to be built

1. `scripts/run_agent_flight.sh`: an ARM argument (`ours` | `waypoints`) selecting the real-only checkpoint
   and, for waypoints, `SNMVP_PIN_DECODE_ONLY=1`; the mailbox refuses `approve` in the waypoint arm.
2. Judges: `agent_judges.py` with `mannequin`, `orbit`, `compound_then_mannequin` (task 2 uses the existing
   compound judge + clearance).
3. `scripts/run_agent_matrix.sh`: N trials of (task, arm), each launching the flight and a reviewer with the
   task's brief file, collecting the per-trial record into `experiments/rung3/agent_eval/<task>_<arm>.tsv`.
4. Brief files under `experiments/rung3/briefs/`: task1..4, with `{BACKEND}` and `{DIR}` placeholders.
5. Cloud-page builder per task (mannequin page exists; orbit and compound pages via build_traj_page).

## Risks

- Sonnet variance is large (3/15 on the compound task before the calibration sheet). Five trials separate the
  arms only when the gap is big; the orbit and the double gate are where the gap should be big, since
  waypoints are open-loop near posts.
- Task knowledge in the briefs must be identical across arms; the mannequin brief's "not visible from the
  start" is kept for both.
- The real-only pin has flown no agent flight yet; task 1 in ours is the first check, before the matrix.
