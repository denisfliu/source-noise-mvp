# Hardware experiment protocol (2026-09-01) — fills the [hardware slot] in the paper

Every phase names its metric, n, the paper element it fills, and its abort criterion.
Serve stack for all phases: xswap checkpoint, mocap state, ground-station GPU, intent
cockpit connected (gate mode = the safety veto), carrot=20 on all sketches, kill switch.
PRE-REGISTER n and success thresholds before flying; log every trial (mocap traj + onboard
video + CLOG) whether it succeeds or not.

## Phase R0 — prerequisites (no flight)
- Recalibrate sigma on real frames: sigma_phase_probe on data_gate_real -> real sigma map.
  (Known gap: sim map's error-ranking does not transfer, corr ~0.)
- Verify mocap->scene-cloud registration against the splat (fly-less check: wand walks).
- Latency budget: replan round-trip < 400 ms sustained over WiFi.

## Phase R1 — closed-loop atomics (fills Table 1's real rows)
- Tasks: left gate, right gate, center-from-left, center-from-right (room in the matching
  configuration per task). THE CENTER TASKS ARE THE HEADLINE ROWS: they have ZERO real
  demonstrations — real success there is the purest sim-to-real task-transfer claim in
  the paper.
- Arms: Ours (xswap) vs fine-tuned pi0 (scratch3). n=10 trials/task/arm, start jitter
  matched to the training distribution.
- Metric: collision-free Success (mocap judge, same route-clean + 0.18 m rule).
- Abort: any contact, or 3 consecutive aborts by the human gate.
- This is the first closed-loop real data; expect the open-loop projections to be
  optimistic. Whatever the number is, it goes in the paper.

## Phase R2 — command fidelity on hardware (fills the compliance paragraph)
- Rotation dose-response live: at mid-corridor, command +/-15 deg via the cockpit;
  measure realized heading change from mocap. n=20 doses. Compare gain to 0.89 (sim/real
  open-loop).
- Tempo: same route at 0.6x/1.0x/1.5x sketch pace, n=5 each; realized vs commanded speed.
- Follow error: per-replan |U^T a_exec - c| from CLOG + mocap; compare to 0.08.

## Phase R3 — sketch missions (fills Table 3's real rows)
- Hand-drawn compound (two-gate) sketch in the room's compound configuration. n=10.
- Minimal 4-5 click sketch, same task. n=10.
- Success: collision-free, ordered gates, route-clean. Prediction from sim: majority
  success; margins are the risk (draw with >=0.35 m margins).

## Phase R4 — relocated gate (the strongest real claim if it lands)
- Physically move one gate (new position + yaw), re-scan OR update registration only
  (test both: the sketch needs only the new aperture pose from a tape measure — this is
  the honest 'no re-scan' variant worth reporting).
- Auto 4-point sketch through the new pose. n=10.

## Phase R5 — novel programs (figure candidates)
- Orbit (r=0.9 around a gate) and figure-eight in open space. n=5 each, wide margins.
- Metric: tracking error vs sketch + completion; video for the paper site.

## Ordering and stop rules
R0 -> R1 (pi0 arm first: it establishes the room baseline and shakes out ops) -> R2 ->
R3 -> R4 -> R5. Any phase failing its abort rule stops the campaign for diagnosis; do not
skip forward past a failed phase. Per the statistics rule, n=10 rows are claim-tier;
n=5 rows are labeled screens in the paper.

## What each phase fills
R1 -> Table 1 real columns + the fine-tuning comparison sentence in the abstract.
R2 -> compliance/verb numbers with 'on hardware' scope.
R3/R4 -> Table 3 real rows; R4 is the flagship real claim.
R5 -> figures + website video.
