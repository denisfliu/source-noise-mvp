# Hardware results, aggregated (as of 2026-09-17)

Every real-drone experiment reported so far, in one place. Room: the three-gate mocap room; policies served
from manaan (`scripts/hw_serve.sh`), flown by the collaborator with the dronevla2.0 gate node; safety pilot
on the transmitter throughout. Per-flight pages with the pilot's eyewitness notes: `~/gate_flights/*_cloud.html`
on manaan. Point clouds of every scored flight: `experiments/rung3/viz/hw_flights.html`.

## How to read the numbers

Three sources, kept apart because they disagree in places:

- **Eyewitness**: the note the pilot entered after each flight ("flew through the gate and hovered over the
  penguin", "hit the left side", "crash"). The only record of contact.
- **Judge on measured poses**: `gate_success.py` on the drone's measured position at each replan (every 5 s at
  apc 50, every 0.8 s at apc 8) joined by straight chords. Settles transit and route; cannot see contact.
- **Goal box**: the simulator's box at (1.525, -0.615, 1.0) +- (0.3, 0.3, 0.5). The pilot's own demonstrations
  end at z 1.25-1.6 and enter it only 33/50 (left) and 25/50 (right), so it is reported but not used as success.

Clearance is unmeasured for every flight so far. The flight node's `traj_<trial>.npy` files were the setpoint
stream, not the drone (fixed in dronevla2.0 c933466, 2026-09-15); the saved per-flight pages' clearance numbers
are the commanded path's clearance. Flight records after the workstation pulls the fix will have both.

## Closed-loop atomics (apc 50 unless noted)

| Arm | Task | Date | n | Through the gate (eyewitness) | Contact (eyewitness) | Judge transit, route-clean | Goal box |
|---|---|---|---|---|---|---|---|
| pi0 baseline (scratch3) | left | 09-11 (apc 8) | 5 | 4 (01 02 04 05) | 2 (03 hit right side; 05 skimmed right side) | 4, all with the goal missed | 0 |
| pi0 baseline (scratch3) | right | 09-16 | 5 | 1 (05) | 4 (01 02 03 hit left side; 04 crashed in the gate) | 1 | 1 |
| pi0 baseline (scratch3) | center from left | 09-16 | 5 | 0 | 0 | 0 | 0 |
| pi0 baseline (scratch3) | center from right | 09-16 | 5 | 0 | 0 | 0 | 0 |
| real-only pin (100 real demos, no swap) | left | 09-15 | 5 | 5 | 0 | 5 | 3 |
| real-only pin | right | 09-15 | 5 | 3 (01 02 04) | 2 (03, 05 "crash") | 3 (01 03 04) | 1 |
| real-only pin | center from left | 09-16 | 5 | 0 | 0 | 0 | 0 |
| ours (xswap) | left | 09-11 (apc 8) | 5 | 0 | 0 | 0 | 0 |
| pi0 real-only (baseline_real) | left | 09-18 | 5 | 4 | 1 (05 crashed into the gate before traversal) | 4 | 3 |
| pi0 real-only (baseline_real) | right | 09-18 | 5 | 4 | 1 (05 crashed before traversal; 02 graze on the setpoint record) | 4 | 4 |
| mixed pin, no swap (gmsig3) | left | 09-18 | 5 | 3 | 5 (2 crashed before the gate, 3 after it) | 3 | 0 |

Notes per cell:

- **Baseline left.** By eye four of five traversed the gate, two touching the right post. On the 0.8-s poses
  the judge finds four transits; the earlier setpoint-record judge flagged wrong-direction passes that the
  measured poses do not show. Flights ran 50-160 s at ~1 m/s peak with long wandering (paths 17-40 m).
- **Baseline right.** Three flights hit the left (west) post, one crashed inside the gate, one clean transit.
  The 09-16 collaborator note on 01 says "hit", the setpoint judge says clean 0.30 m: contact is by eye.
- **Baseline center tasks.** Never attempts the center gate: goes straight to the goal (CFL 02 03 04; CFR 01-04)
  or flies the left-gate route (CFL 01 05); CFR 05 approached the center gate and missed it. Same failure
  family as the simulator's baseline CFR (wrong-task, goal-first).
- **Real-only pin left.** Five of five through the gate and hovering over the penguin, no contact, altitude
  1.45-1.65 m throughout. Gate crossing at ~17 s vs the pilot's 11.6 s median; flights 37-46 s vs 22.5 s.
- **Real-only pin right.** Three through. Right 02: "through the gate" by eye but no crossing in either
  record (pose (0.0, -0.9) then (-0.7, -0.2) five seconds later). Right 03: "crash" by eye, a normal transit
  ending in a hover in the record. Right 05: record ends on the floor at (-0.7, -2.6) 2.5 m from the previous
  pose. The 03/05 notes may be on the wrong flights; video needed.
- **Real-only pin center from left.** Approaches from the left and never reaches the gate, five of five, as
  expected: this arm has no center demonstrations (its sim CFL is 0/10 as well).
- **pi0 real-only, left and right (09-18).** The matched baseline for the real-only pin: four of five through on each gate, one crash before the gate on each. On these numbers the real-only pin (5/5, 3/5) and the real-only pi0 (4/5, 4/5) are not separated at n = 5.
- **Mixed pin without swap (gmsig3), left (09-18).** Sim 40/40 on both seeds; on the drone all five crashed. The command log shows the head aiming at the gate on the approach, then turning toward the goal right after the crossing and cutting back into the east post (min 0.015 m at (1.19, 0.37) in flight 03). Its center-from-left session (log only) shows the head commanding straight toward the goal and hovering at x ~ 1.0, never the center gate: the goal-first plan on real frames that the swap was introduced to fix (open-loop right-gate crossings 8/55 without it, 15-17/55 with it).
- **Ours (xswap) left, 09-11.** Setpoints were not followed (live z fell 0.15-0.2 m per replan while the
  commanded dz was ~0; x advanced 0.13 m against 1.5 m commanded): an execution-side failure, not a policy
  result. The no-swap arm the same afternoon (clog only) showed the same. **The paper's "ours" arm has no
  valid closed-loop hardware cell yet.**

## Sketch (maneuver) sessions: command logs only

No trajectory files reached manaan for these; what exists is the server's per-replan log (drone pose at each
replan, the 16-dim command, sigma). Tracking below is the replan pose's distance to the sketch polyline.

| Arm | Sketch | Date | Attempts | What the replan poses show |
|---|---|---|---|---|
| real-only pin | fig8_denis3 (figure-eight through left + center, 10.5 m) | 09-16 | 2 | both tracked the sketch at 0.10 m median (max 0.29 / 0.69) out to the handback; flight 2 dipped to z 0.83 |
| real-only pin | orbit_wide (1.3 m circle around the right gate, 1.5 loops, 14.0 m) | 09-16 | ~5 | two attempts tracked the loop at 0.06-0.15 m median and reached handback; the others show the log already in the handback state with the pose parked or on the floor (z 0.18): restarted trials or aborts, video needed |
| real-only pin | orbit (0.9 m), fig8 | 09-15 morning | 1 each | the descent problem (below): live z fell to the floor; no result |
| real-only pin | fig8_denis3 | 09-15 12:59 | test | 6 replans, sketch active, altitude held; not a flight |

## Sessions with no result (execution-side descent)

2026-09-11 afternoon (xswap, noswap) and 2026-09-15 morning (xswapc, real-only, orbit, fig8): the live z in the
command log fell 0.15-0.2 m per replan to the floor while the policy commanded level flight (decoded dz over 8
steps 0 to +0.06 m in every replan). The baseline flights bracketing those sessions held altitude, and the
2026-09-15 afternoon flights onward held 1.45-1.65 m, so something on the workstation changed between 13:00 and
15:40 on 09-15. Cause not identified on this box (workstation jsonl: published setpoint z vs mocap z).

## Pace and tracking (real-only pin, from the setpoint record vs measured poses)

The drone trails the setpoint stream by 0.10-0.16 m median and 0.35-0.6 m at chunk ends (about 85-90 % of the
commanded pace). Flights take 1.5-2x the pilot's time, matching the real-frame ledger (real-only pin v95 0.28 m/s
vs the pilot's 0.43). Head uncertainty on live frames sigma_serve 0.36-1.1 (the sigma map was fitted on
simulated frames).

## Rotation, tempo, compound and no-uncertainty rows

Not flown yet (run sheet `docs/real_experiments.tsv`).
