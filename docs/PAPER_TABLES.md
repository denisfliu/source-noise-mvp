# Paper tables (drafted 2026-09-22)

One number per cell. Sim success = per-flight conjunction of transit through the correct aperture in the correct
direction, no wrong-direction pass, and clearance >= 0.18 m to the gate cloud (goal box NOT used: the pilot's own
demonstrations end inside it only 23/50 and 9/50). Real success = the pilot's eyewitness "through the gate" with no
contact (clearance is unmeasured on hardware: the trajectory records before dronevla2.0 c933466 were setpoints).

## Table 1. Real-only arms on the single-gate tasks

| Arm | Data | Left, sim | Right, sim | Left, real (n=5) | Right, real (n=5) |
|---|---|---|---|---|---|
| pi0 (plain fine-tune) | 100 real demos | 5/20 | 9/20 | 4/5 | 4/5 |
| ours (pin + head) | 100 real demos | 0/10 | 5/10 | 5/5 | 3/5 |
| pi0, 25-step replan | 100 real demos | 44/100 | 45/100 | - | - |
| ours, 25-step replan | 100 real demos | 60/100 | 81/100 | - | - |

Sim rows at a 50-step replan unless marked; the 25-step rows are the same checkpoints re-observing every 2.5 s
(RESEARCH_LOG 2026-09-22). 25-step rows n=100: ten batches of ten, one server process and one source-noise seed per batch (both servers take
SNMVP_NOISE_SEED; a restarted server at the same seed replays its trials). Per-batch successes: pi0 left
6,4,2,6,3,4,7,3,5,4; pi0 right 3,3,5,7,4,4,6,6,4,3; ours left 5,7,5,8,7,8,7,5,5,3; ours right 9,10,9,7,8,7,7,7,8,9.
Standard errors ±5 (binomial). Failure modes at 25-step: ours left = 19 route-dirty + 34 grazes, ours right = 19
missed transits and zero grazes; pi0 left = 39 missed transits, pi0 right = 31 missed + 28 grazes. 50-step rows
n=20 (pi0) / n=10 (ours); no further 50-step runs (Denis, 2026-09-24). Real rows: 2026-09-15/18 sessions, apc 50, judged by eyewitness + measured poses.

## Table 2. Movement tests, real-only pin only

Authored routes flown through the real-only pin (100 real demonstrations of straight gate passes; no orbit, no
figure-eight, no compound in its data). Sim: 10 trials per cell for the orbit and figure-eight (5 + 5, 2026-09-24), 5 for the compound; tracking = median distance of the flown path to
the drawn route; clean = clearance >= 0.18 m. Real: the collaborator's flights, command logs only, replan poses
against the route; clearance unmeasured.

| Route | Where | Tracking (m) | Completed | Clearance-clean | Note |
|---|---|---|---|---|---|
| orbit around the right gate, 0.9 m radius | sim (n=10) | 0.065 (0.037-0.081) | 10/10 full loops | 3/10 | clearance lost only in the post-handback hover drift (~69 s), not on the loop |
| figure-eight through left + centre gates | sim (n=10) | 0.061 (0.040-0.072) | 10/10 | 10/10 | |
| hand-drawn compound, left then centre gate | sim | 0.12 | 3/5 reached the goal | 1/5 | cuts the centre-gate corner: no centre-gate demos in its data |
| figure-eight, fig8_denis3 (10.5 m) | real, 2 attempts | 0.10 (max 0.29 / 0.69) | 2/2 reached the handback | unmeasured | flight 2 dipped to z 0.83 |
| orbit around the right gate, 1.3 m, 1.5 loops | real, ~5 attempts | 0.06-0.15 on the 2 that flew | 2 reached the handback | unmeasured | the others aborted or restarted; video needed |

## Table 3. Agent-in-the-loop (reviewer + primitives), sim, 5 trials per cell

Both arms use the real-only checkpoint, the same primitives, brief and budget; ours flies the agent's move through
the pin (the agent may also approve the policy's own proposal), waypoints flies the decoded track with no
denoising and nothing to approve. Each cell: task done (the task's own judge) / clearance-clean / both. A gate touch
(clearance < 0.18 m) is a failure and is counted in the middle number. Success for the search tasks = reached the
figure (within 3 m, or within 0.5 m regardless of facing) and facing it within 45 deg over the final 20 steps
(Denis, 2026-09-23, from the videos). The double gate counts as done when both gates are crossed in order (route-clean); the hover over the penguin is not required (Denis, 2026-09-23). Its budget is 20 decisions from trial 51 onward (14 before).

| Task | Budget | Sonnet + ours | Sonnet + waypoints | Opus + ours | real (to fill) |
|---|---|---|---|---|---|
| find the mannequin (180-degree start) | 14 | 3/5 · 3/5 · **3/5** | 3/5 · 0/5 · **0/5** | 5/5 · 4/5 · **4/5** | __ |
| left gate, then centre gate, hover over the penguin | 14 | 0/5 · 0/5 · **0/5** | 0/5 · 3/5 · **0/5** | (not run on the final brief) | __ |
| one full circle around the centre gate | 20 | 5/5 · 2/5 · **2/5** | 4/5 · 4/5 · **3/5** | 5/5 · 3/5 · **3/5** | __ |
| left gate, then find the mannequin | 24 | 3/5 · 2/5 · **2/5** | 2/5 · 3/5 · **1/5** | 4/5 · 5/5 · **4/5** | __ |

Sonnet trials 11-15 and Opus trials 21-25 of 2026-09-23 (Opus t25 on the last task reached the figure at 0.8 m but
held 50 deg off axis). The double-gate row is from trials 41-45 on the final brief (in-the-opening state, no
reversing to line up, straight-then-yaw after the left gate), Sonnet only: partial credit there is left gate
crossed 4/5 and both gates in order 2/5 for ours, no gate crossed in 5/5 for waypoints (its three clean flights
never approached a post). Earlier double-gate cells on earlier briefs (trials 11-15, 31-35) are in the records but
not in the table. Every waypoint mannequin flight grazed a gate in a 0.15-0.17 m band.
