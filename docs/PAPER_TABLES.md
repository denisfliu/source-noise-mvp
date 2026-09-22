# Paper tables (drafted 2026-09-22)

One number per cell. Sim success = per-flight conjunction of transit through the correct aperture in the correct
direction, no wrong-direction pass, and clearance >= 0.18 m to the gate cloud (goal box NOT used: the pilot's own
demonstrations end inside it only 23/50 and 9/50). Real success = the pilot's eyewitness "through the gate" with no
contact (clearance is unmeasured on hardware: the trajectory records before dronevla2.0 c933466 were setpoints).

## Table 1. Real-only arms on the single-gate tasks

| Arm | Data | Left, sim (n=10) | Right, sim (n=10) | Left, real (n=5) | Right, real (n=5) |
|---|---|---|---|---|---|
| pi0 (plain fine-tune) | 100 real demos | 3/10 | 4/10 | 4/5 | 4/5 |
| ours (pin + head) | 100 real demos | 0/10 | 5/10 | 5/5 | 3/5 |
| pi0, 25-step replan | 100 real demos | 6/10 | 3/10 | - | - |
| ours, 25-step replan | 100 real demos | 5/10 | 9/10 | - | - |

Sim rows at a 50-step replan unless marked; the 25-step rows are the same checkpoints re-observing every 2.5 s
(RESEARCH_LOG 2026-09-22). Real rows: 2026-09-15/18 sessions, apc 50, judged by eyewitness + measured poses.

## Table 2. Movement tests, real-only pin only

Authored routes flown through the real-only pin (100 real demonstrations of straight gate passes; no orbit, no
figure-eight, no compound in its data). Sim: 5 trials per cell; tracking = median distance of the flown path to
the drawn route; clean = clearance >= 0.18 m. Real: the collaborator's flights, command logs only, replan poses
against the route; clearance unmeasured.

| Route | Where | Tracking (m) | Completed | Clearance-clean | Note |
|---|---|---|---|---|---|
| orbit around the right gate, 0.9 m radius | sim | 0.05 | 5/5 full loops | 0/5 | clearance lost only in the post-handback hover drift (~69 s), not on the loop |
| figure-eight through left + centre gates | sim | 0.05-0.07 | 5/5 | 5/5 | |
| hand-drawn compound, left then centre gate | sim | 0.12 | 3/5 reached the goal | 1/5 | cuts the centre-gate corner: no centre-gate demos in its data |
| figure-eight, fig8_denis3 (10.5 m) | real, 2 attempts | 0.10 (max 0.29 / 0.69) | 2/2 reached the handback | unmeasured | flight 2 dipped to z 0.83 |
| orbit around the right gate, 1.3 m, 1.5 loops | real, ~5 attempts | 0.06-0.15 on the 2 that flew | 2 reached the handback | unmeasured | the others aborted or restarted; video needed |

## Table 3. Agent-in-the-loop (Sonnet + primitives), to fill in

Both arms use the real-only checkpoint and the same primitives, brief and budget; ours flies the agent's move
through the pin, waypoints flies the decoded track with no denoising. Sim: 5 trials per cell (10 for claims).
Real: n per cell to be decided by battery count.

| Task | Budget | ours, sim | waypoints, sim | ours, real | waypoints, real |
|---|---|---|---|---|---|
| find the mannequin (180-degree start) | 14 | 1/5 | 0/5 | __/_ | __/_ |
| left gate, then centre gate, hover over the penguin | 14 | 0/5 | 0/5 | __/_ | __/_ |
| one full circle around the centre gate | 20 | 2/5 | 3/5 | __/_ | __/_ |
| left gate, then find the mannequin | 24 | 1/5 | 0/5 | __/_ | __/_ |

Sim columns: Sonnet reviewer, trials 11-15 of 2026-09-23, identical briefs (experiments/rung3/briefs, round 3 + R7).
Partial credit behind the numbers: ours crossed the left gate 5/5 on the double task (waypoints 1/5) and found the
mannequin (within 2.5 m) 3/5 + 3/5 across the two search tasks (waypoints 2/5 + 0/5); waypoints completed the
orbit loop 5/5 with cleaner radii. Clearance-clean flights: ours 8/20, waypoints 7/20.

Success per task: mannequin = within 2.5 m and facing within 30 deg for the final 20 steps, clearance-clean;
double = compound judge (both gates in order, dwell) + clearance-clean; orbit = 360 deg of bearing about the centre
gate without passing through it (r >= 0.3 m) or detouring (r <= 3 m), clearance-clean; left+mannequin = left transit route-clean + clearance-clean + the
mannequin criterion. Secondary columns to add per cell: decisions used, overrides, median reviewer seconds,
min clearance. 
