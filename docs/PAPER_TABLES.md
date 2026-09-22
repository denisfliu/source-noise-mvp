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

## Table 2. Movement tests (authored routes through the pin)

Sim, 5 trials per cell unless stated; tracking = median distance of the flown path to the drawn route; clean =
clearance >= 0.18 m. Real rows are the collaborator's flights (command logs only; replan poses vs the route).

| Route | Arm / data | Where | Tracking (m) | Route completed | Clearance-clean | Note |
|---|---|---|---|---|---|---|
| orbit around the right gate (0.9 m) | ours, real-only | sim | 0.05 | 5/5 full loops | 0/5 | clearance lost only in the post-handback hover drift (~69 s) |
| figure-eight through left + centre | ours, real-only | sim | 0.05-0.07 | 5/5 | 5/5 | |
| hand-drawn compound L->C (goal) | ours, real-only | sim | 0.12 | 3/5 reached the goal | 1/5 | cuts the centre-gate corner: no centre demos in its data |
| hand-drawn compound L->C | ours, mixed (gmsig3) | sim | ~0.07 | 5/5 route-clean | 5/5 | |
| hand-drawn compound R->C | ours, mixed (gmsig3) | sim | ~0.07 | 5/5 route-clean | - | |
| hand-drawn sketches, pooled | ours, mixed (gmsig3) | sim | - | - | 17/20 | |
| 4-click minimal compound, sigma 0 / 0.5 | ours, mixed (gmsig3) | sim (n=10) | - | - | 2/10 / 9/10 | margin is the portability budget |
| bad sketch, line 0.07 m from the post, sigma 0.5 | ours, mixed | sim | 0.13 | 4/5 both gates | 4/5 | v-proj s=0 follows the flaw: 0/5 clean; SDEdit 0/5 |
| bad sketch, line 0.04 m from the post | every arm | sim | - | - | 0/5 | bad in two places; defeats every arm |
| worse sketches w2 / w3 / w4, sigma 0.5 | ours, mixed | sim | 0.12 / 0.12 / 0.14 | 4/5 / 4/5 / 4/5 | 3/5 / 4/5 / 3/5 | the line grazes / enters / passes the wrong side of the post |
| all sketch rows, pooled two seeds | ours (xswap s42 + s7) | sim | - | 49/50 route-clean | geometry-bound | |
| figure-eight (fig8_denis3, 10.5 m) | ours, real-only | real (2 attempts) | 0.10 (max 0.29 / 0.69) | both reached the handback | unmeasured | flight 2 dipped to z 0.83 |
| orbit around the right gate (1.3 m, 1.5 loops) | ours, real-only | real (~5 attempts) | 0.06-0.15 on the 2 that flew | 2 reached the handback | unmeasured | the others aborted or restarted; video needed |
| rotation verb as aim correction | ours, mixed | sim + real anchors | - | heading error 10-20 deg -> 3-5 deg | - | dose gain 0.76 |
| compliance, pin sigma 0.5 vs v-proj s=0.3 | ours, mixed | sim | 0.12-0.14 vs 0.45 | - | 3-4/5 vs 5/5 | pin's command correction 0.22-0.27 sigma_c vs 0.31-0.35 |

## Table 3. Agent-in-the-loop (Sonnet + primitives), to fill in

Both arms use the real-only checkpoint and the same primitives, brief and budget; ours flies the agent's move
through the pin, waypoints flies the decoded track with no denoising. Sim: 5 trials per cell (10 for claims).
Real: n per cell to be decided by battery count.

| Task | Budget | ours, sim | waypoints, sim | ours, real | waypoints, real |
|---|---|---|---|---|---|
| find the mannequin (180-degree start) | 14 | __/5 | __/5 | __/_ | __/_ |
| left gate, then centre gate, hover over the penguin | 14 | __/5 | __/5 | __/_ | __/_ |
| one full circle around the centre gate | 20 | __/5 | __/5 | __/_ | __/_ |
| left gate, then find the mannequin | 24 | __/5 | __/5 | __/_ | __/_ |

Success per task: mannequin = within 1.25 m and facing within 20 deg for the final 20 steps, clearance-clean;
double = compound judge (both gates in order, dwell) + clearance-clean; orbit = 360 deg of bearing about the centre
gate at 0.8-2.0 m radius, clearance-clean; left+mannequin = left transit route-clean + clearance-clean + the
mannequin criterion. Secondary columns to add per cell: decisions used, overrides, median reviewer seconds,
min clearance. Reference points already flown with the mixed pin (gmsig3): mannequin 1/4 attempts, penguin 1/1,
double-gate Sonnet 3/15, Opus 1/1.
