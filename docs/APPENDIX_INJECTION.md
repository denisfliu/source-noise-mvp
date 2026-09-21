# Appendix data: putting a sketch into a flow, and why each way fails (collected 2026-09-20)

All runs in the simulator (Gaussian-splat reconstructions of the flight room), 5 trials per cell, NCH 14 chunks,
APC 50, single training seed. Two data regimes: **real-only** (100 real demonstrations of the left and right
gates; pin `gate_pin_joint_realonly`, plain pi0 `gate_scratch_real`) and **mixed** (the same 100 real plus
200 simulated demonstrations incl. the center tasks; pin `gate_pin_joint_gmsig3`, plain pi0 `gate_scratch3`).
Scripts: `scripts/run_vproj_compare.sh`, `run_vproj_sigma.sh`, `run_badsketch.sh`, `run_badsketch_mixed.sh`,
`run_badsketch_vproj_sigma.sh`; raw scores and ledgers in `experiments/rung3/vproj/`; cloud pages
`experiments/rung3/viz/vproj_compare.html` and `badsketch.html`.

## The four ways

| Arm | What it does | Training |
|---|---|---|
| **pin** (ours) | sketch command c written into the source noise; flow trained with the pin (sigma-conditioned) | yes |
| **SDEdit** t0 = 0.5 | sampler starts at t0 from t0·z + (1 − t0)·a_sketch; unmodified flow | no |
| **inject** | c written into the source noise of the plain pi0; unmodified flow | no |
| **v-proj** s | as inject, and at every Euler step v ← (I − UUᵀ)v + s·UUᵀv; s = 0 carries c exactly, s = 1 is inject | no |

Metrics: tracking = median distance from the flown path to the sketch polyline; clean = flights with
clearance ≥ 0.18 m to the gate cloud; realism = AUC of a window classifier against the pilot's real
demonstrations on the sketch-active segment (0.5 = indistinguishable; the pilot against itself scores 0.55);
zero-acc = fraction of 1-s windows with near-zero acceleration (the pilot: 0.40; the planner's staircase: 0.93).

## A. Authored routes, real-only regime (orbit, figure-eight, hand-drawn compound)

Tracking (m) / clearance-clean:

| Cell | pin σ=0 | SDEdit | inject | v-proj s=0 | s=0.1 | s=0.3 | s=0.5 |
|---|---|---|---|---|---|---|---|
| orbit | 0.05 / 0/5 (post-handback drift) | 0.06 / 1/5 | 0.12 / 1/5 | 0.011 / 5/5 | 0.010 / 5/5 | 0.024 / 5/5 | 0.039 / 2/5 |
| figure-eight | 0.06 / 5/5 | 0.05 / 1/5 | 0.15 / 0/5 | 0.012 / 5/5 | 0.020 / 5/5 | 0.048 / 0/5 | 0.092 / 0/5 |
| compound (goal reached) | 0.12 / 1/5 (3/5) | 0.08 / 0/5 (5/5) | 2.8 (never) | 0.013 / 5/5 (5/5) | 0.043 / 5/5 (1/5) | 0.106 / 5/5 (0/5) | 1.28 / 5/5 (0/5) |

Realism, AUC vs the pilot (lower = closer):

| Cell | pin | SDEdit | inject | s=0 | s=0.1 | s=0.3 | s=0.5 |
|---|---|---|---|---|---|---|---|
| orbit | 0.89 | 0.78 | 0.95 | 0.80 | 0.82 | 0.82 | 0.83 |
| figure-eight | 0.84 | 0.74 | 0.96 | 0.78 | 0.74 | 0.81 | 0.82 |
| compound | 0.88 | 0.84 | 0.95 | 0.78 | 0.81 | 0.86 | 0.90 |

Kinematics (v95 m/s / zero-acc / jerk95 / tilt99 deg / body rate99 deg/s; pilot 0.68 / 0.40 / 2.7 / 5.7 / 22):
pin 0.37 / 0.20-0.27 / 4.0-4.3 / 5-6 / 36-46; SDEdit 0.30 / 0.57-0.71 / 2.1-3.0 / 4 / 25-30;
inject 1.2-2.1 / 0.14-0.26 / 6-23 / 20-37 / 130-218; v-proj s=0 0.35-0.47 / 0.40-0.47 / 3.2-4.2 / 5-6 / 35-47;
s=0.5 0.66-0.95 / 0.34-0.41 / 3.7-8.3 / 10-16 / 74-117.

Why each fails: **inject** runs away (the unpinned flow's velocity along U drives the command off).
**SDEdit** keeps the sketch's velocity staircase and clips gates. **v-proj s > 0** leaks the flow's disagreement
back in, monotonically (tracking 1 cm → metres), never toward the pilot. **v-proj s = 0** is the sketch verbatim
with a pilot-like speed profile; it cannot deviate. **pin σ = 0** (real-only) tracks at 5-12 cm and cuts the
center gate corner: its residual is jerkier than the pilot and has never seen the center gate.

## B. Bad sketches (the 4-click compounds whose own line passes 0.07 m / 0.04 m from the center post)

Both gates latched / clearance-clean / min clearance (m):

| Regime | Arm | L→C (0.07 m) | R→C (0.04 m) |
|---|---|---|---|
| mixed | pin σ=0 | 5/5 / 0/5 / 0.035-0.15 center post | 0/5 / 0/5 / 0.003-0.008 center post |
| mixed | pin σ=0.5 | **4/5 / 4/5 / 0.19-0.26** | 0/5 / 0/5 / 0.004-0.12 |
| mixed | v-proj s=0 | 5/5 / 0/5 / 0.066 (= the sketch's own) | 5/5 / 0/5 / 0.003-0.03 right gate east post |
| mixed | SDEdit | 1/5 / 0/5 / 0.002-0.14 | 2/5 / 0/5 / 0.002-0.05 |
| real-only | pin σ=0 | 1/5 / 0/5 / 0.01-0.08 center post | 3/5 / 0/5 / 0.005-0.05 |
| real-only | pin σ=0.5 | 0/5 / 3/5 / leaves the sketch (0.02-0.22 at the left gate) | 0/5 / 0/5 / leaves the sketch |
| real-only | v-proj s=0 | 5/5 / 0/5 / 0.066 x5 | 5/5 / 0/5 / 0.008-0.020 |
| real-only | SDEdit | 0/5 / 0/5 / 0.001-0.08 left gate | 2/5 / 0/5 / 0.003-0.04 |

Why each fails: **v-proj** reproduces the sketch's flaw exactly, in both regimes. **SDEdit** contacts the first
gate. **pin σ = 0** grazes like the sketch. **pin σ = 0.5** is the only arm that clears the 0.07 m sketch, and
only with the mixed residual: the trained slack spends the residual's scene knowledge, and the real-only
residual has none about the center gate, so it leaves the sketch instead. The 0.04 m sketch is bad in two
places (center post and the right gate's east post) and defeats every arm.

## C. Partial projection on the bad sketches (s = 0.1, 0.3, 0.5)

Both gates latched / clearance-clean / min clearance (m) / tracking (m):

| Regime | s | L→C (0.07 m) | R→C (0.04 m) |
|---|---|---|---|
| mixed | 0 | 5/5 / 0/5 / 0.066 / 0.014 | 5/5 / 0/5 / 0.003-0.03 / 0.017 |
| mixed | 0.1 | 5/5 / 5/5 / 0.182-0.190 / 0.09 | 0/5 / 0/5 / 0.002-0.007 / 0.03 |
| mixed | 0.3 | 5/5 / 4/5 / 0.17-0.21 / 0.42 (flies its own route) | 1/5 / 0/5 / 0.004-0.025 / 0.13 |
| mixed | 0.5 | 0/5 (1/2) / 2/5 / 0.11-0.22 / 0.82 (leaves) | 0/5 / 0/5 / 0.002-0.04 / 0.98 (leaves) |
| mixed | pin σ=0.5 (for reference) | 4/5 / 4/5 / 0.19-0.26 / 0.13 | 0/5 / 0/5 / 0.004-0.12 / 0.15 |
| real-only | 0.1 | 1/5 / 0/5 / 0.09-0.13 / 0.05 | 1/5 / 0/5 / 0.006-0.03 / 0.025 |
| real-only | 0.3 | 2/5 / 3/5 / 0.12-0.24 / 0.27 (leaving) | 0/5 / 0/5 / 0.005-0.015 / 0.19 |
| real-only | 0.5 | 0/5 / 1/5 / 0.11-0.19 / 0.73 (leaves) | 0/5 / 0/5 / 0.002-0.012 / 0.60 (leaves) |

Reading: a small leak of the plain flow's velocity (s = 0.1) on the **mixed** flow moves the path off the
0.07 m sketch by exactly enough to clear the post (0.18-0.19 m, at the clean threshold) while keeping both
gates, at 9 cm tracking; the pin's trained slack clears by more (0.19-0.26 m) at 13 cm tracking with one
flight lost. So on a flow with scene knowledge the training-free slack recovers most of the pin's dodge on
this sketch; the trained slack buys a few centimetres more clearance. Larger s makes the flow fly its own
route (s = 0.3: both gates but 42 cm off the sketch; s = 0.5: leaves). On the **real-only** flow no s helps:
s = 0.1 lifts clearance only to 0.09-0.13 m, s ≥ 0.3 leaves the sketch. The 0.04 m sketch defeats every s in
both regimes. In short: whatever slack mechanism is used, it can only spend knowledge the residual has.

## D. Worse sketches (mixed regime): the line grazes, enters, or misses the post

The 4-click L→C with its center waypoint slid west so the drawn line passes the center gate's west post at
0.02 m (w2), goes through it (w3, 0.009 m), or passes on the wrong side of the post, outside the aperture (w4).
Files `sketch_cmpl_min4_w{2,3,4}[s].json`; script `scripts/run_worsesketch.sh`; page `viz/worsesketch.html`.

Both gates latched / clearance-clean / min clearance (m) / tracking (m):

| Arm | w2 (0.02 m) | w3 (through the post) | w4 (wrong side) |
|---|---|---|---|
| pin σ=0 | 4/5 / 0/5 / 0.002-0.10 / 0.06 | 1/5 / 0/5 / 0.004-0.07 / 0.06 | 0/5 / 0/5 / 0.003-0.06 / 0.06 |
| **pin σ=0.5** | 4/5 / 3/5 / 0.16-0.26 / **0.12** | 4/5 / 4/5 / 0.18-0.28 / **0.12** | 4/5 / 3/5 / 0.16-0.24 / **0.14** |
| v-proj s=0 | 3/5 / 0/5 / 0.014-0.016 / 0.014 | 0/5 / 0/5 / 0.003-0.006 / 0.015 | 0/5 / 0/5 / 0.042 / 0.014 |
| v-proj s=0.1 | 5/5 / 0/5 / 0.12-0.13 / 0.11 | 5/5 / 0/5 / 0.08-0.10 / 0.12 | 5/5 / 0/5 / 0.02-0.04 / 0.11 |
| v-proj s=0.3 | 5/5 / 5/5 / 0.18-0.22 / **0.45** | 5/5 / 5/5 / 0.18-0.24 / **0.45** | 5/5 / 5/5 / 0.20-0.24 / **0.45** |
| SDEdit 0.5 | 0/5 / 0/5 / 0.01-0.13 / 0.07 | 0/5 / 0/5 / 0.02-0.16 / 0.07 | 0/5 / 1/5 / 0.06-0.22 / 0.06 |

Reading: this is where the trained slack and the leak separate. The pin at σ = 0.5 clears the post on all three
sketches, including the one drawn through it and the one drawn around the wrong side, while staying 12-14 cm
from the sketch: it makes the smallest correction that clears. The projection at s = 0.1 keeps the sketch
(11 cm) but does not clear (0.02-0.13 m); at s = 0.3 it clears every time but is 45 cm off the sketch, which
is the plain flow flying its own compound route with the sketch as a loose suggestion. There is no s that
gives both. SDEdit never latches both gates. Exact carry (s = 0) does what it is told: into the post.

## E. Compliance: how far each arm moved the command it was given

`experiments/rung3/compliance.py` replays the sketch tracker exactly as the server ran it and, at every
sketch-active replan, compares the command that was issued with the command realized by the 50 steps
actually flown: E_comp = ‖(Uᵀâ − c)/σ_c‖_rms, in units of the training corpus's per-coordinate command
standard deviation. The rollouts log positions only, so the yaw channel of each horizon band is zeroed on
both sides and the score runs over the 12 translation coordinates of the 16. Median over 5 flights.

| Arm | 0.07 m sketch | w2 (0.02 m) | w3 (through the post) | w4 (wrong side) |
|---|---|---|---|---|
| v-proj s=0 (exact carry) | 0.000 | 0.000 | 0.000 | 0.000 |
| v-proj s=0.1 | 0.116 | 0.108 | 0.100 | 0.096 |
| pin σ=0 | 0.118 | 0.124 | 0.129 | 0.127 |
| SDEdit t0=0.5 | 0.186 | 0.187 | 0.186 | 0.191 |
| **pin σ=0.5** | **0.222** | **0.228** | **0.258** | **0.273** |
| **v-proj s=0.3** | **0.350** | **0.307** | **0.318** | **0.320** |

(real-only regime, 0.07 m sketch: pin σ=0 0.193, pin σ=0.5 0.316, v-proj s=0 0.000.)

Reading: s = 0 scores exactly zero, which is the metric's sanity check — the carry identity holds to
floating point. The two arms that clear the worse sketches are the last two rows, and the pin's correction
is the *smaller* one: 0.23-0.27 σ_c against the projection's 0.31-0.32 σ_c, with the same or better
clearance (3-4/5 vs 5/5) at a third of the tracking error (0.12-0.14 m vs 0.45 m). The pin also *grades* its
deviation with how bad the sketch is (0.222 → 0.228 → 0.258 → 0.273 as the line moves into and past the
post) while the projection's leak is flat (0.35, 0.31, 0.32, 0.32): the leak is a fixed fraction of the
flow's disagreement, the trained slack is a correction sized to the problem. Note also pin σ=0 is not zero
(0.12): a trained flow follows the command closely but not exactly, which is the price of the residual
being free to shape the motion.
