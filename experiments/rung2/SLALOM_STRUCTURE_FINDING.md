# Rung 2b — discovered structure helps a bottlenecked policy (two-obstacle slalom)

Result (2026-07-22): the single-obstacle finding replicates on a strictly harder
task whose discoverable structure is higher-dimensional. On a two-obstacle
**slalom** reach (obstacles on opposite sides of the start→target line force an
S-curve detour) with real robosuite arm kinematics, a coherence-discovered
structure frame + prior raises held-out success from **0.021 (scratch flow)** to
**0.294 (structure)**; a random-frame control stays at **0.035**. The discovered
frame grows from **3 pins** (single obstacle) to **5 pins**, and lateral energy
shifts from ω1-dominated to **ω2-dominated** — the second-harmonic signature of
the S-weave. Same methodology and machinery as `OBSTACLE_STRUCTURE_FINDING.md`.

## Task

Planar EE reach on robosuite `Lift` (Panda), OSC_POSE, low-dim state
(`MUJOCO_GL=egl`). Each scene: target displacement (radius 0.16–0.24 m, angle
uniform in ±150° — the rear cone is excluded because the arm's rearward planar
reach is short) and **two virtual obstacles** on the start→target line at
longitudinal fractions ~0.24–0.34 and ~0.62–0.72, placed on **opposite** lateral
sides (offset ±0.03–0.06 m, radius 0.03–0.05 m). A straight reach hits both; a
single-bend detour clears at most one — the demo must weave (S-curve). The global
weave orientation (which side first) is chosen per scene (bimodal). Success
(offline) = final EE displacement within 0.03 m of target AND every trajectory
point outside **both** obstacle disks.

Demos: plan a 2-D path — smoothstep longitudinal profile reaching the target by
72% of the horizon then dwelling (so the proportional controller settles with
~zero arrival velocity), plus two opposite endpoint-vanishing Gaussian clearance
bumps (width 0.08 in progress-fraction to limit cross-talk; over-cleared by
obstacle-radius + 0.10 m to absorb OSC tracking lag) centered at each obstacle's
longitudinal fraction; small lateral wiggle. OSC-track on the real arm; store the
achieved EE-(x,y) delta chunk (H=32). 120 scenes × 8 demos. Demo success ceiling
**0.596** (lower than the single-obstacle 0.82: two clearances plus the weave).

## Structure discovery (over demos)

Identical procedure to the single-obstacle task (canonicalize to +x reach, rFFT
per axis, cross-demo phase coherence γ = |mean exp(iφ)| with energy-floor 0.10,
coherence 0.6, magnitude-CV 0.15; demos-as-bodies trick). Discovered frame S_F
(**5 pins**):

| axis | ω | mode | mag | meaning |
|---|---|---|---|---|
| progress | 0 | mod2π | ✓ | net displacement / endpoint |
| progress | 1 | mod2π | ✓ | longitudinal velocity profile (smoothstep ramp+dwell) |
| lateral | 1 | mod2π | ✓ | primary bend |
| lateral | 2 | mod2π | ✓ | **S-weave second harmonic (the slalom)** |
| lateral | 3 | mod2π | ✗ | third lateral harmonic |

Lateral energy fraction: ω1 = 0.31, **ω2 = 0.42**, ω3 = 0.21 (vs single-obstacle
where ω1 dominated). Coherence γ ≈ 1.0 at ω1/ω2/ω3. The harder task demands — and
coherence finds — a richer, higher-harmonic subspace.

## Arms (single embodiment; 100 train scenes, 20 held-out; 3 seeds)

Same as single-obstacle: **A** scratch flow (no structure); **F** S_F pinned into
source noise + scene→invariant prior; **Frand** random-frame pin + its prior.
Autograd MLP executor, 8000 steps each. Scene descriptor is 7-D
([radius, o1_cx,o1_cy,o1_r, o2_cx,o2_cy,o2_r], canonical).

## Result (held-out slalom success, pooled over 3 seeds)

| arm | success |
|---|---|
| F (discovered structure) | **0.294** |
| A (scratch) | 0.021 |
| Frand (random frame) | 0.035 |

F − A = +0.273; F − Frand = +0.259; demo ceiling 0.596. Per-seed F:
0.250 / 0.306 / 0.325; A: 0.013 / 0.013 / 0.037.

Interpretation: F/ceiling = 0.49, matching the single-obstacle ratio (0.383/0.82
= 0.47) — the structure recovers the same fraction of the achievable performance
on a harder task requiring a 5-mode, ω2-dominated subspace. Scratch near-fails the
S-weave (the policy is the bottleneck); F ≫ Frand isolates the learned coherence
frame from an arbitrary pin.

## Caveats

- Same as single-obstacle: scratch is a weak baseline (the ratio overstates; the
  load-bearing claims are absolute F = 0.29 and F ≫ Frand); offline geometric
  success on real-kinematics demo chunks, not closed-loop robosuite execution;
  demos are not bit-reproducible (robosuite `reset` randomizes pose), scene list
  seeded `default_rng(0)`, F/A/Frand ordering stable across 3 seeds.
- The structure test is deterministic given a fixed `data_slalom/Panda.npz` and
  fixed seeds (same code path as the single-obstacle test, which re-ran
  bit-identically).

## Reproduce

Box `~/code/source-noise-mvp/experiments/rung2/`. Env: robosuite==1.4.1,
mujoco==2.3.7, `MUJOCO_GL=egl`, python 3.11.

    # 1. collect slalom demos (~25 min; writes data_slalom/Panda.npz)
    bash run_slalom.sh
    # 2. structure vs baseline (CPU/autograd; ~15 min; writes slalom_result.json)
    ../../.venv/bin/python structure_test_slalom.py

Scripts: `collect_slalom.py`, `structure_test_slalom.py`, `run_slalom.sh`.
Result: `slalom_result.json`.
