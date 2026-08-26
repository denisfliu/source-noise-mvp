# Rung 2 — discovered structure helps a bottlenecked policy (obstacle reach)

Result (2026-07-22): on a hard obstacle-reach task with real robosuite arm
kinematics, a coherence-discovered structure frame + prior raises held-out
success from **0.021 (scratch flow)** to **0.383 (structure)**; a random-frame
control stays at **0.008**. Tests original goal (b) — "does grounding a policy
in discovered action structure help" — in the regime where the policy is the
bottleneck. Companion negative at full-data pi0/LIBERO (structure does not help
when perception/data already solve the task) is in `docs/REPRODUCTION_GUIDE.md`.

## Task

Planar end-effector reach on the robosuite `Lift` env (Panda), OSC_POSE
controller, low-dim state only (`MUJOCO_GL=egl`, no rendering). Each scene has a
target displacement (radius 0.16–0.26 m, uniform angle) and a **virtual
obstacle** on the start→target line (longitudinal fraction 0.4–0.6, lateral
offset ±0.04 m, radius 0.03–0.05 m). A straight reach intersects the obstacle;
the demo must detour. Success (offline) = final EE displacement within 0.03 m of
the target AND every trajectory point outside the obstacle disk.

Demos: plan a 2-D detour path (progress ramps to target by 72% of the horizon
then dwells so the proportional controller settles; endpoint-vanishing lateral
clearance bump centered at the obstacle's longitudinal position, over-cleared by
obstacle-radius + 0.085 m to absorb OSC tracking lag; side geometry-forced when
the obstacle is offset, chosen per-demo when centered — a bimodal style choice;
small lateral wiggle). OSC-track the path on the real arm; store the achieved
EE-(x,y) delta chunk (H=32). 120 scenes × 8 demos. Demo success ceiling **0.82**.

## Structure discovery (over demos)

Canonicalize each chunk (rotate so the reach direction is +x). For axis
u ∈ {progress=(1,0), lateral=(0,1)} project the chunk onto u and take the rFFT;
each Fourier bin ω has a complex coefficient. Cross-demo phase coherence at
(u,ω): γ = |mean over demos of exp(i·phase)| (mod-2π), γ2 with 2·phase (mod-π,
side-invariant). Select bins with energy fraction ≥ 0.10, coherence > 0.6, and
mark magnitude-carrying when the cross-demo magnitude CV < 0.15. Discovered frame
S_F (3 pins): **progress ω0 (with magnitude) = net displacement / endpoint;
lateral ω1 (mod-π, with magnitude) = primary detour bend; lateral ω2 (mod-π) =
second lateral harmonic.** Implemented by treating each demo index as a "body"
so `flow_embod.freeze_frame`'s cross-body coherence operates over demos.

## Arms (single embodiment; 100 train scenes, 20 held-out; 3 seeds)

Small flow-matching executor (autograd MLP), 8000 steps each:
- **A** — scratch: plain source noise, no structure.
- **F** — S_F pinned into the source noise; command from a scene→invariant prior
  trained on set demos.
- **Frand** — random-frame pin + its own prior (control).

## Result (held-out obstacle-reach success, pooled over 3 seeds)

| arm | success |
|---|---|
| F (discovered structure) | **0.383** |
| A (scratch) | 0.021 |
| Frand (random frame) | 0.008 |

F − A = +0.362; F − Frand = +0.375; demo ceiling 0.82. Per-seed F:
0.219 / 0.431 / 0.500; A: 0.025 / 0.000 / 0.037.

Interpretation: the scratch policy near-fails the detour (the policy is the
bottleneck); the discovered structure supplies the geometric organization it
cannot find from the data and raises success by ~0.36. F > Frand isolates the
learned coherence frame from an arbitrary pin. F below the 0.82 ceiling reflects
prior quality (same pattern as the toy T-oracle > T and LIBERO).

## Caveats

- Scratch (~0.02) is a weak baseline (hard task, small policy, finite data), so
  the ~18× ratio overstates; the load-bearing claims are absolute F = 0.38 and
  F ≫ Frand.
- Offline geometric success on real-kinematics demo chunks, not closed-loop
  robosuite execution.
- Demos are not bit-reproducible (robosuite `reset` randomizes object/start
  pose); the scene list is seeded (`default_rng(0)`) and the F/A/Frand ordering
  is stable across 3 seeds.

## Reproduce

Box `~/code/source-noise-mvp/experiments/rung2/`. Env: robosuite==1.4.1,
mujoco==2.3.7 (3.x breaks robosuite 1.4.1), `MUJOCO_GL=egl`, python 3.11.

    # 1. collect demos (real robosuite; ~25 min; writes data_obst/Panda.npz)
    bash run_obstacle.sh
    # 2. structure vs baseline (CPU/autograd; ~15 min; writes structure_result.json)
    ../../.venv/bin/python structure_test.py

Scripts: `collect_obstacle.py`, `structure_test.py`, `run_obstacle.sh`.
Result: `structure_result.json`.
