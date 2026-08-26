# Rung 2 scaffold (robosuite, real arm morphologies) — status

Real-perception/small-model scale-up of the cross-embodiment toy
(`docs/rung2_plan.md`). CPU-toy result: transfer robust (T>S, T>Trand all
bodies); pin>conditioning only at very low n; coherence does NOT predict gain
(see `findings/toy_embodiment_*`). Rung 2 asks whether the transfer survives
real kinematics + (later) real perception.

## Working environment (validated 2026-07-19)

- Both box GPUs free; disk ~51%.
- robosuite==1.4.1 + **mujoco==2.3.7** (newer mujoco 3.x breaks robosuite 1.4.1
  with `mj_fullM(): incompatible function arguments` — MUST pin 2.3.x).
- Run headless with `MUJOCO_GL=egl`, `has_renderer=False,
  has_offscreen_renderer=False, use_camera_obs=False` (low-dim state only, no
  GL rendering — the first-pass "privileged state" regime).
- Launch via a box-side script (`run_*.sh` + nohup + disown), not an inline
  `ssh "... &"` — the SSH link to the box drops intermittently and kills inline
  backgrounded launches.
- Invocation: `MUJOCO_GL=egl uv run --with 'robosuite==1.4.1' --with
  'mujoco==2.3.7' --with numpy --python 3.11 python <script>.py`.

## Step 0 DONE — shared action interface (`env_check.py`)

OSC_POSE controller gives Panda/Sawyer/IIWA an identical **action_dim=7**
(6 EE-delta + gripper) on the Lift task, 12 low-dim obs keys each; different
start EE poses / kinematics. => the toy's task-space-actions design holds at
real scale; invariant (EE displacement) linear in this action space.

## Next steps (not yet built)

1. **Scripted demo collector** per arm on a shared task (reach/lift): use the
   object pose from low-dim obs to script OSC_POSE waypoints (approach, grasp,
   lift); save EE-trajectory chunks. Set A = 3-4 arms; held-out = another arm.
2. **Coherence frame** over the arms' EE-trajectory invariants (port
   `toy_embodiment` coherence + energy-floor/CV gating to 6-DOF EE space;
   object/goal-centric, task-frame).
3. **Small flow/diffusion executor + prior** (torch, one GPU) + freeze-and-adapt
   transfer test (T/S/Cond/Trand), then add gsplat perception.

Files: `env_check.py`, `run_envcheck.sh`.
