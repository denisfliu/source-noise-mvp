"""Rung 2 scaffold step 0: confirm robosuite gives multiple arm morphologies a
SHARED task-space (end-effector-delta) action interface on the same task.

OSC_POSE controller = Cartesian EE-delta action space (3 pos + 3 rot + gripper),
identical across arms, realized by each arm's own kinematics -> the real-scale
analog of the toy's task-space-actions design (invariant linear, pin exact;
embodiment difference = kinematics/reachability). Low-dim state only (no
rendering) for the first pass, per the oracle-first discipline.
"""

import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

ARMS = ["Panda", "Sawyer", "IIWA"]
cfg = load_controller_config(default_controller="OSC_POSE")

for robot in ARMS:
    env = suite.make(env_name="Lift", robots=robot, controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, reward_shaping=True, control_freq=20)
    obs = env.reset()
    lowdim = [k for k in obs if "image" not in k]
    print(f"{robot:8} action_dim={env.action_dim} "
          f"eef_pos={np.round(obs['robot0_eef_pos'], 3).tolist()} "
          f"n_lowdim_keys={len(lowdim)}", flush=True)
    a = np.zeros(env.action_dim)
    a[2] = -0.2                      # small downward EE command
    for _ in range(5):
        obs, r, done, info = env.step(a)
    print(f"         after 5 down-steps eef_z={round(float(obs['robot0_eef_pos'][2]),3)}",
          flush=True)
    env.close()

print("ENVCHECK_DONE=ok")
