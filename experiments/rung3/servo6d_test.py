"""Stage-1 de-risk: can we OSC_POSE-servo the Panda EE to a target 6-DOF POSE
(position AND orientation) in sim? Verifies the control layer the closed-loop
pose-reach task will rely on. Reports final position + orientation error over
random target poses."""
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

KP_POS = 12.0
KP_ROT = 4.0
N_STEP = 60
cfg = load_controller_config(default_controller="OSC_POSE")


def main():
    env = suite.make(env_name="Lift", robots="Panda", controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    rng = np.random.default_rng(0)
    pos_errs, rot_errs = [], []
    for trial in range(12):
        obs = env.reset()
        start_pos = obs["robot0_eef_pos"].copy()
        start_quat = obs["robot0_eef_quat"].copy()               # xyzw
        # target: position offset + orientation rotated by a random axis-angle
        tgt_pos = start_pos + np.concatenate([rng.uniform(-0.12, 0.12, 2), [rng.uniform(-0.06, 0.06)]])
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        ang = rng.uniform(0.2, 0.6)                               # rad
        dquat = T.axisangle2quat(axis * ang)
        tgt_quat = T.quat_multiply(dquat, start_quat)
        for t in range(N_STEP):
            eef = obs["robot0_eef_pos"]; q = obs["robot0_eef_quat"]
            dpos = np.clip(KP_POS * (tgt_pos - eef), -1, 1)
            err_quat = T.quat_multiply(tgt_quat, T.quat_inverse(q))
            err_aa = T.quat2axisangle(err_quat)
            daa = np.clip(KP_ROT * err_aa, -1, 1)
            a = np.concatenate([dpos, daa, [0.0]])               # OSC_POSE + gripper
            obs, _, _, _ = env.step(a)
        pe = np.linalg.norm(obs["robot0_eef_pos"] - tgt_pos)
        eq = T.quat_multiply(tgt_quat, T.quat_inverse(obs["robot0_eef_quat"]))
        re = np.linalg.norm(T.quat2axisangle(eq))
        pos_errs.append(pe); rot_errs.append(re)
    env.close()
    pos_errs = np.array(pos_errs); rot_errs = np.array(rot_errs)
    print(f"pos_err  m: median={np.median(pos_errs):.4f} p90={np.percentile(pos_errs,90):.4f}")
    print(f"rot_err rad: median={np.median(rot_errs):.4f} p90={np.percentile(rot_errs,90):.4f}")
    print("SERVO6D_DONE=ok")


if __name__ == "__main__":
    main()
