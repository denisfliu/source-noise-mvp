"""Closed-loop execution, stage 2 (robosuite env): execute the generated pose-delta
chunks from stage 1 on the real Panda arm and measure achieved task success under
simulator dynamics, for the scratch and grid-Laplacian policies. Reports closed-loop
success and its agreement with the offline geometric success recorded in stage 1.

Each generated chunk is a canonical pose-delta trajectory. It is executed in a world
frame with the reach azimuth set to zero, so canonical coordinates equal world
coordinates. The chunk is integrated into an absolute pose waypoint sequence, which
is tracked with the same OSC position and orientation control used to collect the
demonstrations; success is read from the achieved end-effector trajectory.
"""
import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

HERE = os.path.dirname(os.path.abspath(__file__))
KP_POS = 12.0
KP_ROT = 5.0
OVERCLEAR = 0.12
COUPLE = 1.0
TOL_POS = 0.03
TOL_ROT = 0.15
EXTRA_HOLD = 6
cfg = load_controller_config(default_controller="OSC_POSE")


def waypoints(chunk, p0, q0):
    """Integrate a canonical pose-delta chunk (H,6) into absolute world waypoints
    (reach azimuth zero, so canonical equals world). Returns positions (H+1,3) and
    quaternions list (H+1)."""
    dpos = chunk[:, :3]; dori = chunk[:, 3:]
    pos = np.concatenate([p0[None], p0[None] + np.cumsum(dpos, axis=0)], axis=0)
    quats = [q0.copy()]
    for t in range(len(dori)):
        quats.append(T.quat_multiply(T.axisangle2quat(dori[t]), quats[-1]))
    return pos, quats


def execute(env, chunk, sc):
    o = env.reset()
    p0 = o["robot0_eef_pos"].copy(); q0 = o["robot0_eef_quat"].copy()
    pos_wp, quat_wp = waypoints(chunk, p0, q0)
    rad, s_o, r, lat = sc
    tgt_pos = p0 + np.array([rad, 0.0, 0.0])
    side = np.sign(lat)
    bank = COUPLE * (-side) * np.clip(3.0 * (r + OVERCLEAR), 0.0, 0.7)
    tgt_quat = T.quat_multiply(T.axisangle2quat(np.array([bank, 0.0, 0.0])), q0)
    obst_c = p0 + np.array([s_o * rad, lat, 0.0])
    H = chunk.shape[0]
    achieved = [p0.copy()]
    for t in range(H + EXTRA_HOLD):
        k = min(t + 1, H)                                  # hold last waypoint to settle
        eef = o["robot0_eef_pos"]; qc = o["robot0_eef_quat"]
        dpos = np.clip(KP_POS * (pos_wp[k] - eef), -1, 1)
        err_q = T.quat_multiply(quat_wp[k], T.quat_inverse(qc))
        daa = np.clip(KP_ROT * T.quat2axisangle(err_q), -1, 1)
        o, _, _, _ = env.step(np.concatenate([dpos, daa, [0.0]]))
        achieved.append(o["robot0_eef_pos"].copy())
    achieved = np.array(achieved)
    pe = np.linalg.norm(achieved[-1] - tgt_pos)
    re = np.linalg.norm(T.quat2axisangle(T.quat_multiply(tgt_quat, T.quat_inverse(o["robot0_eef_quat"]))))
    clr = (np.linalg.norm(achieved - obst_c, axis=1) > r).all()
    return float(pe < TOL_POS and re < TOL_ROT and clr)


def main():
    ARM = os.environ.get("SNMVP_ARM", "Panda")
    npz = os.environ.get("SNMVP_NPZ", "cle_chunks.npz")
    outf = os.environ.get("SNMVP_OUT", "cle_result.json")
    d = np.load(os.path.join(HERE, npz))
    genS, genG, obs = d["gen_S"], d["gen_GLAP"], d["obs"]
    off_S, off_G = d["off_S"], d["off_GLAP"]
    print(f"executing on arm {ARM} from {npz}", flush=True)
    env = suite.make(env_name="Lift", robots=ARM, controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    M = len(obs)
    cl_S = np.array([execute(env, genS[i], obs[i]) for i in range(M)])
    cl_G = np.array([execute(env, genG[i], obs[i]) for i in range(M)])
    env.close()
    agree_S = float((cl_S == off_S).mean()); agree_G = float((cl_G == off_G).mean())
    print(f"closed-loop success:  scratch {cl_S.mean():.3f}   GLAP {cl_G.mean():.3f}")
    print(f"offline geometric:    scratch {off_S.mean():.3f}   GLAP {off_G.mean():.3f}")
    print(f"per-scene agreement (closed-loop == offline): scratch {agree_S:.3f}  GLAP {agree_G:.3f}")
    import json
    json.dump({"closed_loop": {"scratch": round(float(cl_S.mean()), 3), "GLAP": round(float(cl_G.mean()), 3)},
               "offline": {"scratch": round(float(off_S.mean()), 3), "GLAP": round(float(off_G.mean()), 3)},
               "agreement": {"scratch": round(agree_S, 3), "GLAP": round(agree_G, 3)},
               "arm": ARM, "M": M},
              open(os.path.join(HERE, outf), "w"), indent=2)
    print("EXEC_CLE_DONE=ok")


if __name__ == "__main__":
    main()
