"""Rung 2 step 1: collect planar EE-reach demos across arm morphologies.

Each episode: reset, drive the end-effector from its start pose to a random
target in a fixed-height plane via a proportional OSC_POSE controller; record the
achieved 2D EE (x,y) trajectory. Different arms realize the same reach with
different kinematics (the embodiment difference); the action interface is the
shared EE-delta space. Output per arm: chunks (N,H,2) EE-xy deltas, disp (N,2)
target displacement, success (N,). Low-dim state only (no rendering).

Saved arrays feed the offline coherence + transfer pipeline (reuses the toy
machinery), so robosuite is NOT needed downstream.
"""

import json
import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

ARMS = ["Panda", "Sawyer", "IIWA", "UR5e"]
H = 32
N_PER_ARM = 80
KP = 12.0
TOL = 0.03
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE")


def collect(robot, targets):
    """targets: shared (N,2) list of (angle, radius) so demo i is the SAME scene
    across arms (required for cross-arm coherence)."""
    env = suite.make(env_name="Lift", robots=robot, controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    chunks, disp, succ = [], [], []
    for ep in range(N_PER_ARM):
        obs = env.reset()
        start = obs["robot0_eef_pos"].copy()
        ang, rad = targets[ep]
        tgt = start.copy()
        tgt[0] += rad * np.cos(ang)
        tgt[1] += rad * np.sin(ang)
        traj = [start[:2].copy()]
        for t in range(H):
            eef = obs["robot0_eef_pos"]
            a = np.zeros(env.action_dim)
            a[0] = np.clip(KP * (tgt[0] - eef[0]), -1, 1)
            a[1] = np.clip(KP * (tgt[1] - eef[1]), -1, 1)
            a[2] = np.clip(KP * (start[2] - eef[2]), -1, 1)   # hold height
            obs, r, d, info = env.step(a)
            traj.append(obs["robot0_eef_pos"][:2].copy())
        traj = np.array(traj)
        chunks.append(np.diff(traj, axis=0))
        disp.append(tgt[:2] - start[:2])
        succ.append(float(np.linalg.norm(traj[-1] - tgt[:2]) < TOL))
    env.close()
    return np.array(chunks), np.array(disp), np.array(succ)


def main():
    trng = np.random.default_rng(0)                       # SHARED targets across arms
    targets = [(float(trng.uniform(-np.pi, np.pi)), float(trng.uniform(0.10, 0.28)))
               for _ in range(N_PER_ARM)]
    stats = {}
    for i, robot in enumerate(ARMS):
        ch, dp, sc = collect(robot, targets)
        np.savez(os.path.join(OUT, f"{robot}.npz"), chunks=ch, disp=dp, success=sc)
        stats[robot] = {"n": int(len(ch)), "success_rate": round(float(sc.mean()), 3),
                        "mean_disp_m": round(float(np.linalg.norm(dp, axis=1).mean()), 3),
                        "mean_reach_err_m": round(float(np.mean(
                            [np.linalg.norm(ch[k].sum(0) - dp[k]) for k in range(len(ch))])), 4)}
        print(robot, json.dumps(stats[robot]), flush=True)
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print("COLLECT_DONE=ok")


if __name__ == "__main__":
    main()
