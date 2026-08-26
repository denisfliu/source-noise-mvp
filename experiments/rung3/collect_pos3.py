"""Variable-DOF study: collect a position-only detour reach with the OSC_POSITION
controller (three action channels, orientation held by the controller). The stored
chunk is the achieved end-effector position-delta trajectory (H,3) in the canonical
frame. This is the three-channel embodiment used to test transfer across an
action-dimension change against the six-channel pose embodiments.

Same scene parameters and canonical frame as the pose task, so scenes are paired by
seed across the two action representations. obs (4) = [rad, s_o, r, lateral].
Success = final position within TOL of the target and every point clears the obstacle.
"""
import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

ARM = os.environ.get("SNMVP_ARM", "Panda")
H = 32
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "120"))
N_DEMOS = int(os.environ.get("SNMVP_NDEMOS", "8"))
KP_POS = 12.0
TOL_POS = 0.03
OVERCLEAR = float(os.environ.get("SNMVP_OVERCLEAR", "0.12"))
BUMP_W = float(os.environ.get("SNMVP_BUMPW", "0.16"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_pos3")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSITION")


def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_scene(rng):
    phi = np.radians(rng.uniform(-150.0, 150.0))
    rad = rng.uniform(0.16, 0.24)
    s_o = rng.uniform(0.4, 0.6)
    r = rng.uniform(0.04, 0.06)
    side = rng.choice([-1.0, 1.0])
    lateral = side * rng.uniform(0.02, 0.05)
    return {"phi": phi, "rad": rad, "s_o": s_o, "r": r, "lateral": lateral}


def obs_vec(sc):
    return np.array([sc["rad"], sc["s_o"], sc["r"], sc["lateral"]])


def _bump(q, s_o, amp, w=BUMP_W):
    raw = np.exp(-((q - s_o) ** 2) / (2 * w ** 2))
    ramp = (1 - q) * raw[0] + q * raw[-1]
    peak = 1.0 - ((1 - s_o) * raw[0] + s_o * raw[-1])
    return amp * (raw - ramp) / max(peak, 1e-6)


def plan_world(sc, p0, rng):
    rad, s_o, r, phi = sc["rad"], sc["s_o"], sc["r"], sc["phi"]
    s = np.linspace(0, 1, H + 1)
    p = np.clip(s / 0.72, 0.0, 1.0)
    prog = rad * (3 * p ** 2 - 2 * p ** 3)
    q = prog / rad
    amp = (r + OVERCLEAR) * (1 + rng.normal(0, 0.04))
    side = -np.sign(sc["lateral"])
    bump = _bump(q, s_o, side * amp)
    can = np.outer(prog, np.array([1.0, 0.0, 0.0])) + np.outer(bump, np.array([0.0, 1.0, 0.0]))
    can[:, 2] += rng.normal(0, 0.002, H + 1) * np.sin(np.pi * q)
    can[0] = 0.0; can[-1] = np.array([rad, 0.0, 0.0])
    return p0 + can @ Rz(phi).T


def collect():
    env = suite.make(env_name="Lift", robots=ARM, controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    adim = env.action_dim
    rng = np.random.default_rng(0)
    scenes = [make_scene(rng) for _ in range(N_SCENES)]
    chunks = np.zeros((N_SCENES, N_DEMOS, H, 3))
    succ = np.zeros((N_SCENES, N_DEMOS))
    obsv = np.zeros((N_SCENES, 4))
    for si, sc in enumerate(scenes):
        obsv[si] = obs_vec(sc)
        for di in range(N_DEMOS):
            o = env.reset()
            p0 = o["robot0_eef_pos"].copy()
            positions = plan_world(sc, p0, rng)
            tgt = positions[-1]
            obst_c = p0 + Rz(sc["phi"]) @ np.array([sc["s_o"] * sc["rad"], sc["lateral"], 0.0])
            tr = [p0.copy()]
            for t in range(H):
                eef = o["robot0_eef_pos"]
                dpos = np.clip(KP_POS * (positions[t + 1] - eef), -1, 1)
                a = np.zeros(adim); a[:3] = dpos                # OSC_POSITION + gripper
                o, _, _, _ = env.step(a)
                tr.append(o["robot0_eef_pos"].copy())
            tr = np.array(tr)
            dpos_c = np.diff(tr, axis=0) @ Rz(sc["phi"])        # canonical position deltas
            chunks[si, di] = dpos_c
            pe = np.linalg.norm(tr[-1] - tgt)
            clr = (np.linalg.norm(tr - obst_c, axis=1) > sc["r"]).all()
            succ[si, di] = float(pe < TOL_POS and clr)
    env.close()
    np.savez(os.path.join(OUT, f"{ARM}.npz"), chunks=chunks, obs=obsv, success=succ)
    print(ARM, f"action_dim={adim} demo pos3 success", round(float(succ.mean()), 3), flush=True)
    print("COLLECT_POS3_DONE=ok")


if __name__ == "__main__":
    collect()
