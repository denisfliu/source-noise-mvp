"""Rung 2 (obstacle version): collect OBSTACLE-REACH demos on a real robosuite
arm — the HARD, policy-bottlenecked task needed to test whether discovered
structure helps (the scaled analog of the toy_frame +17-pt learned-frame result).

Each scene = a planar reach with a VIRTUAL obstacle on the start->target line;
the demo must DETOUR around it (a straight reach fails). We plan a 2-D detour
path (endpoint-vanishing lateral clearance bump, side geometry-forced when the
obstacle is offset, style-chosen when centered + small wiggle), then OSC-track it
on the real arm so the achieved EE trajectory carries real tracking dynamics.
Multiple demos per scene (style variation) so coherence can discover the shared
detour structure. Success (offline) = reach target AND every point clears the
obstacle disk. Output: chunks (S,N,H,2) EE-xy deltas, obs (S,4)=[disp_xy,
obst_xy] (scene, all start-relative in the EE plane), obst_r (S,), success (S,N).
"""
import json, os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

ARM = os.environ.get("SNMVP_ARM", "Panda")
H = 32
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "120"))
N_DEMOS = int(os.environ.get("SNMVP_NDEMOS", "8"))
KP = 12.0
TOL = 0.03
CLEAR = 0.02                      # required clearance beyond obstacle radius (m)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_obst")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE")


def make_scene(rng):
    ang = rng.uniform(-np.pi, np.pi)
    rad = rng.uniform(0.16, 0.26)                       # reach distance (m)
    s_o = rng.uniform(0.4, 0.6)                          # obstacle along the line
    lateral = rng.uniform(-0.04, 0.04)                  # small -> often centered (bimodal side)
    obst_r = rng.uniform(0.03, 0.05)
    return {"ang": ang, "rad": rad, "s_o": s_o, "lateral": lateral, "obst_r": obst_r}


def planned_path(sc, rng):
    """2-D EE path (H+1,2) start-relative: progress along the reach direction +
    an endpoint-vanishing lateral clearance bump around the obstacle."""
    rad, s_o, lateral, obr = sc["rad"], sc["s_o"], sc["lateral"], sc["obst_r"]
    s = np.linspace(0, 1, H + 1)
    # progress ramps to the target by s=0.72 then DWELLS (so the proportional
    # controller has ~8 steps to settle within tolerance after the detour).
    p = np.clip(s / 0.72, 0.0, 1.0)                     # progress fraction in [0,1]
    prog = p * rad
    # side: forced away from the obstacle's offset; style choice when centered.
    # Over-clear (obr + 8.5cm) so the OSC-tracked path, which LAGS the planned
    # bump, still clears the disk (empirically the achieved path cuts inside).
    Rm = obr + 0.085
    if abs(lateral) < 0.5 * Rm:
        side = rng.choice([-1.0, 1.0])                  # both viable -> style (bimodal)
    else:
        side = -np.sign(lateral)                        # pass on the far side
    amp = side * (abs(lateral) + Rm)
    # bump as a function of PROGRESS FRACTION, centered at s_o (peaks when the EE
    # is longitudinally at the obstacle); endpoint-vanishing in p so it is 0 at
    # start (p=0) and target/dwell (p=1).
    w = 0.14
    raw = np.exp(-((p - s_o) ** 2) / (2 * w ** 2))
    ramp = (1 - p) * raw[0] + p * raw[-1]
    peak = np.exp(0.0) - ((1 - s_o) * raw[0] + s_o * raw[-1])
    bump = amp * (raw - ramp) / max(peak, 1e-6)
    bump += rng.normal(0, 0.002, H + 1) * np.sin(np.pi * p)   # style wiggle (motion only)
    prog[-1] = rad; bump[0] = 0.0; bump[-1] = 0.0
    c, si = np.cos(sc["ang"]), np.sin(sc["ang"])       # rotate into EE xy plane
    x = prog * c - bump * si
    y = prog * si + bump * c
    return np.stack([x, y], axis=1)                     # (H+1,2), start-relative


def obstacle_xy(sc):
    c, si = np.cos(sc["ang"]), np.sin(sc["ang"])
    return np.array([sc["s_o"] * sc["rad"] * c - sc["lateral"] * si,
                     sc["s_o"] * sc["rad"] * si + sc["lateral"] * c])


def collect():
    env = suite.make(env_name="Lift", robots=ARM, controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    rng = np.random.default_rng(0)
    scenes = [make_scene(rng) for _ in range(N_SCENES)]
    chunks = np.zeros((N_SCENES, N_DEMOS, H, 2))
    succ = np.zeros((N_SCENES, N_DEMOS))
    obsv = np.zeros((N_SCENES, 4)); obr = np.zeros(N_SCENES)
    for si, sc in enumerate(scenes):
        disp = sc["rad"] * np.array([np.cos(sc["ang"]), np.sin(sc["ang"])])
        obst = obstacle_xy(sc)
        obsv[si] = np.concatenate([disp, obst]); obr[si] = sc["obst_r"]
        for di in range(N_DEMOS):
            obs = env.reset()
            start = obs["robot0_eef_pos"].copy()
            path = planned_path(sc, rng)                # start-relative
            traj = [start[:2].copy()]
            for t in range(H):
                eef = obs["robot0_eef_pos"]
                wp = start[:2] + path[t + 1]             # absolute waypoint
                a = np.zeros(env.action_dim)
                a[0] = np.clip(KP * (wp[0] - eef[0]), -1, 1)
                a[1] = np.clip(KP * (wp[1] - eef[1]), -1, 1)
                a[2] = np.clip(KP * (start[2] - eef[2]), -1, 1)
                obs, _, _, _ = env.step(a)
                traj.append(obs["robot0_eef_pos"][:2].copy())
            traj = np.array(traj)
            rel = traj - traj[0]                         # start-relative achieved
            chunks[si, di] = np.diff(traj, axis=0)
            end_ok = np.linalg.norm(rel[-1] - disp) < TOL
            clear = (np.linalg.norm(rel - obst, axis=1) > sc["obst_r"]).all()
            succ[si, di] = float(end_ok and clear)
    env.close()
    np.savez(os.path.join(OUT, f"{ARM}.npz"), chunks=chunks, obs=obsv,
             obst_r=obr, success=succ)
    print(ARM, "demo success rate", round(float(succ.mean()), 3),
          "mean|disp|", round(float(np.linalg.norm(obsv[:, :2], axis=1).mean()), 3),
          flush=True)
    print("COLLECT_OBST_DONE=ok")


if __name__ == "__main__":
    collect()
