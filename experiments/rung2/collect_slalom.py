"""Rung 2 (SLALOM version): a strictly harder task than single-obstacle reach.
Two virtual obstacles sit near the start->target line on OPPOSITE sides, so a
straight reach hits both and a single-bend detour clears at most one — the demo
must weave (S-curve). This forces a RICHER discoverable structure: endpoint (ω0)
plus two opposite lateral bends, i.e. lateral energy at ω1 AND a higher harmonic
with a coherent cross-demo phase relationship. Same collection recipe as
collect_obstacle.py (plan a 2-D path, OSC-track it on a real robosuite arm so the
achieved EE trajectory carries real tracking dynamics), extended to two bumps.

Output npz: chunks (S,N,H,2) EE-xy deltas; obs (S,6)=[disp_xy, o1_xy, o2_xy]
(start-relative, EE plane); obst_r (S,2)=[o1_r,o2_r]; success (S,N).
Success (offline) = reach target within TOL AND every point clears BOTH disks.
"""
import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

ARM = os.environ.get("SNMVP_ARM", "Panda")
H = 32
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "120"))
N_DEMOS = int(os.environ.get("SNMVP_NDEMOS", "8"))
KP = 12.0
TOL = 0.03
OVERCLEAR = float(os.environ.get("SNMVP_OVERCLEAR", "0.10"))   # margin beyond obstacle radius (m)
BUMP_W = float(os.environ.get("SNMVP_BUMPW", "0.08"))          # gaussian bump width (progress frac)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_slalom")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE")


def make_scene(rng):
    ang = np.radians(rng.uniform(-150.0, 150.0))        # exclude rear cone (workspace limit)
    rad = rng.uniform(0.16, 0.24)                       # reach distance (m); rearward reach is short
    s1 = rng.uniform(0.24, 0.34)                        # obstacle 1 along the line
    s2 = rng.uniform(0.62, 0.72)                        # obstacle 2 (well separated -> less bump cross-talk)
    g = rng.choice([-1.0, 1.0])                         # global weave orientation (bimodal)
    d1 = rng.uniform(0.03, 0.06)
    d2 = rng.uniform(0.03, 0.06)
    r1 = rng.uniform(0.03, 0.05)
    r2 = rng.uniform(0.03, 0.05)
    # obstacles on OPPOSITE sides -> a single bend cannot clear both.
    lat1 = g * d1
    lat2 = -g * d2
    return {"ang": ang, "rad": rad, "s1": s1, "s2": s2,
            "lat1": lat1, "lat2": lat2, "r1": r1, "r2": r2, "g": g}


def _bump(p, s_o, amp, w=BUMP_W):
    """endpoint-vanishing lateral bump as a function of progress fraction p,
    peaking when the EE is longitudinally at the obstacle (progress p==s_o)."""
    raw = np.exp(-((p - s_o) ** 2) / (2 * w ** 2))
    ramp = (1 - p) * raw[0] + p * raw[-1]
    peak = 1.0 - ((1 - s_o) * raw[0] + s_o * raw[-1])
    return amp * (raw - ramp) / max(peak, 1e-6)


def planned_path(sc, rng):
    """2-D EE path (H+1,2) start-relative: progress along the reach direction +
    two opposite endpoint-vanishing clearance bumps (the slalom weave)."""
    rad, s1, s2 = sc["rad"], sc["s1"], sc["s2"]
    s = np.linspace(0, 1, H + 1)
    # progress fraction p in [0,1]: reaches 1 by s=0.72 then DWELLS (~9 steps).
    p = np.clip(s / 0.72, 0.0, 1.0)
    # smoothstep the LONGITUDINAL profile so the EE arrives at the target with
    # ~zero velocity (3x^2-2x^3 has zero derivative at the endpoints) — removes
    # the velocity discontinuity that made the arm overshoot during the dwell.
    prog = rad * (3 * p ** 2 - 2 * p ** 3)
    q = prog / rad                                      # true longitudinal fraction in [0,1]
    # pass each obstacle on its FAR side; over-clear (r + 0.085 m) to absorb the
    # OSC tracking lag that cuts the achieved path inside the planned bump. Bumps
    # are centered on the longitudinal fraction q (where the EE actually is).
    a1 = -np.sign(sc["lat1"]) * (abs(sc["lat1"]) + sc["r1"] + OVERCLEAR)
    a2 = -np.sign(sc["lat2"]) * (abs(sc["lat2"]) + sc["r2"] + OVERCLEAR)
    bump = _bump(q, s1, a1) + _bump(q, s2, a2)
    bump += rng.normal(0, 0.0015, H + 1) * np.sin(np.pi * q)   # style wiggle (motion only)
    prog[-1] = rad; bump[0] = 0.0; bump[-1] = 0.0
    c, si = np.cos(sc["ang"]), np.sin(sc["ang"])       # rotate into EE xy plane
    x = prog * c - bump * si
    y = prog * si + bump * c
    return np.stack([x, y], axis=1)                     # (H+1,2), start-relative


def obstacle_xy(sc, s_o, lat):
    c, si = np.cos(sc["ang"]), np.sin(sc["ang"])
    return np.array([s_o * sc["rad"] * c - lat * si,
                     s_o * sc["rad"] * si + lat * c])


def collect():
    env = suite.make(env_name="Lift", robots=ARM, controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    rng = np.random.default_rng(0)
    scenes = [make_scene(rng) for _ in range(N_SCENES)]
    chunks = np.zeros((N_SCENES, N_DEMOS, H, 2))
    succ = np.zeros((N_SCENES, N_DEMOS))
    obsv = np.zeros((N_SCENES, 6)); obr = np.zeros((N_SCENES, 2))
    for si, sc in enumerate(scenes):
        disp = sc["rad"] * np.array([np.cos(sc["ang"]), np.sin(sc["ang"])])
        o1 = obstacle_xy(sc, sc["s1"], sc["lat1"])
        o2 = obstacle_xy(sc, sc["s2"], sc["lat2"])
        obsv[si] = np.concatenate([disp, o1, o2]); obr[si] = [sc["r1"], sc["r2"]]
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
            clr1 = (np.linalg.norm(rel - o1, axis=1) > sc["r1"]).all()
            clr2 = (np.linalg.norm(rel - o2, axis=1) > sc["r2"]).all()
            succ[si, di] = float(end_ok and clr1 and clr2)
    env.close()
    np.savez(os.path.join(OUT, f"{ARM}.npz"), chunks=chunks, obs=obsv,
             obst_r=obr, success=succ)
    print(ARM, "demo slalom-success rate", round(float(succ.mean()), 3),
          "mean|disp|", round(float(np.linalg.norm(obsv[:, :2], axis=1).mean()), 3),
          flush=True)
    print("COLLECT_SLALOM_DONE=ok")


if __name__ == "__main__":
    collect()
