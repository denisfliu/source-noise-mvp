"""Variable-DOF gate collection, SLALOM (bottlenecked) version. Two obstacles on
OPPOSITE sides of the reach line -> the demo must weave (S-curve), a higher-rank
position structure that a scarce-data executor cannot fit (the single-obstacle
version was not bottlenecked: scratch ~0.72). The weave handedness is FIXED
(obstacle 1 on +side, obstacle 2 on -side) so the planned path is deterministic
from the scene (needed for the planned-pin and deconvolution reference); difficulty
comes from the two-bump structure, not from bimodality.

Same canonical frame as collect_vardof_hard, collected under two controllers:
  SNMVP_CTRL=pose : OSC_POSE (6-ch), orientation commanded zero, chunk (H,6)
  SNMVP_CTRL=pos  : OSC_POSITION (3-ch), chunk (H,3)
obs (7) = [rad, s1, s2, d1, d2, r1, r2]. Success = position reach within tolerance
AND every point clears BOTH obstacle disks. Output data_vardof_slalom/<ctrl>_<arm>.npz.
"""
import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

ARM = os.environ.get("SNMVP_ARM", "Panda")
CTRL = os.environ.get("SNMVP_CTRL", "pose")
H = 32
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "120"))
N_DEMOS = int(os.environ.get("SNMVP_NDEMOS", "8"))
KP = 12.0
TOL_POS = 0.03
OVERCLEAR = float(os.environ.get("SNMVP_OVERCLEAR", "0.085"))
BUMP_W = float(os.environ.get("SNMVP_BUMPW", "0.08"))
CDIM = 6 if CTRL == "pose" else 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_vardof_slalom")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE" if CTRL == "pose" else "OSC_POSITION")


def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_scene(rng):
    phi = np.radians(rng.uniform(-150.0, 150.0))
    rad = rng.uniform(0.16, 0.24)
    s1 = rng.uniform(0.24, 0.34)
    s2 = rng.uniform(0.62, 0.72)
    d1 = rng.uniform(0.03, 0.06)
    d2 = rng.uniform(0.03, 0.06)
    r1 = rng.uniform(0.03, 0.05)
    r2 = rng.uniform(0.03, 0.05)
    return {"phi": phi, "rad": rad, "s1": s1, "s2": s2, "d1": d1, "d2": d2, "r1": r1, "r2": r2}


def obs_vec(sc):
    return np.array([sc["rad"], sc["s1"], sc["s2"], sc["d1"], sc["d2"], sc["r1"], sc["r2"]])


def _bump(q, s_o, amp, w=BUMP_W):
    raw = np.exp(-((q - s_o) ** 2) / (2 * w ** 2))
    ramp = (1 - q) * raw[0] + q * raw[-1]
    peak = 1.0 - ((1 - s_o) * raw[0] + s_o * raw[-1])
    return amp * (raw - ramp) / max(peak, 1e-6)


def plan_canonical_xy(sc, rng):
    """canonical (progress x, lateral y) path (H+1,2); fixed handedness: obstacle 1
    on +y so pass on -y, obstacle 2 on -y so pass on +y (the S-weave)."""
    rad = sc["rad"]
    s = np.linspace(0, 1, H + 1)
    p = np.clip(s / 0.72, 0.0, 1.0)
    prog = rad * (3 * p ** 2 - 2 * p ** 3)
    q = prog / rad
    a1 = -1.0 * (sc["d1"] + sc["r1"] + OVERCLEAR)          # obstacle1 at +d1 -> pass -side
    a2 = +1.0 * (sc["d2"] + sc["r2"] + OVERCLEAR)          # obstacle2 at -d2 -> pass +side
    bump = _bump(q, sc["s1"], a1) + _bump(q, sc["s2"], a2)
    bump += rng.normal(0, 0.0015, H + 1) * np.sin(np.pi * q)
    prog[-1] = rad; bump[0] = 0.0; bump[-1] = 0.0
    return np.stack([prog, bump], axis=1)                  # (H+1,2) canonical xy


def collect():
    env = suite.make(env_name="Lift", robots=ARM, controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    adim = env.action_dim
    rng = np.random.default_rng(0)
    scenes = [make_scene(rng) for _ in range(N_SCENES)]
    chunks = np.zeros((N_SCENES, N_DEMOS, H, CDIM))
    succ = np.zeros((N_SCENES, N_DEMOS))
    obsv = np.zeros((N_SCENES, 7))
    for si, sc in enumerate(scenes):
        obsv[si] = obs_vec(sc)
        o1_c = np.array([sc["s1"] * sc["rad"], sc["d1"], 0.0])     # canonical obstacle centers
        o2_c = np.array([sc["s2"] * sc["rad"], -sc["d2"], 0.0])
        for di in range(N_DEMOS):
            o = env.reset()
            p0 = o["robot0_eef_pos"].copy(); q0 = o["robot0_eef_quat"].copy()
            xy = plan_canonical_xy(sc, rng)
            can = np.concatenate([xy, np.zeros((H + 1, 1))], axis=1)   # (H+1,3) canonical
            positions = p0 + can @ Rz(sc["phi"]).T
            tgt = positions[-1]
            obst1 = p0 + Rz(sc["phi"]) @ o1_c; obst2 = p0 + Rz(sc["phi"]) @ o2_c
            pos_tr = [p0.copy()]; quat_tr = [q0.copy()]
            for t in range(H):
                eef = o["robot0_eef_pos"]
                dpos = np.clip(KP * (positions[t + 1] - eef), -1, 1)
                a = np.zeros(adim); a[:3] = dpos
                o, _, _, _ = env.step(a)
                pos_tr.append(o["robot0_eef_pos"].copy()); quat_tr.append(o["robot0_eef_quat"].copy())
            pos_tr = np.array(pos_tr)
            dpos_c = np.diff(pos_tr, axis=0) @ Rz(sc["phi"])
            if CDIM == 6:
                dori_w = np.array([T.quat2axisangle(T.quat_multiply(quat_tr[t + 1], T.quat_inverse(quat_tr[t])))
                                   for t in range(H)])
                chunks[si, di] = np.concatenate([dpos_c, dori_w @ Rz(sc["phi"])], axis=1)
            else:
                chunks[si, di] = dpos_c
            pe = np.linalg.norm(pos_tr[-1] - tgt)
            c1 = (np.linalg.norm(pos_tr - obst1, axis=1) > sc["r1"]).all()
            c2 = (np.linalg.norm(pos_tr - obst2, axis=1) > sc["r2"]).all()
            succ[si, di] = float(pe < TOL_POS and c1 and c2)
    env.close()
    np.savez(os.path.join(OUT, f"{CTRL}_{ARM}.npz"), chunks=chunks, obs=obsv, success=succ)
    print(f"{CTRL}_{ARM} adim={adim} C={CDIM} overclear={OVERCLEAR} demo success {float(succ.mean()):.3f}",
          flush=True)
    print("COLLECT_VARDOF_SLALOM_DONE=ok")


if __name__ == "__main__":
    collect()
