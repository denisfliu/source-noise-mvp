"""Variable-DOF gate collection: the SAME hard position detour-reach collected under
two different controllers / action dimensions, so the shared task (planned position
path) is identical but the achieved realization differs by controller.

  SNMVP_CTRL=pose : OSC_POSE (6-ch action); orientation commanded to zero, stored
                    chunk is the achieved 6-ch canonical pose-delta [dpos(3),dori(3)].
  SNMVP_CTRL=pos  : OSC_POSITION (3-ch action); stored chunk is achieved 3-ch
                    canonical position-delta [dpos(3)].

The obstacle is OFFSET (side forced by its sign) so the planned path is deterministic
from the scene (needed for the planned-pin and the deconvolution reference). Hardness /
bottleneck comes from a TIGHT corridor (small over-clear, larger obstacle) + scarce
training data downstream, not from bimodality. Scenes are seeded identically across
controllers and arms (paired). obs (4)=[rad,s_o,r,lateral]. Success = position reach
within tolerance AND every point clears the obstacle disk.

Env knobs: SNMVP_ARM, SNMVP_CTRL, SNMVP_OVERCLEAR, SNMVP_RLO/RHI. Output:
data_vardof_hard/<ctrl>_<arm>.npz with chunks (S,N,H,C), obs (S,4), success (S,N).
"""
import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

ARM = os.environ.get("SNMVP_ARM", "Panda")
CTRL = os.environ.get("SNMVP_CTRL", "pose")               # pose | pos
H = 32
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "120"))
N_DEMOS = int(os.environ.get("SNMVP_NDEMOS", "8"))
KP = 12.0
TOL_POS = 0.03
OVERCLEAR = float(os.environ.get("SNMVP_OVERCLEAR", "0.06"))   # tight corridor -> bottleneck
BUMP_W = 0.16
RLO = float(os.environ.get("SNMVP_RLO", "0.05"))
RHI = float(os.environ.get("SNMVP_RHI", "0.07"))
CDIM = 6 if CTRL == "pose" else 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_vardof_hard")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE" if CTRL == "pose" else "OSC_POSITION")


def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_scene(rng):
    phi = np.radians(rng.uniform(-150.0, 150.0))
    rad = rng.uniform(0.16, 0.24)
    s_o = rng.uniform(0.4, 0.6)
    r = rng.uniform(RLO, RHI)
    side = rng.choice([-1.0, 1.0])
    lateral = side * rng.uniform(0.03, 0.05)              # offset -> forced side (deterministic plan)
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
    chunks = np.zeros((N_SCENES, N_DEMOS, H, CDIM))
    succ = np.zeros((N_SCENES, N_DEMOS))
    obsv = np.zeros((N_SCENES, 4))
    for si, sc in enumerate(scenes):
        obsv[si] = obs_vec(sc)
        for di in range(N_DEMOS):
            o = env.reset()
            p0 = o["robot0_eef_pos"].copy(); q0 = o["robot0_eef_quat"].copy()
            positions = plan_world(sc, p0, rng)
            tgt = positions[-1]
            obst_c = p0 + Rz(sc["phi"]) @ np.array([sc["s_o"] * sc["rad"], sc["lateral"], 0.0])
            pos_tr = [p0.copy()]; quat_tr = [q0.copy()]
            for t in range(H):
                eef = o["robot0_eef_pos"]
                dpos = np.clip(KP * (positions[t + 1] - eef), -1, 1)
                a = np.zeros(adim); a[:3] = dpos          # orientation commanded zero (pose) / absent (pos)
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
            clr = (np.linalg.norm(pos_tr - obst_c, axis=1) > sc["r"]).all()
            succ[si, di] = float(pe < TOL_POS and clr)
    env.close()
    np.savez(os.path.join(OUT, f"{CTRL}_{ARM}.npz"), chunks=chunks, obs=obsv, success=succ)
    print(f"{CTRL}_{ARM} adim={adim} C={CDIM} overclear={OVERCLEAR} demo success {float(succ.mean()):.3f}",
          flush=True)
    print("COLLECT_VARDOF_DONE=ok")


if __name__ == "__main__":
    collect()
