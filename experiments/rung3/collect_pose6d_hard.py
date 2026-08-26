"""Rung 3 Stage 1b: STRONGLY-BOTTLENECKED 6-DOF pose-reach with POSITION-ORIENTATION
COUPLING, to test whether the cross-channel (grid-Laplacian) pin wins LARGE where
per-channel bases and scratch fail.

Task: reach a target while detouring around an obstacle offset to one side of the
start->target line (side is geometry-forced = observable via the offset). The hand
must BANK into the detour: target orientation = rotation about the reach axis by an
angle whose SIGN is the detour side and whose MAGNITUDE scales with the detour
amplitude (COUPLE knob). So position-detour (channels 0-2) and orientation (channels
3-5) are mechanically COORDINATED — a policy/basis that treats channels independently
cannot represent the coupling compactly; the (time x channel) grid Laplacian can.
Orientation is NOT in the observation but is DETERMINED by the observable offset, so
the scene->coefficient prior can still predict it. Bottleneck comes from low data +
the hardness of learning the coupling, not from hidden information.

COUPLE=0 -> orientation decouples from the detour (pure per-channel task) = built-in
control: the grid-Laplacian advantage should VANISH at COUPLE=0 and grow with COUPLE.

Chunk (H,6): [dpos_canonical(3), dori_world(3)]. obs (4): [rad, s_o, r, lateral].
"""
import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

ARM = os.environ.get("SNMVP_ARM", "Panda")
H = 32
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "120"))
N_DEMOS = int(os.environ.get("SNMVP_NDEMOS", "8"))
KP_POS = 12.0
KP_ROT = 5.0
TOL_POS = 0.03
TOL_ROT = 0.15
OVERCLEAR = float(os.environ.get("SNMVP_OVERCLEAR", "0.12"))
BUMP_W = float(os.environ.get("SNMVP_BUMPW", "0.16"))
COUPLE = float(os.environ.get("SNMVP_COUPLE", "1.0"))      # pos-ori coupling strength
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_pose6d_hard")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE")


def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_scene(rng):
    phi = np.radians(rng.uniform(-150.0, 150.0))
    rad = rng.uniform(0.16, 0.24)
    s_o = rng.uniform(0.4, 0.6)
    r = rng.uniform(0.04, 0.06)
    side = rng.choice([-1.0, 1.0])
    lateral = side * rng.uniform(0.02, 0.05)              # offset -> side observable
    return {"phi": phi, "rad": rad, "s_o": s_o, "r": r, "lateral": lateral, "side": side}


def obs_vec(sc):
    return np.array([sc["rad"], sc["s_o"], sc["r"], sc["lateral"]])


def bank_axisangle(sc):
    """Target orientation (world axis-angle): bank about the reach axis, sign =
    detour side, magnitude scales with detour amplitude (COUPLE)."""
    amp = sc["r"] + OVERCLEAR
    ang = COUPLE * (-sc["side"]) * np.clip(3.0 * amp, 0.0, 0.7)     # detour is on -side
    reach_axis_world = Rz(sc["phi"]) @ np.array([1.0, 0.0, 0.0])
    return reach_axis_world * ang


def _bump(q, s_o, amp, w=BUMP_W):
    raw = np.exp(-((q - s_o) ** 2) / (2 * w ** 2))
    ramp = (1 - q) * raw[0] + q * raw[-1]
    peak = 1.0 - ((1 - s_o) * raw[0] + s_o * raw[-1])
    return amp * (raw - ramp) / max(peak, 1e-6)


def plan_world(sc, p0, q0, rng):
    rad, s_o, r, phi = sc["rad"], sc["s_o"], sc["r"], sc["phi"]
    s = np.linspace(0, 1, H + 1)
    p = np.clip(s / 0.72, 0.0, 1.0)
    prog = rad * (3 * p ** 2 - 2 * p ** 3)
    q = prog / rad
    amp = (r + OVERCLEAR) * (1 + rng.normal(0, 0.04))
    side = -np.sign(sc["lateral"])                        # pass on far side
    bump = _bump(q, s_o, side * amp)
    perp = np.array([0.0, 1.0, 0.0])                      # detour in canonical y
    can = np.outer(prog, np.array([1.0, 0.0, 0.0])) + np.outer(bump, perp)
    can[:, 2] += rng.normal(0, 0.002, H + 1) * np.sin(np.pi * q)    # small z style
    can[0] = 0.0; can[-1] = np.array([rad, 0.0, 0.0])
    world = can @ Rz(phi).T
    positions = p0 + world
    aa = bank_axisangle(sc)
    q_tgt = T.quat_multiply(T.axisangle2quat(aa), q0)
    quats = [T.quat_slerp(q0, q_tgt, float(pp)) for pp in p]
    return positions, quats


def collect():
    env = suite.make(env_name="Lift", robots=ARM, controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    rng = np.random.default_rng(0)
    scenes = [make_scene(rng) for _ in range(N_SCENES)]
    chunks = np.zeros((N_SCENES, N_DEMOS, H, 6))
    succ = np.zeros((N_SCENES, N_DEMOS))
    obsv = np.zeros((N_SCENES, 4))
    for si, sc in enumerate(scenes):
        obsv[si] = obs_vec(sc)
        for di in range(N_DEMOS):
            o = env.reset()
            p0 = o["robot0_eef_pos"].copy(); q0 = o["robot0_eef_quat"].copy()
            positions, quats = plan_world(sc, p0, q0, rng)
            tgt_pos = positions[-1]; tgt_quat = quats[-1]
            obst_c = p0 + Rz(sc["phi"]) @ np.array([sc["s_o"] * sc["rad"], sc["lateral"], 0.0])
            pos_tr = [p0.copy()]; quat_tr = [q0.copy()]
            for t in range(H):
                eef = o["robot0_eef_pos"]; qc = o["robot0_eef_quat"]
                dpos = np.clip(KP_POS * (positions[t + 1] - eef), -1, 1)
                err_q = T.quat_multiply(quats[t + 1], T.quat_inverse(qc))
                daa = np.clip(KP_ROT * T.quat2axisangle(err_q), -1, 1)
                o, _, _, _ = env.step(np.concatenate([dpos, daa, [0.0]]))
                pos_tr.append(o["robot0_eef_pos"].copy()); quat_tr.append(o["robot0_eef_quat"].copy())
            pos_tr = np.array(pos_tr)
            dpos_w = np.diff(pos_tr, axis=0)
            dpos_c = dpos_w @ Rz(sc["phi"])                          # canonical position deltas
            dori_w = np.array([T.quat2axisangle(T.quat_multiply(quat_tr[t + 1], T.quat_inverse(quat_tr[t])))
                               for t in range(H)])                   # world axis-angle deltas
            dori_c = dori_w @ Rz(sc["phi"])                          # canonicalize (azimuth-invariant)
            chunks[si, di] = np.concatenate([dpos_c, dori_c], axis=1)
            pe = np.linalg.norm(pos_tr[-1] - tgt_pos)
            re = np.linalg.norm(T.quat2axisangle(T.quat_multiply(tgt_quat, T.quat_inverse(quat_tr[-1]))))
            clr = (np.linalg.norm(pos_tr - obst_c, axis=1) > sc["r"]).all()
            succ[si, di] = float(pe < TOL_POS and re < TOL_ROT and clr)
    env.close()
    np.savez(os.path.join(OUT, f"{ARM}.npz"), chunks=chunks, obs=obsv, success=succ)
    print(ARM, f"COUPLE={COUPLE} demo pose6d_hard success", round(float(succ.mean()), 3), flush=True)
    print("COLLECT_POSE6D_HARD_DONE=ok")


if __name__ == "__main__":
    collect()
