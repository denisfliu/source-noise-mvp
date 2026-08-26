"""Rung 3 Stage 1: 6-DOF pose-reach around an obstacle (real Panda, closed-loop).

Higher-DOF extension of the planar obstacle task: the EE must reach a target
POSITION and ORIENTATION while detouring around a spherical obstacle. Full
OSC_POSE action (dpos 3 + daxisangle 3). Demos plan a world pose path (smoothstep
longitudinal progress + endpoint-vanishing clearance bump in a scene-chosen y-z
direction; orientation slerp from start to target synced to progress) and
OSC-track it in sim (position + orientation P-control, verified by servo6d_test).
The stored chunk is the ACHIEVED 6-DOF pose-delta trajectory, so success reflects
real closed-loop tracking dynamics, not an idealized plan.

Stored chunk channels (H,6): [dpos_canonical(3), dori_world(3)]. Position is
canonicalized by the reach azimuth (Rz(-phi): reach -> +x) so structure is
azimuth-invariant; orientation deltas are world-frame axis-angle (azimuth-
independent by construction). obs (canonical scene descriptor, 8):
[rad, s_o, r, cos_psi, sin_psi, aa(3)].
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
TOL_ROT = 0.15                                   # rad
OVERCLEAR = float(os.environ.get("SNMVP_OVERCLEAR", "0.12"))
BUMP_W = float(os.environ.get("SNMVP_BUMPW", "0.16"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_pose6d")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE")


def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_scene(rng):
    phi = np.radians(rng.uniform(-150.0, 150.0))     # reach azimuth (world), rear cone excluded
    rad = rng.uniform(0.16, 0.24)
    s_o = rng.uniform(0.4, 0.6)                       # obstacle along the line
    r = rng.uniform(0.03, 0.05)
    psi = rng.uniform(0, np.pi)                       # detour plane direction in y-z
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = rng.uniform(0.2, 0.6)                       # target rotation magnitude (rad)
    aa = axis * ang                                   # world-frame axis-angle
    return {"phi": phi, "rad": rad, "s_o": s_o, "r": r, "psi": psi, "aa": aa}


def obs_vec(sc):
    return np.array([sc["rad"], sc["s_o"], sc["r"],
                     np.cos(sc["psi"]), np.sin(sc["psi"]), *sc["aa"]])


def _bump(q, s_o, amp, w=BUMP_W):
    raw = np.exp(-((q - s_o) ** 2) / (2 * w ** 2))
    ramp = (1 - q) * raw[0] + q * raw[-1]
    peak = 1.0 - ((1 - s_o) * raw[0] + s_o * raw[-1])
    return amp * (raw - ramp) / max(peak, 1e-6)


def plan_world(sc, p0, q0, rng):
    """Planned world pose path (H+1): positions (H+1,3), quats list (H+1,)."""
    rad, s_o, r, phi = sc["rad"], sc["s_o"], sc["r"], sc["phi"]
    s = np.linspace(0, 1, H + 1)
    p = np.clip(s / 0.72, 0.0, 1.0)
    prog = rad * (3 * p ** 2 - 2 * p ** 3)               # smoothstep, dwell after s=0.72
    q = prog / rad                                       # true longitudinal fraction
    amp = (r + OVERCLEAR) * (1 + rng.normal(0, 0.04))
    bump = _bump(q, s_o, amp)
    perp = np.array([0.0, np.cos(sc["psi"]), np.sin(sc["psi"])])
    can = np.outer(prog, np.array([1.0, 0.0, 0.0])) + np.outer(bump, perp)   # canonical pos
    can[:, 1] += rng.normal(0, 0.002, H + 1) * np.sin(np.pi * q)             # style
    can[0] = 0.0; can[-1] = np.array([rad, 0.0, 0.0])
    world = can @ Rz(phi).T                              # (H+1,3) world offsets
    positions = p0 + world
    # orientation: slerp start->target synced to smoothstep progress
    q_tgt = T.quat_multiply(T.axisangle2quat(sc["aa"]), q0)
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
    obsv = np.zeros((N_SCENES, 8))
    for si, sc in enumerate(scenes):
        obsv[si] = obs_vec(sc)
        q_tgt_aa = sc["aa"]
        for di in range(N_DEMOS):
            o = env.reset()
            p0 = o["robot0_eef_pos"].copy(); q0 = o["robot0_eef_quat"].copy()
            positions, quats = plan_world(sc, p0, q0, rng)
            tgt_pos = positions[-1]; tgt_quat = quats[-1]
            obst_c = p0 + Rz(sc["phi"]) @ np.array([sc["s_o"] * sc["rad"], 0.0, 0.0])
            pos_tr = [p0.copy()]; quat_tr = [q0.copy()]
            for t in range(H):
                eef = o["robot0_eef_pos"]; qc = o["robot0_eef_quat"]
                dpos = np.clip(KP_POS * (positions[t + 1] - eef), -1, 1)
                err_q = T.quat_multiply(quats[t + 1], T.quat_inverse(qc))
                daa = np.clip(KP_ROT * T.quat2axisangle(err_q), -1, 1)
                o, _, _, _ = env.step(np.concatenate([dpos, daa, [0.0]]))
                pos_tr.append(o["robot0_eef_pos"].copy()); quat_tr.append(o["robot0_eef_quat"].copy())
            pos_tr = np.array(pos_tr)
            # achieved deltas -> canonical pos, world ori
            dpos_w = np.diff(pos_tr, axis=0)                       # (H,3) world
            dpos_c = dpos_w @ Rz(sc["phi"])                        # Rz(-phi) @ v == v @ Rz(phi)
            dori = np.array([T.quat2axisangle(T.quat_multiply(quat_tr[t + 1], T.quat_inverse(quat_tr[t])))
                             for t in range(H)])                   # (H,3) world axis-angle
            chunks[si, di] = np.concatenate([dpos_c, dori], axis=1)
            # success: reach pos+ori, clear obstacle
            pe = np.linalg.norm(pos_tr[-1] - tgt_pos)
            re = np.linalg.norm(T.quat2axisangle(T.quat_multiply(tgt_quat, T.quat_inverse(quat_tr[-1]))))
            clr = (np.linalg.norm(pos_tr - obst_c, axis=1) > sc["r"]).all()
            succ[si, di] = float(pe < TOL_POS and re < TOL_ROT and clr)
    env.close()
    np.savez(os.path.join(OUT, f"{ARM}.npz"), chunks=chunks, obs=obsv, success=succ)
    print(ARM, "demo pose6d success", round(float(succ.mean()), 3),
          "| pos_reach+ori+clear", flush=True)
    print("COLLECT_POSE6D_DONE=ok")


if __name__ == "__main__":
    collect()
