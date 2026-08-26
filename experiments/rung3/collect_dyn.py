"""Sim-to-real study: collect the coupled 6-DOF pose task on one arm (Panda) under
different DYNAMICS while holding the action interface fixed (OSC_POSE, six channels).
A dynamics variant is set by the low-level controller position gain (kp), its damping
ratio, and an actuation latency (the executed command is delayed by LAT control steps).
Several variants stand in for simulated training bodies and one variant with values
outside their range stands in for the physical system. The stored chunk is the
achieved six-channel pose-delta trajectory in the canonical frame, so the task
structure is shared across variants while the realization differs by dynamics.

Env knobs: SNMVP_KP, SNMVP_DAMP, SNMVP_LAT, SNMVP_VNAME (output file name).
Writes data_dyn/<VNAME>.npz with chunks (S,N,H,6), obs (S,4), success (S,N).
"""
import os
from collections import deque
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

ARM = "Panda"
H = 32
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "120"))
N_DEMOS = int(os.environ.get("SNMVP_NDEMOS", "8"))
KP_TRACK = 12.0
KR_TRACK = 5.0
TOL_POS = 0.03
TOL_ROT = 0.15
OVERCLEAR = 0.12
BUMP_W = 0.16
COUPLE = 1.0
KP_OSC = float(os.environ.get("SNMVP_KP", "150"))
DAMP = float(os.environ.get("SNMVP_DAMP", "1.0"))
LAT = int(os.environ.get("SNMVP_LAT", "0"))
VNAME = os.environ.get("SNMVP_VNAME", f"kp{int(KP_OSC)}d{DAMP}l{LAT}")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_dyn")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE")
cfg["kp"] = KP_OSC
cfg["damping_ratio"] = DAMP


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
    return {"phi": phi, "rad": rad, "s_o": s_o, "r": r, "lateral": lateral, "side": side}


def obs_vec(sc):
    return np.array([sc["rad"], sc["s_o"], sc["r"], sc["lateral"]])


def bank_axisangle(sc):
    amp = sc["r"] + OVERCLEAR
    ang = COUPLE * (-sc["side"]) * np.clip(3.0 * amp, 0.0, 0.7)
    return (Rz(sc["phi"]) @ np.array([1.0, 0.0, 0.0])) * ang


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
    side = -np.sign(sc["lateral"])
    bump = _bump(q, s_o, side * amp)
    can = np.outer(prog, np.array([1.0, 0.0, 0.0])) + np.outer(bump, np.array([0.0, 1.0, 0.0]))
    can[:, 2] += rng.normal(0, 0.002, H + 1) * np.sin(np.pi * q)
    can[0] = 0.0; can[-1] = np.array([rad, 0.0, 0.0])
    positions = p0 + can @ Rz(phi).T
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
            buf = deque([np.zeros(env.action_dim)] * (LAT + 1), maxlen=LAT + 1)
            for t in range(H):
                eef = o["robot0_eef_pos"]; qc = o["robot0_eef_quat"]
                dpos = np.clip(KP_TRACK * (positions[t + 1] - eef), -1, 1)
                err_q = T.quat_multiply(quats[t + 1], T.quat_inverse(qc))
                daa = np.clip(KR_TRACK * T.quat2axisangle(err_q), -1, 1)
                buf.append(np.concatenate([dpos, daa, [0.0]]))
                o, _, _, _ = env.step(buf[0])              # delayed command (latency = LAT)
                pos_tr.append(o["robot0_eef_pos"].copy()); quat_tr.append(o["robot0_eef_quat"].copy())
            pos_tr = np.array(pos_tr)
            dpos_c = np.diff(pos_tr, axis=0) @ Rz(sc["phi"])
            dori_w = np.array([T.quat2axisangle(T.quat_multiply(quat_tr[t + 1], T.quat_inverse(quat_tr[t])))
                               for t in range(H)])
            dori_c = dori_w @ Rz(sc["phi"])
            chunks[si, di] = np.concatenate([dpos_c, dori_c], axis=1)
            pe = np.linalg.norm(pos_tr[-1] - tgt_pos)
            re = np.linalg.norm(T.quat2axisangle(T.quat_multiply(tgt_quat, T.quat_inverse(quat_tr[-1]))))
            clr = (np.linalg.norm(pos_tr - obst_c, axis=1) > sc["r"]).all()
            succ[si, di] = float(pe < TOL_POS and re < TOL_ROT and clr)
    env.close()
    np.savez(os.path.join(OUT, f"{VNAME}.npz"), chunks=chunks, obs=obsv, success=succ)
    print(f"{VNAME} kp={KP_OSC} damp={DAMP} lat={LAT} demo success {float(succ.mean()):.3f}", flush=True)
    print("COLLECT_DYN_DONE=ok")


if __name__ == "__main__":
    collect()
