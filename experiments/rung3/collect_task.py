"""Controlled task-by-embodiment collection. One arm under the fixed six-channel
OSC_POSE interface performs one of three tasks with different action structure, so
that a later study can hold the task and vary the arm, or hold the arm and vary the
task. Tasks: bank, a lateral detour around an obstacle with a roll about the reach
axis; vertical, a vertical detour with a pitch about the lateral axis; slalom, a
two-obstacle lateral S-curve with no orientation change. Stored chunk (H,6) is the
achieved canonical pose-delta [dpos(3), dori(3)]. Scenes are seeded identically per
task, so arms of the same task share scenes.

Env: SNMVP_TASK in {bank, vertical, slalom}, SNMVP_ARM.
"""
import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

ARM = os.environ.get("SNMVP_ARM", "Panda")
TASK = os.environ.get("SNMVP_TASK", "bank")
H = 32
N_SCENES = int(os.environ.get("SNMVP_NSCENES", "80"))
N_DEMOS = int(os.environ.get("SNMVP_NDEMOS", "6"))
KP_POS, KR_POS = 12.0, 5.0
TOL_POS, TOL_ROT = 0.03, 0.15
OVERCLEAR, BUMP_W = 0.12, 0.16
CFG = {
    "bank":     dict(detour=np.array([0., 1., 0.]), oaxis=np.array([1., 0., 0.]), n_obst=1, oscale=1.0),
    "vertical": dict(detour=np.array([0., 0., 1.]), oaxis=np.array([0., 1., 0.]), n_obst=1, oscale=1.0),
    "slalom":   dict(detour=np.array([0., 1., 0.]), oaxis=np.array([1., 0., 0.]), n_obst=2, oscale=0.0),
}[TASK]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_taskembod")
os.makedirs(OUT, exist_ok=True)
cfg = load_controller_config(default_controller="OSC_POSE")


def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def make_scene(rng):
    phi = np.radians(rng.uniform(-150., 150.))
    rad = rng.uniform(0.16, 0.24)
    if CFG["n_obst"] == 1:
        s_o = rng.uniform(0.4, 0.6); r = rng.uniform(0.04, 0.06)
        side = rng.choice([-1., 1.]); off = side * rng.uniform(0.02, 0.05)
        return dict(phi=phi, rad=rad, obst=[(s_o, r, off, side)])
    g = rng.choice([-1., 1.])
    o1 = (rng.uniform(0.26, 0.36), rng.uniform(0.03, 0.05), g * rng.uniform(0.02, 0.05), g)
    o2 = (rng.uniform(0.64, 0.74), rng.uniform(0.03, 0.05), -g * rng.uniform(0.02, 0.05), -g)
    return dict(phi=phi, rad=rad, obst=[o1, o2])


def _bump(q, s_o, amp):
    raw = np.exp(-((q - s_o) ** 2) / (2 * BUMP_W ** 2))
    ramp = (1 - q) * raw[0] + q * raw[-1]
    peak = 1. - ((1 - s_o) * raw[0] + s_o * raw[-1])
    return amp * (raw - ramp) / max(peak, 1e-6)


def plan_world(sc, p0, q0, rng):
    rad, phi = sc["rad"], sc["phi"]
    s = np.linspace(0, 1, H + 1)
    p = np.clip(s / 0.72, 0., 1.)
    prog = rad * (3 * p ** 2 - 2 * p ** 3)
    q = prog / rad
    lat = np.zeros((H + 1, 3))
    for (s_o, r, off, side) in sc["obst"]:
        amp = (r + OVERCLEAR) * (1 + rng.normal(0, 0.04))
        lat += np.outer(_bump(q, s_o, -np.sign(off) * amp), CFG["detour"])
    can = np.outer(prog, np.array([1., 0., 0.])) + lat
    can[0] = 0.; can[-1] = np.array([rad, 0., 0.])
    positions = p0 + can @ Rz(phi).T
    if CFG["oscale"] > 0:
        side = sc["obst"][0][3]; r = sc["obst"][0][1]
        ang = CFG["oscale"] * (-side) * np.clip(3. * (r + OVERCLEAR), 0., 0.7)
        aa_world = (Rz(phi) @ CFG["oaxis"]) * ang
        q_tgt = T.quat_multiply(T.axisangle2quat(aa_world), q0)
        quats = [T.quat_slerp(q0, q_tgt, float(pp)) for pp in p]
    else:
        quats = [q0.copy() for _ in p]
    return positions, quats


def obst_world(sc, p0, s_o, off):
    return p0 + Rz(sc["phi"]) @ (np.array([s_o * sc["rad"], 0., 0.]) + off * CFG["detour"])


def collect():
    env = suite.make(env_name="Lift", robots=ARM, controller_configs=cfg, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, control_freq=20)
    rng = np.random.default_rng(0)
    scenes = [make_scene(rng) for _ in range(N_SCENES)]
    chunks = np.zeros((N_SCENES, N_DEMOS, H, 6)); succ = np.zeros((N_SCENES, N_DEMOS))
    for si, sc in enumerate(scenes):
        for di in range(N_DEMOS):
            o = env.reset(); p0 = o["robot0_eef_pos"].copy(); q0 = o["robot0_eef_quat"].copy()
            positions, quats = plan_world(sc, p0, q0, rng)
            tgt_pos, tgt_q = positions[-1], quats[-1]
            obsts = [obst_world(sc, p0, s_o, off) for (s_o, r, off, side) in sc["obst"]]
            radii = [r for (s_o, r, off, side) in sc["obst"]]
            ptr = [p0.copy()]; qtr = [q0.copy()]
            for t in range(H):
                eef = o["robot0_eef_pos"]; qc = o["robot0_eef_quat"]
                dpos = np.clip(KP_POS * (positions[t + 1] - eef), -1, 1)
                eq = T.quat_multiply(quats[t + 1], T.quat_inverse(qc))
                daa = np.clip(KR_POS * T.quat2axisangle(eq), -1, 1)
                o, _, _, _ = env.step(np.concatenate([dpos, daa, [0.]]))
                ptr.append(o["robot0_eef_pos"].copy()); qtr.append(o["robot0_eef_quat"].copy())
            ptr = np.array(ptr)
            dpos_c = np.diff(ptr, axis=0) @ Rz(sc["phi"])
            dori_w = np.array([T.quat2axisangle(T.quat_multiply(qtr[t + 1], T.quat_inverse(qtr[t]))) for t in range(H)])
            chunks[si, di] = np.concatenate([dpos_c, dori_w @ Rz(sc["phi"])], axis=1)
            pe = np.linalg.norm(ptr[-1] - tgt_pos)
            re = np.linalg.norm(T.quat2axisangle(T.quat_multiply(tgt_q, T.quat_inverse(qtr[-1]))))
            clr = all((np.linalg.norm(ptr - oc, axis=1) > rr).all() for oc, rr in zip(obsts, radii))
            succ[si, di] = float(pe < TOL_POS and (re < TOL_ROT or CFG["oscale"] == 0) and clr)
    env.close()
    np.savez(os.path.join(OUT, f"{TASK}_{ARM}.npz"), chunks=chunks, success=succ)
    print(f"{TASK}_{ARM}: demo success {float(succ.mean()):.3f}", flush=True)
    print("COLLECT_TASK_DONE=ok")


if __name__ == "__main__":
    collect()
