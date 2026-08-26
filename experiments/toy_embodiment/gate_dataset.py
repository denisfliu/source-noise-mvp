"""Gate-passage cross-embodiment task — the drone-navigation analog.

North star: one-shot an IRL drone through a gate (multi-gate = drone racing).
This is the 2D toy proxy. A GATE is an aperture the path must pass THROUGH at a
longitudinal plane (inverts the obstacle-detour task: be INSIDE the slot at the
gate plane, not outside a disk). n_gates in {1,2,3} = increasing racing
difficulty. Drone-analog bodies: `point` (ideal quad), `point_drag` (inertia —
can't turn sharply, the realistic drone that overshoots narrow apertures), arms
(reach-limited). The SHARED structure is the gate-center sequence (observable in
obs); the EMBODIMENT-private part is how a body realizes the commanded weave —
which is exactly what a shared executor prior must supply for one-shot transfer.

Same conventions as mb_dataset_hard: canonical frame (progress -> +x), H=20
tip-delta chunks, ACT_SCALE normalization, bodies realize the planned tip path.

obs = target_xy (2) + per-gate [center_xy, halfwidth] for MAX_GATE slots
(zero-padded) = 2 + 3*MAX_GATE dims.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "toy_frame"))
import dataset as tf                    # noqa: E402

H = tf.H
ACT_SCALE = tf.ACT_SCALE
MAX_GATE = 3
OBS_DIM = 2 + 3 * MAX_GATE             # 11
SUCCESS_TOL = 0.15


def make_scene(rng, n_gates):
    ang = rng.uniform(-np.pi, np.pi)
    rad = rng.uniform(1.2, 1.8)
    bands = np.linspace(0.30, 0.75, n_gates)
    gates = []
    for i in range(n_gates):
        s_g = float(bands[i] + rng.uniform(-0.03, 0.03))
        center = float(rng.uniform(-0.40, 0.40))         # lateral center of aperture
        halfwidth = float(rng.uniform(0.10, 0.18))       # narrow -> precise passage
        gates.append({"s_g": s_g, "center": center, "hw": halfwidth})
    return {"target": rad * np.array([np.cos(ang), np.sin(ang)]),
            "angle": ang, "radius": rad, "gates": gates, "n_gates": n_gates}


def scene_obs(scene):
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    R = np.array([[c, -s], [s, c]])
    v = [scene["target"]]
    for g in scene["gates"]:
        center = R @ np.array([g["s_g"] * scene["radius"], g["center"]])
        v.append(np.concatenate([center, [g["hw"]]]))
    for _ in range(MAX_GATE - scene["n_gates"]):
        v.append(np.zeros(3))
    return np.concatenate(v)


def _bump(warp, s0, width):
    """Endpoint-vanishing localized basis centered at longitudinal fraction s0."""
    raw = np.exp(-((warp - s0) ** 2) / (2 * width ** 2))
    ramp = (1 - warp) * raw[0] + warp * raw[-1]
    shape = raw - ramp
    peak = 1.0 - ((1 - s0) * raw[0] + s0 * raw[-1])
    return shape / max(peak, 1e-6)


def make_demo(scene, rng):
    """One canonical-frame demo (H,2). Structure (scene-determined, shared): the
    path passes through each gate's aperture center. Style (demo-private): the
    exact lateral WITHIN each aperture (jittered inside the slot) + timing +
    a small wiggle -> the model must learn the gate-center sequence as the shared
    invariant while the within-slot choice stays private."""
    rad, gates = scene["radius"], scene["gates"]
    s_grid = np.linspace(0, 1, H + 1)
    gamma = rng.uniform(0.9, 1.1)                         # timing style
    warp = (s_grid ** gamma)
    warp = (warp - warp[0]) / (warp[-1] - warp[0])

    n = len(gates)
    widths = [max(0.10, 0.14) for _ in gates]
    basis = [_bump(warp, g["s_g"], w) for g, w in zip(gates, widths)]
    s_idx = [int(np.argmin(np.abs(warp - g["s_g"]))) for g in gates]
    M = np.array([[basis[k][s_idx[j]] for k in range(n)] for j in range(n)])
    # target lateral at each gate = aperture center + within-slot style jitter
    t = np.array([g["center"] + rng.uniform(-0.5, 0.5) * g["hw"] for g in gates])
    a = np.linalg.solve(M + 1e-6 * np.eye(n), t)
    lat = sum(a[k] * basis[k] for k in range(n))

    # small style wiggle away from gate planes (kept inside apertures near them)
    env = np.ones(H + 1)
    for g, w in zip(gates, widths):
        env = env * (1.0 - np.exp(-((warp - g["s_g"]) ** 2) / (2 * w ** 2)))
    for _ in range(rng.integers(1, 3)):
        k = rng.integers(4, 8)
        lat += rng.uniform(0.015, 0.03) * np.sin(
            np.pi * k * warp + rng.uniform(0, 2 * np.pi)) * np.sin(np.pi * warp) * env
    lat[0] = 0.0; lat[-1] = 0.0

    curve_c = np.stack([warp * rad, lat], axis=1)
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    R = np.array([[c, -s], [s, c]])
    return np.diff(curve_c @ R.T, axis=0) * ACT_SCALE


def planned_positions(scene, rng):
    chunk = make_demo(scene, rng) / ACT_SCALE
    return np.concatenate([[[0.0, 0.0]], np.cumsum(chunk, axis=0)], axis=0)


def body_demo(body, scene, rng):
    P = planned_positions(scene, rng)
    return np.diff(body.realize(P), axis=0) * ACT_SCALE


def make_dataset(bodies, n_scenes, n_demos, rng, n_gates=1):
    scenes, obs, angles = [], [], []
    chunks = {b: [] for b in bodies}
    for _ in range(n_scenes):
        sc = make_scene(rng, n_gates)
        scenes.append(sc); obs.append(scene_obs(sc)); angles.append(sc["angle"])
        for name, body in bodies.items():
            chunks[name].append(np.stack([body_demo(body, sc, rng)
                                          for _ in range(n_demos)]))
    return (scenes, np.array(obs), np.array(angles),
            {b: np.stack(v) for b, v in chunks.items()})


def success(scene, chunk_world_normalized):
    """Passed = reaches target AND crosses every gate plane within its aperture.
    The gate plane is x = s_g*radius in the canonical frame; passing outside
    [center +/- hw] there = hitting the wall."""
    pos = np.cumsum(chunk_world_normalized / ACT_SCALE, axis=0)
    if np.linalg.norm(pos[-1] - scene["target"]) >= SUCCESS_TOL:
        return False
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    Rt = np.array([[c, s], [-s, c]])                     # world -> canonical
    can = pos @ Rt.T
    x, y = can[:, 0], can[:, 1]
    for g in scene["gates"]:
        xg = g["s_g"] * scene["radius"]
        # lateral where the path crosses the gate plane (linear interp in x)
        i = int(np.argmin(np.abs(x - xg)))
        if i == 0 or i == len(x) - 1:
            y_cross = y[i]
        else:
            j = i + 1 if x[i] < xg else i - 1
            j = min(max(j, 0), len(x) - 1)
            denom = (x[j] - x[i])
            frac = 0.0 if abs(denom) < 1e-9 else (xg - x[i]) / denom
            y_cross = y[i] + frac * (y[j] - y[i])
        if abs(y_cross - g["center"]) > g["hw"]:
            return False
    return True
