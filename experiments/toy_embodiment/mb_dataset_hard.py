"""Multi-obstacle cross-embodiment dataset — the task-COMPLEXITY axis.

Tests Denis's hypothesis (2026-07-20): exact source-noise pinning / the
coherence frame win on the single-obstacle toy only because its shared structure
is ~10 bits and LINEARLY pinnable (a few exact FFT phases). Raising the number
of obstacles makes the feasible detour (a) higher-dimensional and (b) NONLINEAR
in obstacle layout — no longer a clean few-FFT-bin structure — which should
degrade the coherence frame while a learned variable-depth OAT bottleneck holds.
The predicted signature is a CROSSOVER in transfer performance vs complexity.

Self-contained (does not modify toy_frame). Same conventions: canonical frame
(progress -> +x), H=20 tip-delta chunks, ACT_SCALE normalization; bodies realize
the planned tip path via embodiments.py (task-space actions, pin exact).

obs = target_xy (2) + per-obstacle [center_xy, r] for MAX_OBST slots (zero-padded
for absent obstacles) = 2 + 3*MAX_OBST dims.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "toy_frame"))
import dataset as tf                    # noqa: E402  (H, ACT_SCALE, to_canonical)

H = tf.H
ACT_SCALE = tf.ACT_SCALE
MAX_OBST = 4
OBS_DIM = 2 + 3 * MAX_OBST             # 14
OBST_MARGIN = 0.10
SUCCESS_TOL = 0.15
CLEAR = 0.06                            # required clearance beyond disk edge


def make_scene(rng, n_obst):
    ang = rng.uniform(-np.pi, np.pi)
    rad = rng.uniform(1.0, 1.7)
    # obstacles in separated longitudinal bands, small enough that their x-ranges
    # don't overlap -> a monotonic-progress path lat(x) can always clear each
    # (difficulty comes from the NUMBER of detours = richer shared structure,
    # not from geometrically infeasible threading).
    bands = np.linspace(0.22, 0.78, n_obst)
    obst = []
    for i in range(n_obst):
        s_o = float(bands[i] + rng.uniform(-0.03, 0.03))
        lateral = float(rng.uniform(-0.28, 0.28))
        obst_r = float(rng.uniform(0.12, 0.18))
        obst.append({"s_o": s_o, "lateral": lateral, "obst_r": obst_r})
    return {"target": rad * np.array([np.cos(ang), np.sin(ang)]),
            "angle": ang, "radius": rad, "obst": obst, "n_obst": n_obst}


def scene_obs(scene):
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    R = np.array([[c, -s], [s, c]])
    v = [scene["target"]]
    for o in scene["obst"]:
        center = R @ np.array([o["s_o"] * scene["radius"], o["lateral"]])
        v.append(np.concatenate([center, [o["obst_r"]]]))
    for _ in range(MAX_OBST - scene["n_obst"]):
        v.append(np.zeros(3))
    return np.concatenate(v)


def _one_bump(warp, s_o, amp, bump_w):
    """Endpoint-vanishing lateral bump centered at longitudinal fraction s_o
    (same construction as toy_frame.make_demo, per-obstacle)."""
    raw = np.exp(-((warp - s_o) ** 2) / (2 * bump_w ** 2))
    ramp = (1 - warp) * raw[0] + warp * raw[-1]
    shape = raw - ramp
    peak = np.exp(0.0) - ((1 - s_o) * raw[0] + s_o * raw[-1])
    return amp * shape / max(peak, 1e-6)


def make_demo(scene, rng):
    """One canonical-frame demo chunk (H,2) (progress x, lateral y), normalized.
    Structure (scene-determined, shared across demos+bodies): each obstacle's
    clearance bump on a geometry-forced side. Style (demo-private): small
    amplitude jitter + a lateral wiggle away from the obstacles."""
    rad = scene["radius"]
    s_grid = np.linspace(0, 1, H + 1)
    # timing: structural slow-down near every obstacle
    speed = np.ones(H + 1)
    for o in scene["obst"]:
        speed -= 0.4 * np.exp(-((s_grid - o["s_o"]) ** 2) / (2 * 0.10 ** 2))
    speed = np.clip(speed, 0.2, None)
    gamma = rng.uniform(0.9, 1.1)
    warp = np.cumsum(speed * np.gradient(s_grid ** gamma))
    warp = (warp - warp[0]) / (warp[-1] - warp[0])

    # geometry-forced side per obstacle (SHARED, scene-determined structure).
    sides, bws = [], []
    for o in scene["obst"]:
        Rm = o["obst_r"] + OBST_MARGIN
        sides.append(1.0 if abs(o["lateral"] + Rm) <= abs(o["lateral"] - Rm) else -1.0)
        bws.append(max(o["obst_r"] * 1.6 / rad, 0.12))

    # lateral = sum_k a_k * bump_k, amplitudes solved so the path hits a
    # clearance waypoint at every obstacle's s_o simultaneously (handles
    # overlapping bumps exactly; matrix is diagonally dominant since obstacles
    # are s-separated). Endpoint stays 0 (each bump is endpoint-vanishing).
    n = len(scene["obst"])
    basis = [_one_bump(warp, o["s_o"], 1.0, bw) for o, bw in zip(scene["obst"], bws)]
    s_idx = [int(np.argmin(np.abs(warp - o["s_o"]))) for o in scene["obst"]]
    M = np.array([[basis[k][s_idx[j]] for k in range(n)] for j in range(n)])
    t = np.array([o["lateral"] + side * (o["obst_r"] + CLEAR + 0.05)
                  for o, side in zip(scene["obst"], sides)])
    a = np.linalg.solve(M + 1e-6 * np.eye(n), t)
    lat = sum(a[k] * basis[k] for k in range(n))

    # style wiggle, suppressed near obstacles
    env = np.ones(H + 1)
    for o in scene["obst"]:
        bw = max(o["obst_r"] * 1.6 / rad, 0.12)
        env = env * (1.0 - np.exp(-((warp - o["s_o"]) ** 2) / (2 * bw ** 2)))
    for _ in range(rng.integers(1, 3)):
        k = rng.integers(4, 8)
        lat += rng.uniform(0.02, 0.04) * np.sin(
            np.pi * k * warp + rng.uniform(0, 2 * np.pi)) * np.sin(np.pi * warp) * env

    # hard per-timestep lateral projection: guarantees the planned path clears
    # every disk at every timestep (so the point robot's ceiling is ~1.0 by
    # construction — the shared structure is a genuinely feasible solution). For
    # each timestep at longitudinal x, an obstacle at (cx,cy,R') blocks the
    # lateral band cy +/- sqrt(R'^2-(x-cx)^2); push lat out to the edge on the
    # obstacle's chosen side. Two passes handle timesteps blocked by two disks.
    x = warp * rad
    for _ in range(12):                                  # iterate to a fixed point
        moved = False
        for o, side in zip(scene["obst"], sides):
            cx, cy, R2 = o["s_o"] * rad, o["lateral"], (o["obst_r"] + CLEAR + 0.02) ** 2
            dx2 = R2 - (x - cx) ** 2
            band = np.sqrt(np.clip(dx2, 0.0, None))
            blocked = (dx2 > 0) & (side * (lat - cy) < band)
            if blocked.any():
                lat = np.where(blocked, cy + side * band, lat)
                moved = True
        if not moved:
            break
    lat[0] = 0.0; lat[-1] = 0.0                          # endpoints exact

    curve_c = np.stack([warp * rad, lat], axis=1)
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    R = np.array([[c, -s], [s, c]])
    curve = curve_c @ R.T
    return np.diff(curve, axis=0) * ACT_SCALE            # global-frame (H,2)


def planned_positions(scene, rng):
    chunk = make_demo(scene, rng) / ACT_SCALE
    return np.concatenate([[[0.0, 0.0]], np.cumsum(chunk, axis=0)], axis=0)


def body_demo(body, scene, rng):
    P = planned_positions(scene, rng)
    return np.diff(body.realize(P), axis=0) * ACT_SCALE


def make_dataset(bodies, n_scenes, n_demos, rng, n_obst=1):
    scenes, obs, angles = [], [], []
    chunks = {b: [] for b in bodies}
    for _ in range(n_scenes):
        sc = make_scene(rng, n_obst)
        scenes.append(sc); obs.append(scene_obs(sc)); angles.append(sc["angle"])
        for name, body in bodies.items():
            chunks[name].append(np.stack([body_demo(body, sc, rng)
                                          for _ in range(n_demos)]))
    return (scenes, np.array(obs), np.array(angles),
            {b: np.stack(v) for b, v in chunks.items()})


def success(scene, chunk_world_normalized):
    pos = np.cumsum(chunk_world_normalized / ACT_SCALE, axis=0)
    if np.linalg.norm(pos[-1] - scene["target"]) >= SUCCESS_TOL:
        return False
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    R = np.array([[c, -s], [s, c]])
    for o in scene["obst"]:
        obst = R @ np.array([o["s_o"] * scene["radius"], o["lateral"]])
        if (np.linalg.norm(pos - obst, axis=1) <= o["obst_r"]).any():
            return False
    return True
