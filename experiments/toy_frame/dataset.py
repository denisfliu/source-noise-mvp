"""Obstacle-detour toy dataset with a PLANTED structure/style split.

Per docs/learned_frame_toy_plan.md (+ the 2026-07-05 reply addendum). Derived
from experiments/toy/toy_flow.py's point-robot conventions (H=20 delta chunks,
ACT_SCALE normalization); scene/demo generation is new.

Scene: target p (radius 1-2, angle uniform in [-180,180)) + one circular
obstacle near the straight start->target line. Obs = (target_xy, obstacle_xy,
obstacle_r), 5 dims.

Planted STRUCTURE (scene-determined, shared by all demos of a scene):
  - endpoint (= target, exact)
  - detour: when the obstacle blocks the corridor, the clearance amplitude of
    the lateral bump (and the side, when geometry forces it)
  - progress timing: slow-down dip near the obstacle's longitudinal position

Planted STYLE (demo-private):
  - bend side when both sides are near-symmetric (kept: it is the diversity
    metric; mod-pi phase coherence must survive it)
  - lateral wiggle (small extra sinusoids, random amp/phase)
  - mild timing jitter

Canonical frame: every chunk is rotated by -target_bearing so progress ~ +x.
The rotation uses scene information only (legitimate at inference).
"""

import numpy as np

H = 20
ACT_SCALE = 5.0
OBST_MARGIN = 0.10
SIDE_STYLE_RATIO = 1.5  # both sides "viable" if deviations within this factor
SUCCESS_TOL = 0.15


def make_scene(rng):
    ang = rng.uniform(-np.pi, np.pi)
    rad = rng.uniform(1.0, 2.0)
    target = rad * np.array([np.cos(ang), np.sin(ang)])
    s_o = rng.uniform(0.35, 0.65)          # longitudinal fraction of obstacle
    lateral = rng.uniform(-0.35, 0.35)     # lateral offset from the line
    obst_r = rng.uniform(0.2, 0.3)
    return {"target": target, "angle": ang, "radius": rad,
            "s_o": s_o, "lateral": lateral, "obst_r": obst_r}


def scene_obs(scene):
    """5-dim observation vector (global frame)."""
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    R = np.array([[c, -s], [s, c]])
    obst_center = R @ np.array([scene["s_o"] * scene["radius"], scene["lateral"]])
    return np.concatenate([scene["target"], obst_center, [scene["obst_r"]]])


def scene_structure(scene):
    """The scene-determined path parameters (canonical frame)."""
    d, R_m = scene["lateral"], scene["obst_r"] + OBST_MARGIN
    blocked = abs(d) < R_m
    if not blocked:
        return {"blocked": False, "amp_up": 0.0, "amp_dn": 0.0, "forced_side": 0}
    # extra headroom beyond tangency so style wiggle can't re-enter the disk
    dev_up = d + R_m + 0.05   # bump value needed to pass above
    dev_dn = d - R_m - 0.05   # (negative) bump value needed to pass below
    if abs(dev_dn) * SIDE_STYLE_RATIO < abs(dev_up):
        forced = -1
    elif abs(dev_up) * SIDE_STYLE_RATIO < abs(dev_dn):
        forced = +1
    else:
        forced = 0            # both viable -> side is a STYLE choice
    return {"blocked": True, "amp_up": dev_up, "amp_dn": dev_dn,
            "forced_side": forced}


def make_demo(scene, rng):
    """One demo chunk (H, 2) in the GLOBAL frame (normalized by ACT_SCALE)."""
    struct = scene_structure(scene)
    rad, s_o = scene["radius"], scene["s_o"]

    # --- timing: structural slow-down near obstacle x mild style jitter ---
    s_grid = np.linspace(0, 1, H + 1)
    speed = np.ones(H + 1)
    if struct["blocked"]:
        speed -= 0.5 * np.exp(-((s_grid - s_o) ** 2) / (2 * 0.12 ** 2))
    gamma = rng.uniform(0.9, 1.1)                       # style
    warp = np.cumsum(speed ** 1.0 * np.gradient(s_grid ** gamma))
    warp = (warp - warp[0]) / (warp[-1] - warp[0])      # progress in [0,1]

    # --- lateral: structural clearance bump + style wiggle ---
    if struct["blocked"]:
        if struct["forced_side"] != 0:
            amp = struct["amp_up"] if struct["forced_side"] > 0 else struct["amp_dn"]
        else:
            side = rng.choice([-1.0, 1.0])              # style
            amp = struct["amp_up"] if side > 0 else struct["amp_dn"]
        bump_w = max(scene["obst_r"] * 1.6 / rad, 0.12)
        # endpoint-vanishing bump: subtract the linear ramp through the raw
        # Gaussian's end values (a raw Gaussian does NOT vanish at s=0/1,
        # which shifted the whole cumsum-built path and biased the endpoint
        # — root cause of the first battery's dirty ceiling), then rescale
        # so the obstacle-center value is exactly `amp`.
        raw = np.exp(-((warp - s_o) ** 2) / (2 * bump_w ** 2))
        ramp = (1 - warp) * raw[0] + warp * raw[-1]
        shape = raw - ramp
        peak = np.exp(-((np.clip(s_o, 0, 1) - s_o) ** 2)) - (
            (1 - s_o) * raw[0] + s_o * raw[-1])
        bump = amp * shape / peak
        # enforce actual clearance of the DISCRETE path (a Gaussian bump can
        # decay faster than the disk edge recedes for near-tangent scenes)
        obst = np.array([s_o * rad, scene["lateral"]])
        for _ in range(12):
            pts = np.stack([warp * rad, bump], axis=1)
            if np.linalg.norm(pts - obst, axis=1).min() >= scene["obst_r"] + 0.06:
                break
            bump = bump * 1.15
        lat = bump
    else:
        lat = np.zeros(H + 1)
    n_wig = rng.integers(1, 3)
    # style wiggle is suppressed near the obstacle so it cannot erode the
    # structural clearance (generator self-collisions broke the success
    # ceiling in the first battery — 2026-07-05 fix, Denis-approved)
    if struct["blocked"]:
        bump_w = max(scene["obst_r"] * 1.6 / rad, 0.12)
        wiggle_env = 1.0 - np.exp(-((warp - s_o) ** 2) / (2 * bump_w ** 2))
    else:
        wiggle_env = np.ones(H + 1)
    for _ in range(n_wig):                              # style wiggle
        k = rng.integers(4, 8)
        lat += rng.uniform(0.02, 0.05) * np.sin(
            np.pi * k * warp + rng.uniform(0, 2 * np.pi)) * np.sin(np.pi * warp) \
            * wiggle_env

    # canonical-frame curve: x = progress * radius, y = lateral
    curve_c = np.stack([warp * rad, lat], axis=1)
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    R = np.array([[c, -s], [s, c]])
    curve = curve_c @ R.T                               # global frame
    chunk = np.diff(curve, axis=0)                      # (H, 2), sums to target
    return chunk * ACT_SCALE


def to_canonical(chunks, angles):
    """Rotate chunks (..., H, 2) by -angle per scene (progress -> +x)."""
    c, s = np.cos(angles), np.sin(angles)
    Rt = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)  # (...,2,2)
    return np.einsum("...hd,...de->...he", chunks, np.swapaxes(Rt, -1, -2))


def make_dataset(n_scenes, n_demos, rng):
    scenes, obs, chunks, angles = [], [], [], []
    for _ in range(n_scenes):
        sc = make_scene(rng)
        scenes.append(sc)
        obs.append(scene_obs(sc))
        angles.append(sc["angle"])
        chunks.append(np.stack([make_demo(sc, rng) for _ in range(n_demos)]))
    return (scenes, np.array(obs), np.stack(chunks),  # (M, N, H, 2)
            np.array(angles))


def success(scene, chunk_normalized):
    """Endpoint within tol AND no timestep inside the obstacle disk."""
    pos = np.cumsum(chunk_normalized / ACT_SCALE, axis=0)
    end_ok = np.linalg.norm(pos[-1] - scene["target"]) < SUCCESS_TOL
    c, s = np.cos(scene["angle"]), np.sin(scene["angle"])
    R = np.array([[c, -s], [s, c]])
    obst = R @ np.array([scene["s_o"] * scene["radius"], scene["lateral"]])
    clear = (np.linalg.norm(pos - obst, axis=1) > scene["obst_r"]).all()
    return bool(end_ok and clear)
