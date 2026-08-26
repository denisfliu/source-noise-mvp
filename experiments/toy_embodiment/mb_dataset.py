"""Multi-embodiment obstacle-detour dataset (Rung 1).

Reuses toy_frame's scene + ideal-path generator (the PLANNED tip trajectory,
shared across bodies) and realizes it per embodiment via embodiments.py. The
planned path is exactly toy_frame's point-robot demo; each body's stored demo
is its ACHIEVED tip-delta chunk. Invariant lives in the tip (task) frame and is
linear in every body's (tip-delta) action space.

Named mb_dataset (not dataset) to avoid a sys.modules collision with
toy_frame/dataset.py, which this imports as `tf`.

Target radius restricted to [1.0, 1.7] (vs toy_frame's [1.0, 2.0]) so all arms
can reach the goal and the embodiment divergence appears on the detour.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "toy_frame"))
import dataset as tf                    # noqa: E402  (toy_frame/dataset.py)

H = tf.H
ACT_SCALE = tf.ACT_SCALE


def make_scene(rng):
    """toy_frame scene but with target radius in [1.0, 1.7]."""
    while True:
        sc = tf.make_scene(rng)
        if 1.0 <= sc["radius"] <= 1.7:
            return sc


def planned_positions(scene, rng):
    """Ideal tip path (H+1, 2) in world frame from toy_frame's make_demo
    (which returns the tip-delta chunk that sums to the target)."""
    chunk = tf.make_demo(scene, rng) / ACT_SCALE
    return np.concatenate([[[0.0, 0.0]], np.cumsum(chunk, axis=0)], axis=0)


def body_demo(body, scene, rng):
    """One achieved tip-delta chunk (H, 2), normalized by ACT_SCALE."""
    P = planned_positions(scene, rng)
    A = body.realize(P)
    return np.diff(A, axis=0) * ACT_SCALE


def make_dataset(bodies, n_scenes, n_demos, rng):
    """scenes, obs (M,5), angles (M,), chunks dict body -> (M,N,H,2) world."""
    scenes, obs, angles = [], [], []
    chunks = {b: [] for b in bodies}
    for _ in range(n_scenes):
        sc = make_scene(rng)
        scenes.append(sc); obs.append(tf.scene_obs(sc)); angles.append(sc["angle"])
        for name, body in bodies.items():
            chunks[name].append(np.stack([body_demo(body, sc, rng)
                                          for _ in range(n_demos)]))
    return (scenes, np.array(obs), np.array(angles),
            {b: np.stack(v) for b, v in chunks.items()})


def canonical(chunks, angles):
    return tf.to_canonical(chunks, angles)


def success(scene, chunk_world_normalized):
    return tf.success(scene, chunk_world_normalized)
