"""Evaluation probes for Phase 1.

Wrong-invariant probe: command an invariant that contradicts the scene and
measure whether the rollout follows the noise (commanded) or the vision
(scene-consistent). This is the primary gate metric — it is the only readout
that cleanly separates "reads the pinned channel" from "conditioning happened
to agree with what vision suggests anyway".
"""

import numpy as np


def sample_wrong_invariant(true_invariant, stats, rng, min_angle_deg=90.0):
    """Sample a commanded invariant whose translation component points at
    least `min_angle_deg` away from the scene-consistent one, with magnitude
    resampled from dataset stats (so it is wrong in direction, plausible in
    scale).

    true_invariant: (D,) raw L units; dims 0..2 assumed translation.
    stats: dict from invariants.compute_dataset_stats.
    """
    inv = np.array(true_invariant, dtype=float)
    t = inv[:3]
    norm = np.linalg.norm(t)
    if norm < 1e-8:
        direction = _random_unit(rng)
    else:
        direction = _rotated_away(t / norm, min_angle_deg, rng)
    mag = abs(rng.normal(loc=np.linalg.norm(stats["mean"][:3]),
                         scale=np.linalg.norm(stats["std"][:3])))
    out = inv.copy()
    out[:3] = direction * max(mag, norm)  # at least as salient as the true one
    return out


def adherence_error(realized_chunk, commanded_invariant):
    """|L(realized) - commanded| per dim. realized_chunk: (H, D)."""
    realized = np.asarray(realized_chunk).sum(0)
    return np.abs(realized - np.asarray(commanded_invariant))


def follow_rate(errors_cmd, errors_scene):
    """Fraction of episodes where the rollout is closer (translation L2) to
    the commanded invariant than to the scene-consistent one.

    errors_cmd / errors_scene: (N, D) adherence errors against each reference.
    """
    cmd = np.linalg.norm(np.asarray(errors_cmd)[:, :3], axis=1)
    scene = np.linalg.norm(np.asarray(errors_scene)[:, :3], axis=1)
    return float((cmd < scene).mean())


def residual_diversity(chunks):
    """Trajectory spread at fixed invariant/observation across noise draws.

    chunks: (N, H, D) rollouts under identical conditions. Returns mean
    per-step std after removing each chunk's time-mean (i.e. spread in the
    unpinned subspace only).
    """
    arr = np.asarray(chunks)
    centered = arr - arr.mean(1, keepdims=True)
    return float(centered.std(0).mean())


def _random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _rotated_away(unit, min_angle_deg, rng, max_tries=100):
    cos_max = np.cos(np.deg2rad(min_angle_deg))
    for _ in range(max_tries):
        cand = _random_unit(rng)
        if np.dot(cand, unit) <= cos_max:
            return cand
    return -unit
