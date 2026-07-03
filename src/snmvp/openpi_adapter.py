"""Inference-side adapter for openpi.

openpi's `Policy.infer(obs, noise=...)` already threads caller-supplied noise
into `sample_actions` (batched, device-placed) — no openpi patch needed at
inference. This module builds the calibrated noise to pass in.

Usage (arm C rollout):

    from snmvp.openpi_adapter import make_calibrated_noise

    noise = make_calibrated_noise(
        invariant=commanded_L,        # NORMALIZED action units, len <= action_dim
        action_horizon=50,            # model config
        action_dim=32,                # model config (padded motor dim)
        rng=np.random.default_rng(seed),
    )
    result = policy.infer(example, noise=noise)

Critical: `invariant` must be expressed in the model's normalized action
space (post q01/q99), matching what the training-side pin saw. Normalize a
physical command with the run's norm_stats.json before calling this, and pad
handling is automatic (dims beyond len(invariant) are left unpinned).
"""

import numpy as np

from .source_constructor import SourceConstructor


def make_calibrated_noise(invariant, action_horizon, action_dim, rng,
                          alpha=1.0):
    """Returns (action_horizon, action_dim) float32 noise with the invariant
    pinned into the leading len(invariant) action dims."""
    invariant = np.asarray(invariant, dtype=np.float32)
    if invariant.ndim != 1 or len(invariant) > action_dim:
        raise ValueError(f"invariant must be 1-D with <= {action_dim} entries")
    eps = rng.normal(size=(action_horizon, action_dim)).astype(np.float32)
    full = np.zeros(action_dim, dtype=np.float32)
    full[: len(invariant)] = invariant
    sc = SourceConstructor(pinned_dims=list(range(len(invariant))),
                           alpha=alpha)
    return sc(eps, full)


def normalize_invariant(physical_invariant, norm_stats, dims=None):
    """Convert a physical chunk-displacement command into normalized action
    units using openpi norm_stats (q01/q99 scaling: a_norm = 2*(a-q01)/(q99-q01)-1
    applied per step; a sum of H per-step deltas scales as
    L_norm = 2*(L_phys - H*q01)/(q99-q01) - H ... which depends on H).

    NOTE: the exact affine map depends on the normalization openpi applied
    (q01/q99 vs mean/std) and acts per *step*, so the chunk sum transforms
    with an H-dependent offset. The robust route — used by the Phase 1 oracle
    — is to compute invariants of ALREADY-NORMALIZED demo chunks and command
    in that space directly, skipping physical units. This helper is a
    placeholder; fill in against the actual norm_stats.json schema of your
    training run before commanding physical-unit invariants.
    """
    raise NotImplementedError(
        "command invariants in normalized space (extract from normalized "
        "chunks); see docstring before implementing physical-unit conversion"
    )
