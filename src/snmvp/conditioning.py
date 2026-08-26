"""Arm B: branch-carried invariant conditioning (MVP plan Phase 1).

Injects the same chunk invariant that arm C pins into the source noise, but as
a conditioning INPUT: the z-normalized invariant is written into trailing
(padding) dims of the proprio state vector. For LIBERO, the real state is
8-dim inside pi0's 32-dim padded state, so the trailing dims are constant zero
and unused — writing there gives the action expert a proprio-style
conditioning token through the existing `state_proj` weights with an exactly
unchanged parameter count (the design requirement for a fair B-vs-C
comparison).

The invariant is z-normalized against dataset statistics (compute them with
scripts/compute_invariant_stats.py) so the written coordinates are O(1) like
the rest of the normalized state — mirroring the statistics-matching argument
for the noise pin: the branch gets the same information at a sane scale, not a
detectably out-of-distribution one.

Training-side use is via patches/openpi_arm_b_conditioning.patch (env
SNMVP_COND_STATS=/path/to/invariant_stats.json enables it). Eval-side, call
`inject_invariant_state` on the observation's state tensor before
`sample_actions` / `policy.infer`, with the SAME stats file.
"""

import json

from .source_constructor import _like


def load_invariant_stats(path):
    """Reads {"mean": [k], "std": [k]} JSON written by compute_invariant_stats."""
    with open(path) as f:
        d = json.load(f)
    return d["mean"], d["std"]


def inject_invariant_state(state, invariant, mean, std):
    """Write the z-normalized invariant into the TRAILING dims of state, in place.

    state: (..., S) torch tensor or numpy array (normalized proprio, S=32 for pi0)
    invariant: (..., k) raw invariant L(a0) in normalized-action units (k=7 LIBERO)
    mean/std: length-k dataset statistics of the invariant

    Returns state (mutated in place; also returned for convenience). The
    trailing k dims must be padding for the robot at hand — for LIBERO
    (8 real state dims) any k <= 24 is safe.
    """
    k = invariant.shape[-1]
    z = (invariant - _like(invariant, mean)) / _like(invariant, std)
    state[..., state.shape[-1] - k:] = z
    return state
