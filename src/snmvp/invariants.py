"""Invariant extraction and dataset statistics.

The Phase 1 invariant is the chunk-level summed action delta L(a) = sum_t a_t,
computed on *normalized* actions (post openpi q01/q99 normalization), because
that is the space the flow head operates in and the space where L is the pin.

Rotation caveat: for delta-orientation action dims, sum-of-deltas is a linear
functional of the chunk (well-defined pin) but only approximates the net
orientation change for small per-step deltas. Interpret adherence metrics for
rotation dims accordingly; translation dims are exact.
"""

import json

import numpy as np

from .source_constructor import extract_invariant, pin_coordinate_std


def compute_dataset_stats(chunks):
    """chunks: iterable of (H, D) normalized action arrays (or one (N, H, D)).

    Returns dict with per-dim mean/std of L(a), the pinned-coordinate std
    (compare vs 1.0), and H.
    """
    arr = np.asarray(list(chunks) if not isinstance(chunks, np.ndarray) else chunks)
    if arr.ndim != 3:
        raise ValueError(f"expected (N, H, D), got {arr.shape}")
    n, h, d = arr.shape
    inv = extract_invariant(arr)  # (N, D)
    return {
        "H": h,
        "D": d,
        "n_chunks": n,
        "mean": inv.mean(0).tolist(),
        "std": inv.std(0).tolist(),
        "pin_coordinate_std": pin_coordinate_std(inv, h).tolist(),
    }


def save_stats(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path):
    with open(path) as f:
        return json.load(f)
