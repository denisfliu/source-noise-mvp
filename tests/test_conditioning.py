"""Unit tests for arm B conditioning injection. Runs as plain python or pytest."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snmvp.conditioning import inject_invariant_state, load_invariant_stats  # noqa: E402

S, K = 32, 7
RNG = np.random.default_rng(0)


def test_inject_writes_trailing_dims_only():
    state = RNG.normal(size=(4, S))
    before = state.copy()
    inv = RNG.normal(2.0, 0.5, size=(4, K))
    out = inject_invariant_state(state, inv, [2.0] * K, [0.5] * K)
    assert out is state
    assert np.array_equal(state[:, : S - K], before[:, : S - K])
    assert np.allclose(state[:, S - K:], (inv - 2.0) / 0.5)


def test_znorm_scale():
    inv = RNG.normal(3.0, 2.0, size=(5000, K))
    state = np.zeros((5000, S))
    inject_invariant_state(state, inv, [3.0] * K, [2.0] * K)
    z = state[:, S - K:]
    assert abs(z.mean()) < 0.1 and abs(z.std() - 1.0) < 0.05


def test_stats_roundtrip():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"mean": [1.0] * K, "std": [2.0] * K}, f)
        path = f.name
    mean, std = load_invariant_stats(path)
    assert mean == [1.0] * K and std == [2.0] * K


def _torch_mirror():
    try:
        import torch
    except ImportError:
        print("torch not installed — skipped torch mirror test")
        return
    state = torch.randn(4, S, dtype=torch.float64)
    before = state.clone()
    inv = torch.randn(4, K, dtype=torch.float64)
    inject_invariant_state(state, inv, [0.0] * K, [1.0] * K)
    assert torch.equal(state[:, : S - K], before[:, : S - K])
    assert torch.allclose(state[:, S - K:], inv)
    print("torch mirror test passed")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"{fn.__name__} passed")
    _torch_mirror()
    print(f"\nall {len(fns)} numpy tests passed")
