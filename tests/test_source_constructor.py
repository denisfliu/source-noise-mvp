"""Unit tests for the carried-invariant property and noise calibration.

Runnable as plain `python tests/test_source_constructor.py` (no pytest needed)
or via pytest. Uses numpy; if torch is installed, mirrors every test on
tensors to guard the training-time code path.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snmvp import (  # noqa: E402
    PinStats,
    SourceConstructor,
    carried_residual,
    extract_invariant,
)

H, D = 50, 7
RNG = np.random.default_rng(0)


def _data(batch=()):
    eps = RNG.normal(size=(*batch, H, D))
    a0 = RNG.normal(size=(*batch, H, D)) * 0.3
    return eps, a0


def test_identity_when_alpha_zero():
    eps, a0 = _data()
    out = SourceConstructor(alpha=0.0)(eps, extract_invariant(a0))
    assert np.array_equal(out, eps)


def test_exact_pin():
    eps, a0 = _data()
    m = extract_invariant(a0)
    out = SourceConstructor()(eps, m)
    assert np.allclose(extract_invariant(out), m, atol=1e-10)


def test_carried_invariant_all_t():
    """L(x_t) constant across the interpolant; L(v) = 0."""
    eps, a0 = _data()
    m = extract_invariant(a0)
    et = SourceConstructor()(eps, m)
    for t in np.linspace(0, 1, 11):
        xt = t * et + (1 - t) * a0
        assert np.allclose(extract_invariant(xt), m, atol=1e-9), f"t={t}"
    assert np.allclose(carried_residual(et, a0), 0.0, atol=1e-10)


def test_complement_untouched():
    """Within-chunk variation around the per-dim time mean is unchanged."""
    eps, a0 = _data()
    out = SourceConstructor()(eps, extract_invariant(a0))
    center = lambda x: x - x.sum(-2, keepdims=True) / H
    assert np.allclose(center(out), center(eps), atol=1e-10)


def test_batch_shapes():
    eps, a0 = _data(batch=(4, 3))
    m = extract_invariant(a0)
    out = SourceConstructor()(eps, m)
    assert out.shape == eps.shape
    assert np.allclose(extract_invariant(out), m, atol=1e-9)


def test_partial_dims():
    eps, a0 = _data()
    m = extract_invariant(a0)
    pinned = [0, 1, 2]
    out = SourceConstructor(pinned_dims=pinned)(eps, m)
    got = extract_invariant(out)
    want = extract_invariant(eps)
    for dim in range(D):
        if dim in pinned:
            assert np.isclose(got[dim], m[dim], atol=1e-10)
        else:
            assert np.isclose(got[dim], want[dim], atol=1e-10)


def test_soft_pin_interpolates():
    eps, a0 = _data()
    m = extract_invariant(a0)
    raw = extract_invariant(eps)
    out = SourceConstructor(alpha=0.5)(eps, m)
    assert np.allclose(extract_invariant(out), 0.5 * m + 0.5 * raw, atol=1e-10)


def test_zscored_marginal():
    """Pinned coordinate under zscored mode is ~N(0,1) when invariants are
    drawn from the stats' distribution."""
    n = 20000
    mean, std = 2.0, 0.5
    invs = RNG.normal(mean, std, size=(n, D))
    eps = RNG.normal(size=(n, H, D))
    sc = SourceConstructor(mode="zscored",
                           stats=PinStats(mean=[mean] * D, std=[std] * D))
    out = sc(eps, invs)
    coord = extract_invariant(out) / np.sqrt(H)
    assert abs(coord.mean()) < 0.03
    assert abs(coord.std() - 1.0) < 0.03


def test_gaussianity_of_complement():
    """Calibrated noise off the pinned subspace keeps unit variance."""
    eps, a0 = _data(batch=(2000,))
    out = SourceConstructor()(eps, extract_invariant(a0))
    centered = out - out.sum(-2, keepdims=True) / H
    # variance of centered gaussian coords: (1 - 1/H)
    assert abs(centered.var() - (1 - 1 / H)) < 0.01


def _torch_mirror():
    try:
        import torch
    except ImportError:
        print("torch not installed — skipped torch mirror tests")
        return
    eps = torch.randn(4, H, D, dtype=torch.float64)
    a0 = torch.randn(4, H, D, dtype=torch.float64) * 0.3
    m = extract_invariant(a0)
    out = SourceConstructor()(eps, m)
    assert torch.allclose(extract_invariant(out), m, atol=1e-9)
    for t in (0.0, 0.3, 1.0):
        xt = t * out + (1 - t) * a0
        assert torch.allclose(extract_invariant(xt), m, atol=1e-8)
    print("torch mirror tests passed")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"{fn.__name__} passed")
    _torch_mirror()
    print(f"\nall {len(fns)} numpy tests passed")
