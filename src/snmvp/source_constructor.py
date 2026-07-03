"""Source-noise calibration for flow-matching action heads.

Core object: given Gaussian noise eps of shape (..., H, D) (H = chunk length,
D = action dim) and an invariant m of shape (..., D) in raw summed-delta units,
produce calibrated noise eps_tilde such that

    L_d(eps_tilde) = sum_t eps_tilde[..., t, d] = m_d      (for pinned dims d)

while leaving the orthogonal complement (all within-chunk variation around the
per-dim time mean) untouched.

Why this pin: L is linear, so for any interpolant x_t = t*eps_tilde + (1-t)*a0
with m = L(a0),

    L(x_t) = t*L(eps_tilde) + (1-t)*L(a0) = L(a0)   for all t,

and the flow target v = eps_tilde - a0 satisfies L(v) = 0. The invariant is
carried into the regression target at every noise level; deviation from it is
directly penalized by the flow loss.

Works with numpy arrays and torch tensors (only uses shape, sum, arithmetic,
and ellipsis indexing, which both support).

Modes
-----
exact (default): writes m in raw units. Carried-invariant property holds
    exactly. The pinned coordinate's marginal distribution is that of L(a0)
    (dataset-dependent), not N(0,1); check `pin_coordinate_std` against 1.0
    and consider whether the drift from Gaussian stats matters at your scale.
zscored: writes a z-normalized coordinate so the pinned dim marginally matches
    N(0,1). Carried property becomes approximate (exact only if stats are
    (0,1) already). Use for ablation.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class PinStats:
    """Dataset statistics of the invariant L(a0), per action dim."""

    mean: Sequence[float]
    std: Sequence[float]


@dataclass
class SourceConstructor:
    """Calibrates source noise to carry a linear chunk invariant.

    Parameters
    ----------
    pinned_dims: indices of action dims to pin (None = all dims).
    alpha: pin strength in [0, 1]. 1.0 = hard overwrite (exact carry),
        0.0 = identity (baseline arm A/B behavior).
    mode: "exact" or "zscored" (see module docstring).
    stats: required for mode="zscored".
    """

    pinned_dims: Optional[Sequence[int]] = None
    alpha: float = 1.0
    mode: str = "exact"
    stats: Optional[PinStats] = None
    _warned: bool = field(default=False, repr=False)

    def __call__(self, eps, invariant):
        """eps: (..., H, D) Gaussian noise. invariant: (..., D) raw L units.

        Returns eps_tilde of the same shape and type as eps.
        """
        if self.alpha == 0.0:
            return eps
        H = eps.shape[-2]
        D = eps.shape[-1]

        current = eps.sum(-2)  # (..., D) current L(eps)

        if self.mode == "exact":
            target = invariant
        elif self.mode == "zscored":
            if self.stats is None:
                raise ValueError("mode='zscored' requires stats")
            # write the z-scored coordinate scaled back so that the pinned
            # coordinate u^T eps_tilde = (m - mean)/std, i.e. marginally ~N(0,1)
            # under the dataset distribution of m. Raw-unit target:
            #   L(eps_tilde) = sqrt(H) * (m - mean) / std
            sqrt_h = H ** 0.5
            mean = _like(invariant, self.stats.mean)
            std = _like(invariant, self.stats.std)
            target = sqrt_h * (invariant - mean) / std
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        correction = (target - current) / H  # (..., D) per-step shift

        if self.pinned_dims is not None:
            mask = _dim_mask(invariant, D, self.pinned_dims)
            correction = correction * mask

        return eps + self.alpha * correction[..., None, :]


def extract_invariant(chunk):
    """L(a0): summed per-step deltas over the chunk. chunk: (..., H, D)."""
    return chunk.sum(-2)


def carried_residual(eps_tilde, a0):
    """L(v) for the flow target v = eps_tilde - a0. Zero iff exactly carried."""
    return extract_invariant(eps_tilde) - extract_invariant(a0)


def pin_coordinate_std(invariants, H):
    """Std of the pinned noise coordinate u^T eps_tilde = L/sqrt(H) under the
    dataset distribution of invariants (..., D). Compare against 1.0 (the std
    the denoiser expects of a Gaussian source coordinate)."""
    coord = invariants / (H ** 0.5)
    mean = coord.sum(0) / coord.shape[0]
    var = ((coord - mean) ** 2).sum(0) / coord.shape[0]
    return var ** 0.5


def _dim_mask(ref, D, dims):
    """(D,) 0/1 mask, same array type as ref."""
    vals = [1.0 if i in set(dims) else 0.0 for i in range(D)]
    return _like(ref, vals)


def _like(ref, values):
    """Build an array of `values` matching ref's library and dtype/device."""
    if hasattr(ref, "new_tensor"):  # torch
        return ref.new_tensor(values)
    import numpy as np

    return np.asarray(values, dtype=ref.dtype)
