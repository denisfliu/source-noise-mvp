# Learned orthonormal basis for the pass-through pin — NEGATIVE (direction retired)

Result (2026-07-22): generalizing the fixed Fourier factoring to a learned
orthonormal basis gives **no gain**. Across every properly-trained,
policy-bottlenecked comparison the learned basis (L) never beats Fourier (F).

## Theory (why pass-through is basis-general)

The pin is not specific to Fourier. Pass-through holds for any orthonormal
transform T of the action chunk: T preserves the linear flow path and the
isotropic Gaussian source, so clamping a T-subspace gives zero velocity on those
coordinates and the output carries the command exactly. Fourier is one such T.
The earlier OAT pin failed because a nonlinear bottleneck breaks pass-through
(the latent is neither Gaussian nor linearly tied to the output); the implied fix
is to constrain the learned transform to be orthonormal — which is what L does.

## Controlled comparison

Only the basis U differs across arms (same general-subspace projection pin
`c = Uᵀa`, same prior, same flow executor):
- **A** scratch (no pin)
- **F** Fourier basis, top-k directions by the coherence objective
  `S(e) = (eᵀΣ_b e)/(eᵀΣ_w e)` (high between-scene, low within-scene variance)
- **L** learned: unrestricted top-k generalized eigenvectors of (Σ_b, Σ_w),
  orthonormalized (the argmax of S over all orthonormal subspaces)
- **R** random orthonormal (control)

## Results (held-out success, pooled over 3 seeds)

Smooth multi-obstacle (`learned_basis_toy.py`, well-calibrated, pinned near ceiling):

| n_obst | A | F | L | R | ceil |
|---|---|---|---|---|---|
| 1 | 0.50 | 0.95 | 0.97 | 0.93 | 1.00 |
| 2 | 0.28 | 0.88 | 0.88 | 0.90 | 1.00 |
| 3 | 0.20 | 0.62 | 0.60 | 0.47 | 0.89 |

`basis_lab.py` (C-channel generalization):

| task | A | F | L | R | ceil |
|---|---|---|---|---|---|
| waypoint2d K=6 | 0.96 | 1.00 | 0.98 | 0.99 | 1.00 |
| waypoint2d K=4 | 0.96 | 0.99 | 0.98 | 0.98 | 1.00 |
| reach3d K=8 | 0.44 | 0.59 | 0.50 | 0.39 | 0.96 |

## Conclusion

- **L never beats F.** Smooth multi-obstacle: L ≈ F. reach3d (bottlenecked): F > L.
  waypoint2d: tie at ceiling (uninformative — see below).
- The pin helps only when structure is **smooth** and the policy is
  **bottlenecked**; for smooth structure the Fourier modes already span the
  coherent subspace (Fourier ≈ eigenbasis of smooth-trajectory covariance), so a
  learned basis adds nothing.
- The regime that would favor a learned basis (bottlenecked AND non-Fourier AND
  low-rank) did not naturally occur: non-Fourier (localized) structure was either
  easy for the MLP to produce directly (not bottlenecked — waypoint2d: scratch
  reaches 0.96 because obs encodes the waypoints) or high-rank (no low-k basis
  helps). The L > F seen only at a 600-iteration smoke was an undertraining
  artifact (the pin gives an early head start that scratch erases by convergence).
- **reach3d is a positive for the OTHER track**: the Fourier factoring extends to
  a third action channel (z) and helps (F 0.59 > A 0.44 > R 0.39), prior-limited
  below ceiling exactly as in the 2-D robosuite obstacle/slalom results. For 6-DOF
  / drones: keep Fourier, add channels.
- Reconciles the earlier "Fourier degrades 5→3→3": that was the adaptive
  energy-floor gating dropping pins, not the basis — fixed-k Fourier does not
  collapse (n_obst=3 F = 0.62 ≫ scratch 0.20).

## Reproduce

Box `~/code/source-noise-mvp/experiments/toy_embodiment/` (CPU/autograd):

    ../../.venv/bin/python learned_basis_toy.py            # smooth multi-obstacle
    SNMVP_TASK=waypoint2d SNMVP_K=6 SNMVP_ITERS=4000 ../../.venv/bin/python basis_lab.py
    SNMVP_TASK=reach3d   SNMVP_K=8 SNMVP_ITERS=6000 ../../.venv/bin/python basis_lab.py

Results: `learned_basis_result.json`, `basis_lab_<task>_K<k>.json`.
