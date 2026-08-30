# The key math of the pin stack (2026-08-29 summary)

Everything the system does reduces to seven small identities. Notation: a = action chunk,
vec'd to R^1600 (50 steps x 32 padded dims) in NORMALIZED units (q01/q99 stats, shared
across sim and real); U in R^{1600 x 16}, orthonormal columns; c = U^T a in R^16.

## 1. The carry (why a command survives denoising)
pi0's interpolant is x_t = t z + (1-t) a (z = source noise). For any LINEAR functional:
U^T x_t = t (U^T z) + (1-t)(U^T a). Training constructs z so U^T z = c := U^T a, hence
U^T x_t = c for ALL t, and the flow target v = z - a satisfies U^T v = 0 — the command sits
in the regression target itself, not in a conditioning branch. Serve-time: sample g~N(0,I),
z = g - U U^T g + U c. Deterministic Euler only (stochastic sampling would erode the carry).

## 2. The basis (what the 16 coordinates mean)
mh16 = QR-orthonormalization of the prefix-sum functionals sum_{t<=H} a_t^{(ch)} for
H in {6,12,25,50} x ch in {x,y,z,yaw}. Empirically each dim k decodes to "net ch-motion
over the first ~H80(k) steps" with H80 = {5,11,23,45} (verified 2026-08-29). So c is a
16-word sentence: how much x/y/z/turn, at four timescales.

## 3. Decode / interpretability (pin -> movement)
U c is the MINIMUM-NORM chunk consistent with the sentence; its integrated path
p_k = anchor + cumsum(denorm(U c)[:, :3]) is "what the pin says". The generated chunk a_hat
satisfies U^T a_hat ~= c (measured: |U^T a_hat - c| = 0.070 cstd real obs, 0.068 synth) —
the declaration is a measured contract, not a story.

## 4. The trust dial (sigma)
Training: z's pin component = c + sigma * eps, sigma ~ U[0, 1.5] * cstd (isotropic, pin
only), sigma given to the model. Serve: sigma = 0 => exact obedience; larger sigma =>
licensed deviation. sigma_serve for the head's own commands comes from a monotone map
sigma* -> sigma fit per checkpoint on demo-frame error quantiles (maps are
CHECKPOINT-SPECIFIC; corr(sigma*, err) = 0.82-0.94 in sim, ~0 on real frames until
recalibrated on real data).

## 5. Command sources (everything speaks c)
- Head: GMM/MDN over c (FiLM diet inputs), NLL at lambda=0.3, argmax + pi-hysteresis serve.
- Sketch: polyline resampled at 0.025 m/step (demo speed) -> per-step deltas -> normalize
  -> window projection c = U^T a_window at each replan; forward-monotonic nearest-point
  progress (capped ~ replan stride), handback requires spatial ARRIVAL at the end.
- Carrot (lateral servo): window built from a rejoin curve, offset decays (1 - k/L)_+ over
  L steps => per-step correction |offset|/L stays in the trained delta regime.
- Rotation verb: c' = U^T (R_z(theta) applied rowwise to U c) ~= M(theta) c; achieved
  rotation/commanded = 0.76 on real frames.
- Cross-domain: any of the above authored in sim executes on real observations (execution
  gap ~0), because the channel is upstream of perception.

## 6. Cross-supervision (xswap — why it worked where dsplit failed)
With matched state pairs (argmin |dp| + 0.3|dyaw|), swap real frames' action chunks for the
matched sim chunk with prob p=0.5 INSIDE mixed training. The in-graph pin c and head target
follow the swapped chunk, so the model learns p(a | obs, c) with BOTH styles commandable —
no alignment loss, no sequencing. dsplit's sequential phases moved the VLM features out
from under the frozen head (readout R^2 -> -0.19): the head must co-train with whatever
moves its features.

## 7. Judging (the claims' denominators)
Strict success = directional aperture transit + ROUTE-CLEAN (zero wrong-direction aperture
crossings anywhere; demos are unanimously clean under this) + min clearance 0.18 m to the
gate cloud + human video. Sim-real distance = the pin-gap triplet at matched states:
execution |U^T a_hat - c| (~0), prediction |c_head(real) - c_head(synth)| vs the in-domain
floor (~at floor), behavior |c_oracle_real - c_oracle_synth| (0.62, endgame-heavy — the
actionable term).

## Current flagship + replication status
xswap (S3 recipe): sim 40/40 route-clean + 40/40 clearance-clean; real-frame own-head
right-gate crossings 15/39 pre-gate anchors (vs gmsig3's 8), chunk speed 0.79 m. Seed-7
replication running 2026-08-29 overnight (gate_pin_joint_xswaps7). gmsig3/gmsig3s7 pooled
80/80 remain the seed-replicated baseline pair.
