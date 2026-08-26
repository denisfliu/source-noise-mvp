# toy_cmdhead — command-head architecture under branch-state ambiguity (2026-08-19, local CPU)

Continuation of the generative-command-head line after box access loss (RESEARCH_LOG
2026-08-13; status_latest.md). Isolates the HEAD axis (MSE vs concat-CFM vs FiLM-CFM vs
GMM/MDN) from the feature axis, on a synthetic task replicating the measured pathologies:
start states identical across tasks with language the only disambiguator, phase
observability degraded near the tail, nonzero normalized stop signature. 5 training seeds
per head; Bayes-optimal stop posterior computed from the generative model as the
calibration reference. See `toy_cmdhead.py` header for the full design.

## Results (5 seeds, mean [min..max]; results/toy_cmdhead.json)

| head       | startL-mode | tail validity (σ) | tail calib MAE | mean→mode @ambiguous (σ) |
|------------|-------------|-------------------|----------------|--------------------------|
| mse        | 1.00        | 9.23              | 0.31           | 10.67                    |
| cfm        | 1.00        | 1.89              | 0.14           | 8.92                     |
| cfm_film   | 1.00        | 1.60              | 0.13           | 8.78                     |
| gmm        | 1.00        | 1.38              | 0.13           | 8.66                     |
| gmm_argmax | 1.00        | **0.31**          | 0.29           | **0.29**                 |

## Findings

1. **The box's start coin-flip does NOT reproduce with clean conditioning**: every head,
   every seed, uses language perfectly at the ambiguous start (left-mode fraction 1.00;
   box CFM ranged 0.0–0.6 and flipped on seed). Conditioning neglect is not intrinsic to
   concat-CFM at matched capacity — the box's calibration lottery lives in the FEATURE/
   COUPLING side (language diluted 15-vs-780 tokens in the pool, scene–prompt confounds),
   not in the head class. Caveat: toy ctx is 11-d and clean; the low-t gradient-
   concentration mechanism may only bite at scale with weak, high-dimensional ctx.
2. **Mode-averaging replicated and quantified**: MSE mean at Bayes-ambiguous tail rows is
   ~10σ from any valid mode with |fwd| ≈ 2.0 — below all mode magnitudes, the exact mh16
   shrinkage signature (2.48 below 3.0–6.8). All generative heads emit valid draws.
3. **GMM ≈ CFM on distributional metrics** (validity, calibration-vs-Bayes at the ~0.15
   8-sample noise floor) — the mixture head loses nothing.
4. **The GMM's differentiators are serve-side**: (a) explicit π(o) — the posterior is an
   inspectable, thresholdable quantity instead of an emergent property of a velocity
   field (the box's start-draw histogram instrument becomes a direct readout); (b)
   **argmax-mode serve**: 0.3σ validity, deterministic, zero sampling jitter — the box
   currently tames CFM jitter with k=8-mean + EMA, but the toy shows an 8-sample MEAN at
   an ambiguous state is ~8.7σ invalid (mode averaging reintroduced exactly where it
   matters); argmax-GMM commits to a mode instead of averaging, at the cost of hard
   switching (calib MAE 0.29 — intentionally overconfident; hysteresis/latching on π is
   the natural smoother).

## Implication for the box line

The GMM head is worth a box arm not as a validity fix (CFM already has that) but as the
**calibration-observability + jitter-free-serve** arm: same information diet as genfilm
(state, language pool, image pool), NLL loss, argmax serve with π-hysteresis. Predicted
box outcome given finding 1: it will inherit the same feature-side calibration problem as
CFM unless paired with the FiLM-style explicit channels — π makes that failure *visible*
(π(left)≈0.5 at start) rather than needing rollouts to detect.
