# Phase 1 results — pooled (sweep complete 2026-07-07 ~02:05 UTC)

7-run sweep done: A x3, C x3, B x1, all pi0_libero, 15k steps, cosine decay
rescaled to 15k, batch 32, single GPU/run, `SNMVP_PINNED_DIMS=7` for arm C.
Phase 0 exit gate PASSED first (30k baseline 94.6% vs ~96% ref, 50-trial).
Success = LIBERO-Spatial, 10 trials/task; protocol noise floor ~+-5-6 pts
(same weights re-evaluated: 85/92, 94/89, 95/87 across the A twin checkpoints).

## Headline: the coupling spectrum

Same invariant information, same learned prior p(invariant|obs) as command
source, same data — the ONLY variable across B/C is where the signal enters
the flow head (conditioning state token vs source noise).

| arm | channel | adherence: oracle / contradictory / negated (err-to-command, dataset scale ~123) | follow rate | success canonical / held-out (pooled) |
|---|---|---|---|---|
| A | none | — | — | **90.3% / 89.7%** |
| B | conditioning token | 6.15 / **82.7** / **138.8** | 0.625 | **76% / 73%** (1 seed) |
| C | source-noise pin | 3.07 / **7.82** / 9.99 | 1.000 | **49.7% / 53.3%** |

Per-seed (C, 3 seeds): channel oracle 2.8/3.3/3.1, contradictory 7.1/8.2/8.1
(follow 1.0 every checkpoint every seed); success canonical 46/51/52,
held-out 51/52/57. Seed-stable to a few percent throughout training.

## What the numbers say

1. **The source-noise channel binds ~11x tighter than conditioning under
   contradiction** (C 7.8 vs B 82.7) and unboundedly tighter under negation
   (C 10.0 vs B 138.8). B's contradictory/negated errors EXCEED its own
   plain-noise control (77.2) — i.e. conditioning doesn't just weakly follow
   a contradictory command, it actively follows the SCENE instead, moving
   away from the command. Follow rate 1.00 (C) vs 0.625 (B).

2. **Both non-A channels capture the model** (stock eval, command-less: B
   0-4%, C 0% across training). "Conditioning is ignorable / the branch gets
   bypassed" is false at 3B scale — the difference is coupling STRENGTH, not
   whether coupling happens.

3. **Coupling strength sets an obedience<->success operating point.**
   Loose coupling (B) recovers most of A's success (76 vs 90) from a noisy
   (~25% err) prior because vision overrides bad commands — but is
   unsteerable against the scene. Tight coupling (C) is fully steerable but
   pays a ~40-pt success tax executing the prior's errors faithfully. A has
   the success and no control channel. This frontier is the Phase 1 result.

4. **No placement generalization gap for any arm** (A 90.3 vs 89.7 held-out;
   C 49.7 vs 53.3; B 76 vs 73). LIBERO-Spatial placement variation does not
   discriminate — the plan's original H1 framing (C beats B/A on held-out
   SUCCESS) is empirically void at this scale; the mechanism claim lives in
   adherence/steering, not success.

## Caveats / honest reading

- B is n=1 seed (a control, per the revised design); C/A are n=3.
- C's success ceiling is set by the prior's ~25% error, NOT the channel:
  oracle-command adherence is 3.07 (2.4% of scale), so a better command
  source raises C's success directly. The ~40-pt tax is a prior-quality
  statement, not a channel limit. (prior-v2 levers: temporal context,
  bigger net, confidence-gated alpha, per-replan residual.)
- Success uses the learned prior for BOTH B and C (symmetric); the stock
  0% evals are the command-less artifact and are not the arms' results.

## Provenance
Per-run manifests + all JSONs in this dir; channel curves in
phase1_C_s*_step*_probe.json; success in evals/. Oracle post-mortem
(why geometric oracles were replaced by the prior) in
oracle_iterations_summary.md.
