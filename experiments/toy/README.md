# Toy validation: source-noise pinning vs conditioning branch

2D point-robot reach task, H=20 delta-action chunks, bimodal path style
(left/right bend), invariant = chunk displacement L(a) = sum of deltas.
Tiny flow-matching MLPs (3x128 relu, ~10k Adam steps, CPU, ~25 s/arm via
`autograd`). Run: `python toy_flow.py --arm C`; aggregate: `--report`.

## Results (seeds: A,D n=1; B,C n=3)

| arm | invariant path | endpoint err (in / held-out) | probe err-to-command | diversity (frac left / spread) |
|---|---|---|---|---|
| A | none | 0.038 / 0.036 | — | 0.42 / 0.37 |
| B | conditioning input | 0.057 / 0.064 | **0.702** | 0.51 / 0.48 |
| C | source noise | **0.018 / 0.017** | **0.027** | 0.49 / 0.43 |
| D | both | 0.011 / 0.011 | 0.021 | 0.45 / 0.45 |

(Targets sit at radius 1–2, so err 0.02 is ~1% of trajectory scale.)

## Reading

1. **The mechanism works end-to-end.** An optimizer trained with pinned
   noise produces a policy whose rollouts satisfy the commanded invariant to
   ~1% — 3–4x tighter than the conditioning branch, and tighter than the
   unconditioned baseline (the pin adds precision, it doesn't just match it).
2. **The sharp difference is adherence under contradiction, not follow
   rate.** Commanded an invariant contradicting the observation, both B and
   C move toward the command (follow rate 1.0 for both — in this toy, obs
   and m are redundant and low-dim, so B learned to use m). But B executes
   the contradictory command sloppily (lands 0.70 away); C executes it
   near-exactly (0.03). Target-carried control is *optimized*; branch-carried
   control is approximate. This is the paper's central claim in miniature.
   => Plan implication: the Phase 1 gate metric should be
   **error-to-command under the wrong-invariant probe**, with follow rate as
   a secondary binary check.
3. **Diversity survives the pin.** Both path modes (left/right bend) appear
   at roughly equal rates in C/D at fixed invariant — the unpinned noise
   dimensions still drive style sampling. No mode collapse from pinning.
4. **What the toy cannot tell us:** whether a 3B VLA also learns to read the
   pinned channel when the observation is high-dim pixels and far more
   informative than in this toy (where B's branch had an easy time — real
   vision-dominant models may ignore a branch input more, making C's
   advantage larger, or attend to it fine, making it smaller). That is
   exactly what Phase 1 on LIBERO measures.

## Notes

- Actions must be normalized to O(1) (`ACT_SCALE`) — with raw small deltas
  the flow model underfits badly (endpoint err ~0.5). The real pipeline's
  q01/q99 normalization plays this role; the pin must be computed in the
  same normalized space (already noted in docs/openpi_integration.md).
- `autograd` used instead of torch (sandbox constraint); the model is small
  enough that this changes nothing. `pip install autograd` to rerun.

## Addendum (2026-07-05): CFG-style pin dropout kills the channel

Tested before committing GPU time to a pin-dropout arm (P) and a variant with
a CFG-style "pin present" flag input (Q). Both train with the pin on 80% of
samples. Results (3 seeds):

| arm | probe err-to-command | unpinned endpoint err | pinned endpoint err |
|---|---|---|---|
| C (always pinned) | **0.027** | — | 0.018 |
| P (dropout 0.2) | 2.43 (90x worse) | 0.115 (3x worse than A) | 0.024 |
| Q (dropout + flag) | 2.26 | 0.116 | 0.024 |

Follow rate 0.0 at every inference alpha for both P and Q; rollouts land on
the OBS target (err_to_obs ~0.11-0.17) — the model corrects the contradictory
pin away. Mechanism: with dropout, the obs->endpoint decode must be learned
for the unpinned samples, and once learned it is correct on ALL samples, so
no gradient pressure maintains the pin-read. The always-on pin of arm C wins
because it is the *cheaper* feature (linear read off x_t), not because obs is
insufficient. The presence flag does not change the economics.

=> Design consequence: the pin must be always-on in training. There is no
free "unpinned/plain policy" mode from one checkpoint; a scene-consistent
invariant must be supplied at inference (oracle now, learned prior in
Phase 2). Inference-time alpha<1 on an always-pinned model is a different
regime and remains a valid ablation.
