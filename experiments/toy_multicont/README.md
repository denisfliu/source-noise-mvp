# toy_multicont — B1: multi-continuation de-risk gate (2026-08-04)

Can a pinned flow executor learn MULTI-CONTINUATION data — the same observation
carrying forward, time-reversed (`b_t = -a_{H-1-t}`), and hover (zero-chunk)
continuations, each row pinned with its own chunk's invariant exactly as
standard training does — and select among them by the commanded invariant at
inference, without mode collapse and without degrading the forward task?

Code: `multicont.py` (imports `toy_frame/dataset.py`, `toy_frame/pin.py`,
`toy_embodiment/flow_embod.py` unchanged — loss + pin construction identical
to the standard pipeline). Pins: all-LINEAR set — mod2pi + mag at omegas
{0,1,2} on both canonical axes (each = the full complex rfft coefficient = two
real linear functionals; omega-0 is the chunk-displacement invariant). The
frozen HYBRID_PINS' mod-pi/phase-only entries were excluded per the
pre-registration rule (not linear functionals).

Arms (same arch / 8000 iters / seeds 0,1,2): **A** = fwd-only (240 rows);
**B** = multi (720 rows: fwd + reversed + hover from the SAME obs). Eval: 60
held-out scenes x 4 rollouts; commands from held-out chunks of each type,
executed from the same start states. `err` = ||F_realized - F_commanded||_2
over the 6 pinned complex coefficients (canonical frame) / RMS forward command
norm (8.10). `chunk_rmse` = full-chunk RMSE to the command chunk (diagnostic;
typical action entry ~0.26, style-level floor ~0.037).

## Numbers (pooled over 3 seeds; per-seed in results/multicont.json)

| arm | command | err-to-command | chunk_rmse | mean abs action | fwd success |
|-----|---------|---------------:|-----------:|----------------:|------------:|
| A (fwd-only) | fwd   | 0.0102 | 0.040 | 0.259 | 1.00 |
| A            | rev   | 0.0152 | **0.194** | 0.290 | — |
| A            | hover | 0.0122 | 0.099 | **0.079** | — |
| B (multi)    | fwd   | 0.0111 | 0.037 | 0.259 | 1.00 |
| B            | rev   | 0.0098 | 0.036 | 0.259 | — |
| B            | hover | 0.0106 | 0.023 | 0.018 | — |

Per-seed spread is small (A.rev err 0.0138/0.0165/0.0153; B.rev 0.0098/0.0112/
0.0085). Endpoint spread across rollouts (diversity) is preserved for B on all
types (0.004-0.007, same as A on fwd) — no mode collapse.

## Pre-registered bars

- **Bar 1 — PASS.** B executes reverse (0.0098) and hover (0.0106) at <= 2x
  B's forward err (2x bar = 0.0222); both are in fact at or below B's forward
  err itself.
- **Bar 2 — PASS.** B fwd 0.0111 <= 1.5 x A fwd 0.0102 (bar 0.0153; ratio
  1.09). Forward success 1.00 for both.
- **Bar 3 — FAIL as pre-registered.** A does NOT fail reverse on the pinned
  err metric: A.rev = 0.0152 = 1.49x A.fwd, well under the 2x failure flag.

## Why bar 3 came out this way (the real finding)

The pin construction makes the flow regression target identically zero at
pinned coordinates (v = eps~ - a0 with F(eps~) = F(a0)), so ANY pinned-trained
executor learns to leave the pinned coordinates of the source noise
approximately untouched — even for off-distribution commands. Pinned-coordinate
err-to-command therefore CANNOT distinguish "genuinely executes the commanded
continuation" from "pinned coefficients pass through while the rest of the
trajectory stays on-manifold-forward".

The trajectory-shape diagnostic shows the motivating gap is real, just not
where the pre-registered metric looked:

- A commanded reverse produces Frankenstein chunks: pinned coords match the
  reverse command, but full-chunk RMSE to it is 0.194 — 5.2x B's 0.036 and
  ~75% of typical action magnitude. **Forward-shaped elsewhere, not
  reverse-shaped.**
- A commanded hover keeps moving (mean |action| 0.079 = 30% of normal); B
  actually hovers (0.018 = 7%, mostly unpinned style wiggle).
- B's reverse/hover chunk_rmse (0.036/0.023) sit at the style-noise floor —
  the multi-trained executor genuinely executes all three continuations.

## Verdict

Multi-continuation training works: no mode collapse, no forward degradation,
clean selection among contradictory continuations by the commanded invariant
(bars 1-2 pass cleanly). Bar 3 fails as written — but because the metric, not
the hypothesis, was wrong: fwd-only A does fail reverse/hover in trajectory
shape (5x chunk_rmse; doesn't stop when told to hover) while the pin's
passthrough guarantee keeps its pinned err low. Carry-forward: any follow-rate
claim measured ONLY at pinned coordinates is confounded by passthrough; pair
it with a whole-trajectory (or unpinned-coordinate) consistency check.

Reproduce: `~/.local/bin/uv run --with autograd --with numpy --python 3.11
python experiments/toy_multicont/multicont.py` (~4 min CPU; `--smoke` for a
3 s plumbing check). Results: `results/multicont.json`.
