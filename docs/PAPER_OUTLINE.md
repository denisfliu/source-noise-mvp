# Paper outline — action factoring through source noise

Living document (started 2026-08-08). Update as results land; keep the evidence table honest
about tier (lead = 5 seeds, claim = >=10 seeds + video review). Numbers here must match
`docs/RESEARCH_LOG.md`; if they diverge, the log wins.

## Thesis

An action chunk can be factored into a **coarse command** carried in the *source noise* of a
flow-matching action head and the **residual detail** produced by denoising. Supplying the
coarse component (a) makes the action head steerable and interpretable, and (b) reduces the
demonstrations needed for competence, because the hard-to-learn part is supplied rather than
learned.

Formally: with pi0's interpolant `x_t = t*noise + (1-t)*a` and an orthonormal `U` (D x K),
set `noise = (I - UU^T) g + U c`. Then `U^T v_target = U^T(noise - a) = c - U^T a`, so pinning
`c = U^T a` in training makes the coarse component of the regression target identically zero:
the flow never learns it, and at inference `c` passes through exactly.

## Contributions (claim -> evidence)

1. **A control channel with exact pass-through**, obtained by construction rather than by
   learning to obey a conditioning input. Toy: ~1% command error vs ~26x worse through a
   conditioning branch, diversity preserved. Drone: pin steers ~5.5x harder than language
   conditioning and overrides the visible scene when commanded against it.
2. **Closed-loop competence on a full task** with a deliberately simple system: hard pin +
   clockless state prior. 19/20 strict full successes (L 10/10, R 9/10), no wall clock, no
   VLM in the command loop. [claim tier pending video review]
3. **Low-data result, with a measured crossing point**: with the coarse action supplied,
   3 demonstrations per task give 20/20 gate transits; the matched no-pin fine-tune manages
   10-14/20 transits and 0/20 completions up to 40 demos/task. By 60 demos/task the no-pin
   arm catches up (transits tie 20/20; completions 13/20 vs 9/20, not significant at n=20).
   So the factorization buys competence in the data-starved regime and its advantage decays
   as data grows — the same conditional the LIBERO line found, now located on one curve.
4. **Mechanism decomposition** — the experiment that separates our claim from cheaper
   explanations (see Section 5; this is the section that makes the paper defensible).
5. **A measurement methodology for command heads**: offline R^2 does not predict closed-loop
   viability; basin width, pose-information ceiling, phase-resolved error, and jitter do.

## Section plan

**1. Introduction.** Factoring movement from detail; why the control channel should live in
the source distribution rather than the conditioning branch.

**2. Method.** Interpolant algebra, choice of U (reduced-rank regression: the K directions of
chunk variance most predictable from observation features), soft pin (sigma), serving.

**3. Does the channel work?** (Fig. 1) Toy contradictory-command experiment; drone steering
vs conditioning; command-following error.

**4. Closed-loop system.** (Fig. 2 + video) Record configuration, strict scoring
(directional transit judge + gate-cloud clearance + human video review), 19/20. Include the
goal-phase story: the failure was command-side (prior never commands the settle), fixed by
weighting goal-phase rows 4x — a 2-minute retrain, no change to the flow.

**5. Why does it work? Mechanism decomposition.** (Fig. 3 — the key experiment)
Three controls, all with identical commands from a demo-derived oracle:
   [all cells below are 10 trials/side, reported per 20]
   - **Additive-edit control**: plain flow + post-hoc algebraic overwrite
     `a' = a + U(c - U^T a)`. Reproduces steering exactly (transits 20/20, full 4/20 vs the
     pin flow's 3/20) but clearance halves (8/20 vs 18/20; clips to 0.002 m).
     => the coarse channel steers *algebraically*; pin training buys residual coherence.
   - **Zero-pin control**: source orthogonalized exactly like the pin but `c = 0` — same
     "familiar" source, no answer, no channel. 0/20 transits at n=12, vs pin 20/20 and
     plain scratch 12/20 => the low-data advantage is the *supplied answer*, not source
     consistency; a zeroed source is worse than plain N(0,I). (Its 20/20 clearance is an
     artifact of never approaching the gate — clearance is conditional on transit.)
   - **Oracle ceiling per data size**: record 10/7/10 (transit/full/clean per 10), n160
     10/2/10, n40 10/5/6, n12 8/0/4. => under perfect commands, transit steering is nearly
     free at any data size; goal-phase execution and cleanliness are what demonstrations buy.

**6. Low-data ladder.** (Fig. 4) Stratified subsets (real/synth interleaved per gate task,
nested), equal 5k steps, pin arms served with pin, no-pin arms served plainly, one command
source throughout. x-axis is demos *per gate task* (3, 10, 40, 60, ~100). Two findings:
   - **Route-following is data-free with the pin**: transits 20/20 at every rung including
     3 demos/task; the no-pin arm is 10-14/20 through 40 demos/task with 0/20 completions,
     then ties at 60 (20/20 transits, 9/20 completions).
   - **Terminal precision is a threshold**: completions (single prior, 20 trials/rung)
     3/20 at 3 demos/task, 4/20 at 10, 3/20 at 40, **13/20 at 60**, 19/20 at ~100. Flat
     through 40, then a sharp rise. The pin removes the route problem entirely and leaves
     the settle problem, which needs roughly an order of magnitude more data.
   Report clearance conditioned on completion (flows that attempt the goal box necessarily
   fly closer to the gate).

**7. A grounded command source.** c = MLP([state, prompt embedding]) where the embedding is
the post-fusion language-token representation of the live instruction — no task list, no
classifier, no string matching. 13/20 completions vs 19/20 for the string-matched scaffold on
the same flow; clearance is the weak axis (5/20). Held-out paraphrases preserve command
direction (cos 0.94-0.99) but lose precision (c-R^2 0.94 -> 0.65-0.89); closed-loop paraphrase
cost measured separately. Phase-resolved error shows this head's weak phase is the START
(early 0.881 vs transit 0.966, tail 0.924) — the mirror image of the state prior.

**7b. Where the remaining bottleneck is: predicting c precisely.** Pose information limit of pooled VLM features
(0.14-0.20 m localization -> ~0.21 m command floor; +0.33 m mapping error) against a ~5 cm
gate tolerance; basin width (state prior gain grows with deviation 0.39 -> 0.73 at 1 m;
feature heads decay 0.23 -> 0.11); coverage (union data -> first state-free language-grounded
transits, 4/5); jitter (removable). Grounded command source: [IN PROGRESS].

**8. Negative results** (short, kept because they constrain the design space):
   - Error-matched pin noise (train the flow on the predictor's measured error distribution)
     is *worse* than isotropic: correlated, signal-aligned noise teaches the flow to ignore
     the command subspace (L 0/5 vs 8-9/10).
   - Per-domain U: refitting the basis on real or pooled data lowers held-real command
     predictability (0.50 / 0.69 vs 0.69 / 0.72 for the deployed basis).
   - Learned U end-to-end: joint training assigns the pin to high-frequency detail, not
     coarse movement — coarseness must be imposed by the basis objective.
   - Attention readout over VLM tokens vs mean pooling: no clean win at this data scale
     (R^2 0.912 vs 0.905).
   - Concatenated state+feature heads: the feature channel takes ~85% of the command
     response regardless of information quality.

**8b. Cross-domain: no significant benefit, but a rule about sigma.** A LIBERO low-data
ladder (40 tasks, 2 demos per task, same protocol) does NOT establish the benefit: at 100
episodes per arm, soft pin sigma=0.7 scores 0.49 vs 0.41 for a matched no-pin fine-tune
(+8 points, p~0.25). A 50-episode pass read +18 and was an artifact of a low scratch estimate.
What the sweep does establish, on contrasts far outside noise, is that pin HARDNESS is
decisive there: hard pin 0.16, sigma=0.35 0.24, sigma=0.7 0.49. On the drone the ordering is
reversed: commands there are accurate (prior c-R^2 0.97) and the hard pin is best. The single
rule that covers both: **sigma must scale with command error**, because the pin is
pass-through — accurate commands make hardness free, inaccurate commands require the flow to
retain capacity to correct. Reporting the hard-pin LIBERO numbers alone would have supported
the opposite (and wrong) conclusion that the method is domain-limited.
Basis variants (gripper-free, per-suite, displacement K=6/K=7) were all trained with hard
pins and therefore do not isolate their own effect; the fair soft-pin x basis comparison is
in progress.

**9. Limitations.** Simulation only (gsplat-rendered drone scenes + LIBERO); command source
in the headline system is a scaffold (see 7); most cells are 5 trials (protocol noise
+-5-6 points); single flow architecture (pi0).

## Evidence status

| Result | Tier | Missing before submission |
|---|---|---|
| Toy channel vs conditioning | solid | — |
| Drone steering vs conditioning | solid | — |
| 19/20 closed-loop | claim tier | human video review (pending) |
| Ladder pin vs scratch (crossing at ~60 demos/task) | claim (20/cell) | no-pin arm at ~100 demos/task |
| Ladder threshold (flat to 40, rises 60, saturates ~100) | claim (20/rung, single prior) | — |
| Enumeration-free language prior | claim (20 trials) | paraphrase flight test running |
| Additive-edit control | claim (10/side) | ideally also at n40 |
| Zero-pin control | claim (10/side) | second data size |
| Oracle ceilings | mixed | n160 at claim tier; n40/n12/record still 5/side |
| Command-head measurement suite | solid | — |
| Grounded (non-enumerated) command source | claim: 13/20 completions | paraphrase transfer (flight test running) |
| Generality beyond the drone scenes | NOT established (0.49 vs 0.41, p~0.25) | more episodes, or a domain whose coarse structure dominates |
| sigma scales with command error (cross-domain rule) | lead (both domains, single-seed sigma sweep) | sigma sweep at claim tier |

## Figure list (draft)

1. Method schematic + interpolant algebra (why `U^T v = 0`).
2. Command-following: toy contradictory commands; drone steering strength.
3. **Mechanism decomposition**: three-bar panel (pin / additive / zero-pin) on transit, full,
   clean — plus the oracle-ceiling curve vs data size.
4. Ladder curve: transits and completions vs demo count, pin vs scratch.
5. Bottleneck panel: basin gain vs perturbation radius (prior vs feature head); pose-error
   cascade; phase-resolved command error before/after tail weighting.
6. Qualitative: trajectory overlays for record, additive (clipping), zero-pin (never arrives).

## Open questions worth stating in the paper

- Does the coarse/residual split transfer across embodiments and to real data? (bases share a
  displacement core but diverge in the tail; principal angles 6-83 deg depending on pair.)
- Can the command be produced from language and vision at the precision closed-loop control
  needs, or does it require a different representation than pooled VLM features?
- Is there a re-parameterization of `c` (direction + progress rather than metric displacement)
  that makes closed-loop success less sensitive to command error?
