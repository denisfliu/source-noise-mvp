# Status — 2026-08-13 ~21:00 UTC (written before scheduled loss of box access)

Successor to the joint-training arc entries in RESEARCH_LOG (which remains the authoritative
chronology). This is the catch-up page: where every line stands, what is still running
autonomously, and what to check first on reconnect. Physical inventory: docs/EXTRACTION.md.

## The one-paragraph story

The grounded command source's historical 0/10 was a feature-source pairing bug; the joint
(in-checkpoint) head fixed that by construction. The endgame ("tail") problem then decomposed,
by measurement, into: (a) the flat basis cannot EXPRESS the stop (captures 0.34 of stop-segment
variance; the multi-horizon basis mh16 captures 0.81); (b) regression heads emit INVALID
mode-averages at branch states (measured shrinkage; fixed by the generative/CFM head — sampled
commands are full-magnitude and the sampler picks "stop" 85% of the time in the goal region);
(c) what remains is mid-flight command PRECISION (tightest margin, 1.3-1.4x), whose accumulated
error delivers arrivals 0.2-0.7 m outside the goal box, where a correctly-chosen stop then
freezes the miss; and (d) posterior CALIBRATION (which route-mode gets sampled) is uncontrolled —
coupling strength/detachment/basis demonstrably do not govern it (lam=1: 0.0 left-mode; lam=0.3:
0.5; detached: 0.6; and gate ownership flips on training seed alone). The measured causes of (c)
and (d): the head's information diet (no state, language diluted 15-vs-780 tokens in the 4-query
pool, scene-prompt confounded data) and ignorable concat conditioning. The fix now training:
genfilm — the CFM head conditioned EXCLUSIVELY via FiLM from three explicit channels (state,
language-token pool, image pool), i.e., the record system's information diet with language
instead of one-hot, through conditioning the network cannot ignore.

## Running autonomously right now (will complete without supervision)

- GPU1 `c2s7`: seed-7 replication of C2's right-gate 10/10 strict (the only arm that ever solved
  the stop). Eval ~22:10 UTC. Scores -> /home/ubuntu/ctxrun/arm_c2s7_scores.txt, then center
  add-on -> ctr_c2s7_scores.txt. IF THE 10/10 REPLICATES: C2's recipe genuinely solves the right
  endgame; study its mid-flight accuracy (its arrivals land in-box). If not: training lottery,
  C2's headline downgraded like gen16's center completions.
- GPU0 `genfilm`: the FiLM arm (mh16 basis, lam=0.3, single-variable vs gen16). Training until
  ~22:20, then readout gate -> left/right 10-trial eval -> center/compound add-on overnight.
  Read in this order on reconnect:
    1. arm_genfilm_scores.txt readout-gate line (is the distribution's center accurate);
    2. clog_genfilm.npy start rows -> c2-component histogram (left-mode fraction; ladder to beat:
       0.6 detached / 0.5 lam.3 / 0.0 lam1; calibrated ~= 1.0);
    3. the six cells (arm_ + ctr_ scores), esp. right-gate goal entry and center.

## Decision state (Denis)

- CFG: rejected as a primary fix (bandaid); only ever as a final sharpener.
- Contrastive CFM training: rejected (enumeration through the back door).
- Restoring-field / coverage data work: real, measured (b2lam03-right settles outside the box
  with no corrective vocabulary; demos contain no recovery), EXPLICITLY DEFERRED as a data
  problem, alongside covariate shift.
- Tail-weighting and per-band soft pin: rejected as regime-specific patches.
- Claim-rule addition: training-seed variance is ~±5 strict points at 10 rollouts (b2lam03 s42 vs
  s7; gen16 gate ownership flipped on seed) — cross-arm deltas under ~5 need seed replication.
- Pending Denis review for claim tier: C2-right videos (artifact 447bd6f4), b2lam03 compound-left
  5/5 both-gates (corrected directional prompts), the one-hot right-gate record (b5dfc23f).

## Standing instruments (all in experiments/rung3/, headers document usage)

readout gate (joint_head --check) · start-draw histogram (clog first rows; calibration) ·
feature_separation_probe (task info by phase) · manifold_tail_probe (restoring field / on-vs-off
manifold) · tail_attribution_probe (3-link endgame chain) · mh_basis_audit / refit_rrr_basis
(basis geometry) · residual_bimodality_audit · sim_real_c_probe · confirm_vlm_rrr (LIBERO).

## Artifacts (Denis's pages)

grid (all arms x six scenes): 2c0f3000-9f98-4287-9148-236d28b7736a ·
tracker: 430ab907-7c05-4a5b-a7df-bd740f1294f9 · C2-right review videos: 447bd6f4-c6d8-4486-ad06-d74eda0c3977

## Late addition (21:58): third autonomous overnight arm

- GPU1 after c2s7's add-on: `c2genfilm` — C2 routing x FiLM generative head on mh16 basis (the
  first combination of the replicated-endgame routing with the full-information sampler). Same
  read order as genfilm; scores in arm_c2genfilm_scores.txt / ctr_c2genfilm_scores.txt.

## Final pre-cutoff state (22:38)

- genfilm: readout gate PASS 0.8525 (> gen16's 0.831 same basis). Rollouts started at cutoff;
  full six cells + calibration histogram complete overnight (read order above).
- c2s7 add-on complete: center 0/20 success (9-10/10 clean), compounds 0 — C2's full signature
  (right solved, left/center blind) replicates across seeds in every cell.
- c2genfilm: queued on GPU1, will train+eval overnight.
