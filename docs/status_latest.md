# Status — 2026-08-27 (local-4090 arc, pre-handoff to the big box)

Single source of truth for "where is the science right now." Newest-first pointers:
`docs/RESEARCH_LOG.md` (bottom = frontier), `experiments/FINDINGS_INDEX.md` (one-liners),
`docs/ITERATION_QUEUE.md` (what to do next and why). Repo:
github.com/denisfliu/source-noise-mvp (private).

## Headline results (all route-clean judge + clearance; video = the remaining claim gap)

- **Flagship, seed-replicated: gmsig3 + gmsig3s7 = 80/80 route-clean judge on the four
  atomic gate cells (77/80 clearance-clean).** sigma-conditioned GMM head x mh16 basis on
  gate_nav3 (real 0-99 + synth 100-299, shared norm stats). Scratch control 72/80 (CFR 7/10
  both seeds) — the pin's edge is seed-stable and lives at CFR + precision.
- **Sketch prompting (the human element): hand-drawn pin sketches fly both compounds 5/5
  route-clean** (CMPL also 5/5 clearance); 4-5-click minimal sketches replicate across
  training AND rollout seeds once near-structure waypoints sit on true geometry ("margin is
  the portability budget"). Flights track drawn polylines to ~7 cm.
- **Cross-domain contract verified zero-shot, both directions**: synth-authored pins
  (oracle AND sim-twin head) execute task-validly on real observations (11/11 in-aperture
  right-gate crossings, full speed); a real demo replayed as a pin sketch flies the sim
  cell 5/5 strict clean. "Plan in sim, fly in real through the pin."
- **Pin command vocabulary on real perception**: sketch = route topology, rotation verb =
  aim (heading error 10-20 deg -> 3-5 deg, dose gain 0.76), sigma = trust. Language alone
  cannot redirect (0.05 cstd contrast, both domains).
- **Sim-to-real is measured, not vibes — the pin-gap triplet**: execution gap ~0,
  prediction gap at the head's own floor (the head is already domain-reconciled; cheap
  adapters ruled out with evidence), behavior gap 0.62 cstd concentrated in the endgame
  (under-speed + descent corridor) = data authoring, tunable offline.

## In flight

- **dsplit experiment** (Denis-approved): phase A synth-only+head 4000 steps -> phase B
  real-only flow-matching (+1500, HEAD_LAM=0). Pre-registered criteria: hold sim six cells
  at gmsig3 level AND improve real-frame speed/crossing. Forgetting => interleave next.
  Chain: `scripts/run_dsplit.sh`; post-eval still to be run when it lands.

## Standing flaws / gaps

- Center-west-post goal-descent graze: the ONE systematic clearance flaw (CFR atomics,
  sketch returns, both seeds). Data-side fix via behavior-gap-driven course tuning.
- Trust dial not calibrated on real frames (corr(sigma*, err) ~0 vs 0.82 in-domain) —
  recalibrate on data_gate_real before any real closed loop.
- Compounds remain 0/5 autonomous (structural, both seeds, all arms) — the flywheel
  (distill sketched successes) is the planned attack.
- Claim-tier: human video review pending for every 2026-08-25+ row.

## Judging rules (upgraded this arc — do not regress)

Strict success = falsify posthoc transit judge + ROUTE-CLEAN (zero wrong-direction aperture
passes, demos unanimously clean) + `gate_clearance.py` (0.18 m) + human video. The
right_and_center safety YAML gate_1 was a half-width box (fixed 2026-08-26; region-box bug
class) — treat every region box as suspect until checked against the scene cloud.

## Operating notes for the next Claude

- **Every run's trajectories get a point-cloud page, unprompted** (Denis standing rule;
  `experiments/rung3/viz/cloudviewer.py` + build_*_page.py patterns, publish as artifact).
- Screens = 5 trials; claims >= 10 trials + seed rep (protocol noise +/-5-6 pts).
- Never train product models on sim ground truth; deployable supervision = demos, human
  input (sketches/corrections), generic perception. Sketch UIs may show judge overlays as
  scaffold but humans draw against the perception-derived cloud.
- Ordering rule: U from action statistics -> flow trained with oracle c = U^T a -> features
  -> prior/head.
- Norm-space footgun: every c computation uses normalized actions (shared gate_nav stats).
