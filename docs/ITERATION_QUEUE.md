# Iteration queue — 2026-08-27 (written for the big-box handoff, 2 x ~98 GB GPUs)

Ranked open lines, each with context pointer, success criterion, and what the compute boost
changes. Read `docs/status_latest.md` first. The 4090 serialized everything through one
24 GB card; the box removes that constraint — the queue below marks what parallelizes.

## 1. dsplit verdict (in flight; finish first)
Domain-split training (synth learns pin, real learns denoising) is training via
`scripts/run_dsplit.sh`. TODO when the checkpoint lands: sigma probe + map (`--data-dir
data_gate_synth3`!), six sim cells (forgetting detector), and the real-anchor probe suite
(`synthpin_in_real.py`, `real_angle_fix.py` heading/speed/crossing metrics) vs gmsig3 as
control. Criteria pre-registered in RESEARCH_LOG 2026-08-27. If phase-B forgetting shows in
the sim cells -> build v2: per-sample head-loss masking (thread episode_index through
data_loader; real<100/synth>=100 in gate_nav3) + matched-pair swap augmentation (~10-15% of
batches; match tables via `pin_gap_probe.py` machinery). Box: train v2 on one GPU while the
other re-runs all evals.

## 2. Flywheel: sketched successes -> autonomous composition
~40 route-clean sketched compound rollouts exist (traj_skd*, skm5*, skdns1* + clogs with
the served c). Fine-tune flow+head on (compound prompt, executed trajectory) pairs; re-fly
compounds UNGUIDED. Success: unguided CMPL/CMPR > 0/5. This is the north-star low-data
claim end-to-end (human supplies the hard part once; system learns it). Box: whole loop
(train + six cells) fits in an evening with both GPUs.

## 3. Behavior-gap-driven course tuning (fixes 3 things at once)
The sim's endgame flies unlike real (behavior gap 0.62 -> 0.72 endgame; under-speed;
descent graze). We own the planner: iterate course endgame params, score candidates OFFLINE
by |c_oracle_real - c_oracle_synth| at matched states (`pin_gap_probe.py`) — no rendering
until a variant wins, then regen + retrain. Box: render/train several variants in parallel;
this was infeasible serially on the 4090.

## 4. Claim-ladder industrialization (the statistics rule finally affordable)
Two-tier rule: screens 5 trials, claims >= 10 trials + seed rep; 10-SEED runs were never
affordable locally. Box: run 5-10 training seeds of the flagship recipe, 10-trial cells
each, in days. Also promote sketch rows to >= 10 trials/cell and queue Denis's video
reviews (reels already on disk for gmsig3/s7 L/R).

## 5. Real-deployment checklist (blocked only on hardware access)
(a) Recalibrate sigma on real frames (`sigma_phase_probe.py` on data_gate_real -> real
map). (b) Approach-line monitor to auto-issue the rotation verb (gain 0.76, command
dtheta/0.76). (c) Closed-loop real atomics, then a sketched compound in the mocap room.
The Sketchpad works as-is (scene cloud = the real room's reconstruction).

## 6. Scratch-sketch mechanism ablation (cheap, paper-critical)
Serve scratch3 through the sketch pipeline (needs a ~20-line plain-serve sketch variant —
current sketch serve calls the head, which scratch lacks). Prediction: scratch ignores the
noise; isolates the prompt-swap contribution. Converts "sketch works" into "works through
the source-noise channel".

## 7. Paper consolidation
PAPER_OUTLINE.md predates: trust dial, route-clean methodology, sketch prompting, seed
robustness, pin-gap triplet, rotation verb, cross-domain contract, adapter negative result.
The arc is publishable-shaped; consolidate once 1-3 land.

## Infrastructure notes for the box
- BOX_TRANSFER.md is the restore manifest (repo clone + patches + data regen/rsync).
- Two GPUs: server/client eval pairs can use separate cards (drop the 0.45 mem squeeze in
  the six-cell scripts); compounds no longer need sequential clients.
- JAX grabs all GPUs — scope with CUDA_VISIBLE_DEVICES per process, always.
- The box is sometimes shared: nvidia-smi before launching; don't stomp other procs.
- Long jobs: setsid + persistent-dir logs; verify log growth before trusting a launch.
- Every result: RESEARCH_LOG append (absolute date) + FINDINGS_INDEX one-liner + point-cloud
  artifact page + commit + push to origin.
