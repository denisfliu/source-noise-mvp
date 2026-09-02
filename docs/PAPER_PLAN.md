# ICLR paper plan (2026-09-01) — supersedes PAPER_OUTLINE.md

SPINE: a mechanism paper — control belongs in the source distribution, not the
conditioning branch. Title direction: "Commands in the Noise: Steering Flow-Matching
Policies through the Source Distribution."

1. INTRO — conditioning obeys by choice (toy: 26x worse); factorization hypothesis; hook =
   zero-demo composition, adversarial gate relocation, orbit/fig8, declared intent.
   Contributions: mechanism+sigma; test-time command interface + zero-demo results;
   sim<->real analysis; ablations.
2. RELATED — source-distribution diffusion (ICLR'26 re-rendering; our delta: policies +
   commands), flow-matching VLAs, guidance (language cannot redirect — measured),
   behavior prompting (sketches = kinematically naive demos), latent-conditioned policies.
3. METHOD — 3.1 carry identity (U^T v = 0); 3.2 command space + decode (interpretability by
   construction); 3.3 uncertainty conditioning; 3.4 test-time sources (head / sketching +
   state machine + carrot / rotation + tempo verbs); 3.5 cross-domain command supervision
   (co-training, not sequencing).
4. SETUP — domain; data table (center tasks sim-only, compounds ZERO demos = the design);
   Success = collision-free completion; pi0 baseline; seeds/trials protocol.
5. RESULTS + ABLATIONS — Table 1 (80/80 vs 60/80); Table 2 checkmark grid; three ablation
   sentences (channel necessary + injection harms channel-blind; swap drives real command
   quality; sigma trades 1.4 deg for the trust interface); following fidelity 0.070~0.068.
6. COMMANDING WITHOUT DEMONSTRATIONS — compounds via sketch; 14 relocated poses (70/70);
   orbit/fig8/tempo (0.9 gain); carrot disturbance rejection; mechanism control closes.
7. SIM-TO-REAL (open-loop, honest) — pin-gap triplet; synth commands valid on real;
   rotation 0.89; NEGATIVE RESULTS BOX: adapters ruled out, sequential curricula strand
   the head (feature drift).
8. LIMITATIONS — one room/embodiment; open-loop real; linear invariants; composition needs
   the human interface; screen-tier ns flagged.
APPENDIX — derivations (KEY_MATH); judging protocol + route-clean + aperture-bug hygiene
   case; per-seed tables; sketch state-machine engineering; intent cockpit; repro (repo).

FIGURES: F1 method/carry diagram; F2 vocabulary specimen + intent decode; F3 sketch
compounds (clouds); F4 moved-gates montage; F5 orbit/fig8/tempo; F6 ablation fans.
TABLES: 1 main, 2 checkmark ablations, 3 test-time commanding (with dual-number honesty on
thin-margin rows; relocated-gate row pending transformed-cloud clearance).

OPEN (Denis): (1) sketch/human-element placement — main showcase vs applications;
(2) resurrect the toy 26x figure in the intro or cite only.
