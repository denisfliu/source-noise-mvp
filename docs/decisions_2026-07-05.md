# Decisions on status_latest DECISIONS NEEDED — 2026-07-05, from Denis via laptop-side Claude

## D1: Option H — GO, option (a)

Run the hybrid pin. Fix the generator's 19% self-collision rate first so the
ceiling is clean, then: full complex coefficient (phase + magnitude) at
magnitude-coherent bins (lat-1 clearance; use a magnitude-agreement criterion
analogous to the energy floor — pre-register the threshold before looking at
arm results), phase-only elsewhere; prior predicts magnitudes alongside
resultant vectors; confidence gating unchanged. Re-run the full gate battery,
not just G3 — G4 especially (magnitude pinning carries more information;
watch side diversity under the mod-pi bin and the leakage R^2, which should
rise from 0.08 but stay well short of transcription).

Framing to record in the toy_frame README (it's a finding, not a footnote):
the paper's phase-only form restriction is not a neutral simplification that
happened to fail — in images, magnitude IS appearance, so phase/magnitude
maps exactly onto structure/appearance. The toy shows this split is
image-specific: in control, safety-critical structure is partly metric
(clearance amplitude), so structure straddles both components. G3's failure
mode (F-prior BELOW A, not merely at it) is the dropout economics operating
again: the always-on pin displaced obs->clearance learning while the
amplitude slot stayed Rayleigh-random. Write it that way.

If H passes G3: claim = "coherence-discovered complex structure improves
no-oracle success." If H fails G3 with endpoints/shape still good: the
problem is deeper than form; stop and bring the failure analysis for
discussion before touching the criterion (D1 option (c) is NOT
pre-authorized).

## D2: Arm C success oracle — option (b) first, then (a) conditionally

Build the minimal oracle now (gross displacement toward the known target
object per replan, ~2h): it tests the pin-source plumbing end-to-end and is
unbiased. Promote to the scripted per-task oracle (a) only if minimal-oracle
numbers are sane (C within striking distance of A, no pathologies). Fairness
constraint to hold in both versions: the oracle may only translate task+scene
into the same goal A's vision already sees — no phase logic that amounts to
planning help, and document exactly what the oracle reads from sim state.
Deferring (c) is rejected — the A-vs-C success comparison is the H1 gate.

## D3: Step budget — rule stands, no new decision

Pre-registered rule unchanged. Treat the 95 @ 20k as unresolved noise until
the 25k point; do not restart anything on its account. If the 25k point
triggers the flag-back, include: restart cost for C_s42 at that moment, and
the ceiling-compression counterargument (at 95%+ arm differences squeeze;
89% @ 15k preserves headroom for the A-vs-C comparison, which is the point of
the sweep — the per-arm curves already answer the "would it wash out at
convergence" critique empirically). Denis leans keep-15k.

## D4: Phase 2 shape — defer until H lands, but record two notes

1. The amplitude finding does NOT hand the fork to VQ codes by default: the
   coherence criterion generalizes naturally to complex coefficients
   (magnitude agreement across demos is measurable the same way phase
   agreement is — Option H's bin selection is exactly that estimator in
   miniature). If H passes, coherence-first stays live with "structure =
   coherent complex content," and VQ-on-the-residual remains the layered
   option.
2. Whatever wins, Phase 2 codes must (i) be always-on with a prior at
   inference (dropout finding) and (ii) carry metric content (tonight's
   finding). Add both as hard constraints in any Phase 2 design doc.

## Standing instruction

Same protocol as tonight for anything new: flag decisions in status_latest
under DECISIONS NEEDED with options + a recommendation; CPU-side work that is
pre-authorized above needs no further sign-off.
