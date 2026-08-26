---
name: source-noise-mvp
description: "Denis's source-noise action-steering research project on the EC2 box (ssh alias `ec2`) — status, motivation, and key design findings"
metadata: 
  node_type: memory
  type: project
  originSessionId: a7c6735e-c699-40b9-9582-acb02c308eb7
---

Denis's active research project (as of July 2026): testing whether movement commands
embedded in the flow-matching **source noise** of a VLA (π0/openpi on LIBERO) bind
harder than conditioning inputs. Repo `~/code/source-noise-mvp` on the EC2 box
(ssh alias `ec2`, ubuntu@10.150.0.76); docs sync locally to `~/Desktop/snmvp-docs/`.
Adapted from an ICLR'26 submission on subspace phase-invariant sources for image
re-rendering (`~/Downloads/unpaired_rerendering_subspace.pdf`).

**Denis's real motivation** (clarified in discussion 2026-07-04): not command-following
but a *learned geometric frame* for actions — the model should learn what actions ARE
geometrically, yielding (a) task generalization after embodiment is learned and
(b) better learning via geometric grounding. He is skeptical of oracle-based evals:
the always-on pin means arm C is never evaluated without oracle info arm A lacks, so
Phase 1 tests only the delivery mechanism, not the motivation. The honest test is
no-oracle eval through a learned scene→structure prior.

**Key established findings:**
- Toy: noise-pinned arm executes contradictory commands ~26× more precisely than a
  conditioning branch; adherence *precision* (err-to-command), not follow rate, is
  where channels separate.
- CFG-style pin dropout KILLS the channel (the model learns the obs decode instead —
  the pin only survives as the *cheaper* feature). Pin must be always-on; no dual-mode
  checkpoint exists.
- Phase 1 sweep (A×3, C×3, B×1, 15k steps, pi0_libero) launched 2026-07-04 on the
  box's two GPUs; ETA ~Jul 7.

**Learned-frame toy result (2026-07-05, MILESTONE)**: coherence-discovered structure
(external criterion, NOT the flow loss — separation of powers is load-bearing against
transcription) pinned into source noise improved no-oracle held-out success **+17.3 pts
over baseline** (F-prior 62.3% vs A 45.1% vs F-rand control 44.1%; prior ≈ oracle).
First evidence for Denis's hypothesis (b). Two design findings en route:
- Phase-only pins FAIL (first G3 run: F-prior *below* A): the paper's phase/magnitude =
  structure/appearance split is image-specific; in control, safety-critical structure is
  partly metric (clearance amplitude). Fix = hybrid pin (complex coefficient at
  magnitude-coherent bins, CV<0.15 criterion). Discovered structure also BEAT the
  hand-defined displacement invariant (C-disp ≈ A).
- Phase 2 hard constraints: codes always-on with a prior at inference; must carry
  magnitude. Coherence-first stays live ("structure = coherent complex content").
Caveat: toy has planted structure; LIBERO-scale coherence frame (canonical rotation,
multi-demo modalities) is the open design question. Docs on box: `docs/
learned_frame_toy_plan.md`, `docs/decisions_2026-07-05.md`, `experiments/toy_frame/`.
Workflow with box-side Claude: it flags DECISIONS NEEDED in `docs/status_latest.md`;
Denis replies via rsynced decision docs drafted here.

**Phase 1 final result (2026-07-07)**: a COUPLING SPECTRUM, not the original binary.
Both conditioning (arm B) and noise-pin (arm C) capture the model (command-less eval
~0% for both — "conditioning is ignorable" is FALSE at 3B scale). Difference is
coupling STRENGTH: C binds ~11x tighter under contradiction, fully steerable
(follow 1.0, calibrated to mm), but pays ~40-pt success tax executing the prior's
~25% errors; B is loosely coupled (76% success, unsteerable against scene). A (no
channel) 90%. No held-out-placement gap for any arm (original H1 metric void on
LIBERO-Spatial). Steerability is the clean win; task success deprioritized by Denis.

**Cross-embodiment direction (2026-07-17, aim (a)/H3)**: architecture = VLA split at
one seam — FROZEN shared front-half (VL trunk + invariant readout head, gives language
conditioning free) produces an embodiment-agnostic OBJECT/GOAL-CENTRIC invariant;
LEARNED per-embodiment executor (flow head, invariant pinned into its source noise) is
the ONLY thing re-learned on a new body. Iterate cheaply by caching frozen front-half
invariants. Cross-embodiment COHERENCE (toy_frame estimator, modalities→embodiments)
discovers+MEASURES shared structure (predicts transfer; handles arm-vs-drone workspace
divergence as a measured number). Ladder: Rung 1 = 2D multi-morphology toy (arms +
holonomic point-robot=drone analog, CPU) → Rung 2 = robosuite arms + small diffusion
policy + gsplat perception (1 GPU) → Rung 3 = OXE/VLA confirmation. Rung 1 spec at
`docs/toy_embodiment_plan.md` (CPU, kicked off 2026-07-17); design at
`docs/cross_embodiment_plan.md`; overview in `docs/REPRODUCTION_GUIDE.md` §11.

**Rung 1 RESULT (2026-07-17, built+run by Claude on the box, `experiments/
toy_embodiment/`)**: G-frame PASS (cross-body coherence recovers shared frame;
c(arm,arm)=0.93 > c(arm,point)=0.80). **G-transfer PASS** — freeze coherence-learned
frame S_A + scene->invariant prior on set A {arm2,arm3,arm4}, adapt ONLY the executor
on a held-out body's few demos: pooled over 3 seeds at low n, T beats scratch,
conditioning, AND random-frame on BOTH held-out bodies (point/drone-analog: T=0.449 vs
S=0.356/Cond=0.399/Trand=0.361; arm4: T=0.387 vs 0.33/0.33/0.30). First positive
evidence for aim (a)/H3. Signal cleanest at n=10-25 (n=5 very noisy). T-oracle >> T
confirms prior quality is the ceiling (same as LIBERO). **G-predict did NOT hold with
n=2**: lower-coherence point had HIGHER transfer gain (0.094) than higher-coherence arm4
(0.057) — reversed; needs a proper body-ladder. Design choice (Denis): task-space
actions for all bodies (invariant linear, pin exact); embodiment = reach/feasibility.
Note: box disk was 99% (Phase 1 checkpoints 1.2TB); pruned intermediates -> 51%, kept
all finals.

**Rung 2 pre-step (2026-07-18, `results/transfer_v2/`)**: tightened S_A to 3 pins
(prog-omega0 WITH magnitude + lat-omega1/2 phase-only; energy floor 0.10). Added a
4-body held-out ladder (arm5/arm_short/point/point_drag). Findings: (1) core transfer
robust — T>S AND T>Trand for ALL 4 bodies at both n=10 and n=25, so transfer helps and
the LEARNED frame specifically matters, even with the tighter 3-pin frame. (2) Pin-vs-
conditioning edge is LOW-DATA-ONLY: at n=10 T>=Cond for all bodies; by n=25 Cond
overtakes T (T-Cond ~ -0.05). Quantifies the coupling-spectrum/"grounding helps when
data is scarce" story; for cross-embodiment few-shot (n~10) the pin is still the right
choice. The auto G_transfer_pass=false is an artifact of pooling n=25 (where Cond wins)
into "low n". (3) G-predict untested by that ladder (coherence didn't spread; the pooled c() metric
was dominated by set-A self-agreement).

**G-predict RESOLVED (2026-07-18, `results/transfer_v3/`)**: fixed the metric
(alignment-to-set-A-consensus, sensitive per-body) and built a CONTROLLED coherence
sweep via point_phase(theta) bodies (lateral detour phase rotated theta=0/15/30/45,
endpoint exact). Got a real spread: alignment 0.797/0.725/0.718/0.639. Transfer gain
(T-S): 0.117/0.085/0.096/0.098 — **does NOT track coherence** (concordance 0.5 =
unrelated; only signal is most-aligned phase0 has highest gain, rest ~flat). VERDICT:
**coherence does NOT reliably predict transfer gain.** Interpretation: transfer benefit
rides on the ENERGY-DOMINANT shared structure (endpoint/prog-omega0), which is
embodiment-invariant by construction; degrading a minor component's (lateral phase)
alignment doesn't remove the benefit. The Rung-1 "reversal" was noise. This STRENGTHENS
the main story: transfer is robust because the goal/safety-critical structure is
physically shared across embodiments, not a fragile function of coherence. Core transfer
result held throughout all 3 batteries: T>S and T>Trand for every body (transfer helps,
learned frame specifically); pin>conditioning only at very low n (n=10), conditioning
catches up by n=25. Next: Rung 2 proper (robosuite + small diffusion policy + gsplat,
free GPU). Spec: `docs/rung2_plan.md`.

**Rung 2 scaffold (2026-07-19, `experiments/rung2/`)**: env validated. Working config
(CRITICAL): robosuite==1.4.1 + **mujoco==2.3.7** (mujoco 3.x breaks robosuite 1.4.1 via
mj_fullM signature); headless MUJOCO_GL=egl, low-dim state only; launch via box-side
run_*.sh script + nohup (SSH to box drops intermittently — inline `ssh "...&"` launches
die). env_check.py PASSED: Panda/Sawyer/IIWA share action_dim=7 (6 EE-delta OSC_POSE +
gripper) on Lift = the toy's task-space-actions design at real scale. Next unbuilt:
scripted multi-arm demo collector -> coherence frame in 6-DOF EE space -> small flow/
diffusion executor + prior + freeze-and-adapt transfer -> gsplat perception. Both GPUs
free. Scaffold status doc: `experiments/rung2/README.md`. NOTE: box SSH connection has
been flaky all session (frequent timeouts, always works on retry) — worth checking VPN.

**Rung 2 FIRST REAL RESULT (2026-07-20, `experiments/rung2/`)**: collected planar
EE-reach demos across 4 robosuite arms (Panda/Sawyer/IIWA/UR5e, shared targets, 80/arm;
demo success 90/64/76/89% = real cross-arm tracking differences). Offline coherence +
freeze-and-adapt transfer (held-out UR5e, reuses toy machinery). VERDICT: on easy
free-space reach, **transfer does NOT help — scratch beats it at every n** (T/S:
0.31/0.62 n5, 0.60/0.75 n10, 0.66/0.79 n20) and T≈Trand (learned frame no better than
random). Expected & consistent: free-space reach is trivial for scratch, the cross-arm
prior only adds error (obedience tax, like LIBERO), and the only structure in a
straight reach is the endpoint which scratch gets from obs for free. => Transfer benefit
requires a HARD task where few-demo scratch struggles (the toy needed OBSTACLES). NEXT:
add obstacles to the robosuite reach and rerun. Pipeline bugs fixed en route: flow_embod
H hardcoded to toy 20 (set fe.H=32), action normalization pitfall (raw ~0.006 m/step
deltas underfit -> ACT_SCALE=263), and the pkill/pgrep pattern matching the ssh shell
itself (put kills inside box-side scripts). Result: `findings/rung2_transfer/`.

**OAT / VLA^2 direction (2026-07-20)**: Denis's idea = a "VLA^2" where a top model
predicts an action-embedding and per-embodiment denoisers consume it; explored via the
OAT paper (arXiv:2602.04215, Ordered Action Tokenization — FSQ bottleneck + nested-dropout
ORDERED tokens: early=coarse/shared, late=fine). Built toy-scale OAT in autograd
(scratchpad->box `experiments/toy_embodiment/oat.py`, `gate_oat.py`, `oat_transfer.py`;
all CPU). NOTE: box `.venv` had been wiped to bare numpy (repo imports autograd
everywhere) — installed autograd via `~/.local/bin/uv pip install`, added `autograd>=1.9`
to pyproject deps. **Rung-1 GATE PASS**: OAT tokenizer trained jointly on set-A canonical
chunks puts goal (radius/endpoint MI 1.22 bits, detour) in tokens 0-1 and leaks ~0 body
identity (body decode at chance from first 2 tokens, rises only in tail) — nested-dropout
ordering does NOT transcribe embodiment; coarse prefix is a legit learned invariant.
Caveat: body-info is genuinely low in the 3-arm toy (reach only bites near limit), so the
gate proves the prefix is CLEAN not that it's USEFUL. **Transfer battery VERDICT: OAT
invariant LOSES to the hand-built coherence frame at toy scale.** (1) OAT prefix transfers
(OATcond > scratch, > shuffled-prefix control) but only at n=25, not n=10 (net-harmful at
low n even with oracle prefix — executor burns its data budget learning to read the
invariant). (2) CONDITIONING >> source-noise PINNING for OAT (opposite of coherence): the
coherence pin is exact FFT phase = linearly the action's low-freq content (works free);
the OAT subspace-pin drops 8 abstract latent dims into raw noise coords, nonlinear map
must be learned -> too costly at low n. **The pin (project's signature mechanism) does not
transfer to dense learned latents.** (3) Coherence beats OAT decisively at n=10, OATcond
only matches cohT at n=25 (pooled OATpin -0.18 vs cohT). Open cheap follow-ups (not yet
run): K_pref-pin sweep {1,2} (gate showed goal captured & body cleanest at K=1-2, so a
low-dim OAT pin might recover coherence-like low-n strength); VLA^2 proper = joint OAT
encoder + per-body conditioning decoders + decoder-diversity co-training, sweep K as the
coupling knob. Results: `results/oat_gate/`, `results/oat_transfer/`.

**Complexity-crossover result (2026-07-20, `results/oat_complexity/`, `mb_dataset_hard.py`)**:
tested Denis's hypothesis that exact-pin/coherence wins only because the single-obstacle
toy is ~10 bits & linearly FFT-pinnable. Built a multi-obstacle task (n_obst=1/2/3, each
detour = richer/more-nonlinear shared structure; generator uses linear-solve bump
amplitudes + hard per-timestep lateral projection so the point robot's planned path is
feasible ~0.9-1.0; obstacles kept small & x-separated so a monotonic-progress path can
clear each). Held-out point+arm4, fixed n=25, 3 seeds. VERDICT: **partial vindication,
direction confirmed.** On the richer task OATcond (conditioning) BEATS coherence at BOTH
n_obst=1 (+0.05) and n_obst=2 (+0.07), margin GROWING; and by n_obst=2 **coherence has
collapsed to scratch level** (cohT 0.398 / cohCond 0.348 vs S 0.356) while OATcond (0.472)
still transfers (+0.12 over scratch). Coherence's frozen pin count DEGRADES with
complexity (5->3->3) as detour energy spreads across FFT bins — the predicted mechanism.
So coherence's earlier win was task-simplicity-dependent; on richer tasks the learned OAT
invariant wins. CAVEATS: (1) not a clean "coherence-leads-then-OAT-overtakes" crossover —
OAT already led at n_obst=1 (smaller obstacles meant coherence only got 5 pins even there,
not its clean best case). (2) n_obst=3 is a DATA-STARVATION FLOOR: n=25 can't learn the
3-obstacle task at all (ALL arms 0.09-0.15, ORACLE only 0.226), so that point is
uninformative and drags the auto crossover-trend metric negative; the high-complexity end
needs higher n (50/100) or a gentler complexity step to read. (3) subspace-PIN stays worse
than conditioning throughout (0.685 vs 0.742; 0.354 vs 0.472) — pin confirmed the wrong
coupling for dense learned latents; CONDITIONING is OAT's channel. NEXT (Denis chose VLA^2
proper): build joint OAT encoder + per-body CONDITIONING decoders + decoder-diversity
co-training, sweep prefix K as coupling knob; and re-run the high-complexity end with
n=50/100 for a clean crossover figure. Env fix applied: added `autograd>=1.9` to
pyproject deps (box .venv had been wiped to bare numpy).

**Gate/drone pivot + adapter architecture (2026-07-20)**: Denis's north star clarified —
one-shot an IRL drone through a gate; architecture = train ONE big thing once (frozen) +
MINIMAL training of a tiny per-embodiment adapter on drone data. Built gate-passage task
(`gate_dataset.py`: pass THROUGH apertures, n_gates=1/2/3 = drone racing; drone analogs
point=ideal quad, point_drag=inertia/realistic — ceiling 0.94/0.78/0.34 as gates rise).
Crossover n=100 (`results/oat_complexity_n100/`): OATcond/pin BEAT coherence by +0.08..0.10
across n_obst=1,2 (coherence again decays to scratch by n_obst=2); NOT a clean monotone
"advantage grows with complexity" crossover; n_obst=3 too hard to read even at n=100
(all ~0.16). Pin ties cond at n=100 (pin is data-hungry not dead). Gate one-shot
(`results/gate_oneshot/`, UNFAITHFUL eval — scores policy output directly, no dynamics):
shared invariant HURTS, scratch dominates (point n10 scratch 0.99 vs OAT 0.05), one-shot
lift ~0 for all — because gate geometry is OBSERVABLE (in obs) and the eval lacked an
execution loop. **KEY RESULT — gate ADAPTER (`gate_adapter.py`, `results/gate_adapter/`,
FAITHFUL: policy->command->body.realize()[dynamics]->score realized path)**: the fix for
the wrong-split failure = FREEZE the task INTENT (trained once on set-A ARM expert cmds),
make the tiny adapter the DYNAMICS/inverse-model, compose as RESIDUAL (a_cmd = shared(obs)
+ g_body(obs)), BC targets = per-body ILC inverse-dynamics expert commands (ILC recovers
point_drag pre-compensation: 63%->93%). VERDICT: **train-once + tiny residual adapter is
2-4x more DATA-EFFICIENT than scratch at low drone data** (n=3-10; e.g. ng2 point n3
adapter 0.34 vs scratch 0.08, n10 0.57 vs 0.20; ng1 point_drag n1 0.24 vs 0.07), across
both bodies & gate counts; scratch only catches up by n=30. Validates Denis's architecture
WITHOUT OAT/VLA^2. Design lessons: (a) freeze INTENT adapt DYNAMICS (not the reverse);
(b) RESIDUAL composition not conditioning; (c) inverse-dynamics (ILC) BC targets. CAVEATS:
absolute passage still modest (adapter 0.24-0.57 at n<=10 vs ceiling ~0.9) — shared model
is a BC regressor that underfits multimodal within-aperture style; zero-shot low (0.19-0.29).
Next to strengthen toward true one-shot: flow/diffusion shared intent (not BC regression);
proper inverse-dynamics adapter (obs+desired->cmd, not obs-only residual); then a real
drone sim (gym-pybullet-drones) keeping the same architecture. OAT/coherence invariant is
orthogonal to the drone task (geometry observable there) — matters where shared structure
is UNobservable.

**RE-CENTERED on steerability+interpretability (2026-07-20/21)**: Denis clarified the
PROJECT GOAL = steer actions by intelligently constructing the source noise, with
interpretability; x-embodiment is a SIDE BONUS. Papers to use: original basis paper
(unpaired_rerendering_subspace, subspace phase-invariant sources) + NEW **CSFM**
(arXiv:2602.05951, "Better Source Better Flow", Kim/NYU-KAIST): flow matching doesn't need
a fixed Gaussian — LEARN a condition-dependent source p_phi(X0|C)=N(mu_phi(C),sigma_phi(C)),
jointly with velocity field; variance-only reg (KL sigma->1, mean FREE, Eq9) + directional
cosine align (Eq10); reduces the FM intrinsic-variance term (Eq6) -> straighter flows,
3x faster. SYNTHESIS: Denis's pin overwrites a FIXED Gaussian -> raises intrinsic variance
-> the Phase-1 ~40pt success tax; CSFM = "the pin done right" (learn/relocate source mean
instead of overwriting). Built steerability batteries in `experiments/toy_frame/`
(`steer_battery.py` single-axis, `steer_multiaxis.py` two-axis; command = lateral FFT
bins). RESULTS (obstacle-detour toy, steer the lateral bend): compare injection points
pin vs condition vs CSFM vs plain. **CSFM steers cleanly (slope~1.1, r2~1.0) and BEATS the
PIN decisively as a source-construction channel.** SURPRISE contradicting the project's
founding thesis: the **PIN UNDER-RESPONDS** (single-axis slope 0.48, saturates at ~half the
commanded range; multi-axis Jacobian diagonal all <=0.47) — in a small-MLP toy the pinned
source competes with obs during ODE integration and partially loses. So "pin binds 11-26x
tighter" (LIBERO/pi0, 3B, preserved subspace) does NOT replicate at toy scale/naive single-
bin pin — likely scale/architecture/preserved-subspace-dependent (NEEDS verification with
the proper HYBRID preserved-subspace pin before over-claiming). Multi-axis: **CSFM best
disentanglement (leakage 0.155 < condition 0.204 < pin 0.218), strongest per-knob response
(diag ~0.9 on 3/4 coords), best follow.** No "tax" regime in this toy (plain weak 0.38, so
commanding the correct bend HELPS all arms). CSFM ~ plain conditioning on raw follow/success
(condition slightly higher succ 0.81 vs 0.71) -> CSFM's justification over conditioning is
the STRUCTURED/INSPECTABLE source LATENT (disentangled multi-axis control), which the
multi-axis leakage numbers support. Composition additivity messy for all (nonlinear).
Building an HTML artifact visualizing the 5x5 trajectory steering grid per arm as the
presentable demo. Data: `results/steer_battery/`, `results/steer_multiaxis/`
(+trajectories.json). NEXT: verify pin fairly (preserved HYBRID subspace); push CSFM
interpretability (source-space structure, more axes); a nicer steering demo.

**CORRECTION — pin IS the best steerer, thesis VINDICATED (2026-07-21,
`steer_probe.py`, `steer_demo_gen.py`, `results/steer_probe/`, `results/steer_demo2.json`)**:
the earlier "pin under-responds (slope 0.48) / CSFM beats pin" was a FINITE-DIFFERENCE
JACOBIAN ARTIFACT in steer_battery/steer_multiaxis, NOT real. Verified by a DIRECT
command->produced coefficient sweep (pure Im1, Re=0): **pin tracks EXACTLY** (cmd Im1
-2.58->produced -2.58, slope 1.00, zero Re1 cross-leak). Clean per-arm metrics (direct
sweeps, 60 held scenes): PIN slope 1.00 / phase err 1.1deg / mag@2x 1.00 / cross-leak 0.018
(all perfect); CONDITION slope 0.92 / 9.8deg / mag@2x 0.61 / leak 0.166; CSFM slope 0.92 /
15.2deg / mag@2x 0.56 / leak 0.215. So the **source-noise PIN is the most precise,
disentangled, EXTRAPOLATING steering channel** (forces the source coordinate, flow
preserves it -> obeys 2x commands); conditioning & CSFM steer DIRECTION but SATURATE in
magnitude (~60% of a 2x command, regularize to in-distribution) and leak more.
**"CSFM is the pin done right" was WRONG for STEERING** — CSFM improves generation
(FID/convergence), not steering precision; for exact control the hard pin already wins.
CSFM source latent DOES have structure (dmu/dC full rank-6, mean off-diag cos 0.21,
Fourier-alignment 0.42) but that doesn't beat the pin's exactness. LESSON: measure steering
by DIRECT command->produced sweeps, not finite-diff Jacobians around the natural point.
Artifact rebuilt (pin=winner, canonical frame, verified metrics), same URL. Flip-flapped
once (presented CSFM-wins then corrected) — verification caught it. NEXT: this is a clean
presentable result (pin steers exactly, disentangled, extrapolates); could add the
preserved-subspace multi-pin story and connect to interpretability (source axes = Fourier
modes you can inspect/compose).

**KEY MECHANISTIC TRUTH — the pin is a PASS-THROUGH, not denoising (2026-07-21,
`steer_passthrough.py`, `results/passthrough.json`)**: Denis asked "is the pin just
predicting the denoised action so it never gets denoised?" ANSWER: YES, confirmed.
Instrumented the ODE: for the pin, the pinned lateral bin-1 coefficient of x_t is DEAD
CONSTANT = command (2.0) across ALL 20 Euler steps, and the velocity component along the
pinned direction is EXACTLY 0. Reason: training pins source coord to the action's OWN coord
=> FM target v=eps-a0=0 there => model learns zero velocity => coordinate is CLAMPED, never
denoised. Conditioning by contrast GENERATES the coeff (evolves 0.25->1.94 over steps, ~20%
of velocity on that dir). IMPLICATIONS (reframes "pin wins at steering"): (1) pin's slope-1
exactness + 2x extrapolation are TAUTOLOGICAL (read back the clamp), so "pin beats
conditioning at steering" is APPLES-TO-ORANGES — pin gets exactness by NOT generating the
coord; conditioning pays for actually generating it (=> its saturation). (2) pin CANNOT
refine/refuse a command => executes infeasible commands verbatim = the Phase-1 ~40pt
obedience tax, now mechanistically explained. (3) The genuinely non-trivial part: the
COMPLEMENT is still denoised conditioned on the clamp (velocity ~all on unpinned dirs) —
the model builds a coherent valid action AROUND the dictated subspace. HONEST framing of
the whole method: NOT "model learned to obey" but **"clamp a chosen source subspace; flow
generates the rest conditioned on it."** Demo caption ("best steerer") should be recast to
this. Open questions: which subspace can be clamped w/o breaking the task (obedience-tax
feasibility); is clamp+complete more useful than a conditioned generator that can refine?
Did NOT re-edit the artifact yet (flip-flopped once already; get Denis's read on framing
first).

**DEFLATION — the pin == predicting Fourier modes; flow adds nothing here (2026-07-21,
`steer_ablation.py`, `results/ablation.json`)**: Denis asked "are we just forgoing flow
matching and predicting Fourier modes? does it generalize x-embodiment?" ABLATION: a plain
MLP regressor (obs->canonical action chunk, MSE, NO flow, NO source noise) with the
structure modes (lateral bins 1,2) OVERWRITTEN by the command MATCHES the pin: steer slope
1.0 (exact by construction), success_natural 0.767 vs pin 0.80, success_contradictory 0.067
(same obedience-tax collapse). => **flow matching does ZERO work for control or success in
the toy; the source-noise pin is equivalent to "predict the action, assign the structure
Fourier modes."** Flow's only distinct capability (multimodal residual completions) is
irrelevant to steering/success here (CAVEAT: may matter at real-VLA scale where actions are
high-dim/entangled/multimodal — untested; but the TOY has never shown a flow/source-noise
benefit over mode prediction). Q2 x-embodiment: shared structure modes transfer ONLY in a
shared task/EE space and only the physically-shared ones (endpoint/coarse) — exactly the
earlier cross-embodiment result; native-space modes differ; the body-specific REALIZATION
(dynamics) is NOT in the modes = the residual/adapter (the drone gate finding). UNIFIED
PICTURE across the whole project: **structure = few task-space Fourier modes (exactly
controllable, interpretable, embodiment-invariant, transfers) vs realization = residual
(body-specific dynamics, needs generation/adaptation, where flow/adapter earns its keep).**
HONEST CONCLUSION: "steering by constructing source noise" has DEFLATED to Fourier-mode
prediction in this regime; the source-noise machinery adds nothing a regressor lacks. The
real, defensible content is the STRUCTURE/REALIZATION decomposition + which modes are
shared vs body-specific — NOT the source-noise mechanism. Two forks offered Denis: (a) test
at real-VLA/pi0 scale whether the flow residual is actually irreplaceable (the only place
source-noise steering could still be non-trivial); (b) pivot the framing to the
structure/realization decomposition and drop the source-noise-is-special narrative.

**TOY IS STRUCTURALLY INADEQUATE — multimodality is 1-D & Fourier-isolatable (2026-07-21,
`steer_multimodal.py`, `results/multimodal.json`)**: Denis asked (i) does predicting a
Fourier mode bypass the modality gap (mode-averaging) diffusion solves, and (ii) if a
deliberately-bad model still succeeds on the toy, the toy is the problem. Built a CENTERED-
obstacle bench (obstacle ON the line, lateral=0 => BOTH detour sides equally valid =>
irreducible bimodality given obs; data mean|bin1|=3.0, 50/50 sides, ceiling 1.0). Results:
REGRESS (MSE, no flow) success 0.075 — averaged the bimodal bin-1 (3.02->0.74) -> straight
into obstacle (mode-averaging pathology CONFIRMED; FFT is linear so Fourier regression does
NOT bypass the gap). BUT REG_ASSIGN (regress + OVERWRITE the one multimodal coefficient with
a definite side, still NO flow) success 0.637 — beat plain FLOW (0.22, does sample both
sides 0.47) and FLOW_PIN (0.34, likely undertrained). CONCLUSION: (Q1) Fourier-mode
prediction bypasses the modality gap ONLY via external mode ASSIGNMENT (a command/prior),
not by regression; assignment works because the multimodality is confined to identifiable
modes. (Q2/THE PROBLEM) the toy's multimodality is ONE-DIMENSIONAL (just the side,
isolatable to lateral bin-1), so a flow-free assign-one-mode model always suffices and FLOW
IS NEVER REQUIRED — the toy CANNOT justify flow/source-noise over "assign structure modes,"
no matter the tuning. Flow is only necessary for HIGH-D ENTANGLED multimodality (no small
known mode-set to assign), which the toy family lacks. => the entire toy program cannot
settle whether source-noise steering / flow earns its keep. Strong push to fork (a): test
at pi0/LIBERO scale (high-D entangled multimodal actions) whether regress+assign matches
the pinned flow; that is the ONLY bench that can decide it.

**LIBERO-SCALE ABLATION — deflation HOLDS at pi0 scale (2026-07-21, `scripts/serve_mean.py`,
`scripts/eval_mean.sh`, `experiments/phase1/results/evals/ABL_*.json`)**: ran the mode-
averaging test on the real pi0 checkpoints (phase1_A_s42/14999) via openpi serve + LIBERO
sim (harness: `eval_checkpoint.sh` / `serve_snmvp_policy.py`; checkpoints A/B/C x seeds
42-44 @ 14999 survived in `~/code/openpi/checkpoints/pi0_libero/`). MEAN server averages K
flow samples per infer = E[action|obs] = the MSE-regressor output. RESULT (libero_spatial,
3 trials x 10 tasks): arm A single-sample = 0.933; **MEAN k=8 = 0.933 (IDENTICAL, zero cost)**.
Sample-diversity probe: per-infer samples DO differ (MAD/scale ratio ~0.2, i.e. ~20% of
action scale) yet averaging is harmless => **pi0's LIBERO action distribution is UNIMODAL
WITH SPREAD, not multimodal** (averaging lands near the single mode, stays valid; if
multimodal it would collapse to invalid between-mode actions). CONCLUSION: the modality gap
that justifies diffusion/flow is NOT active for pi0 on LIBERO-spatial => a plain regressor
matches the flow => the source-noise pin has NO advantage over "regress + assign" at scale,
same as the toy. **The deflation is confirmed at real scale, not just a toy artifact.**
SCOPE CAVEATS: libero_spatial only (easy, strongly conditioned); small eval (28/30, 1 seed)
so exact tie may be coincidental but no-collapse is robust; K=8. To find any regime where
flow / source-noise steering is non-trivial would need GENUINELY MULTIMODAL action data
(ambiguous tasks, diverse human demos, harder suites libero_10/goal). IMPLICATION for the
project: source-noise steering is not special even at scale on LIBERO; the honest,
defensible contribution is the STRUCTURE/REALIZATION decomposition + interpretable control
by assigning structure modes — NOT the source-noise mechanism or flow. Forks: (1) test a
harder/multimodal suite to see if flow EVER earns its keep (would give source-noise a
regime); (2) accept the deflation, pivot framing.

**CONFOUND CHECKED, DEFLATION CONFIRMED open-loop (2026-07-21)**: Denis pushed back ("LIBERO
is where VLAs differentiate"). Re-examined: (a) VLAs differentiate on LIBERO via
pretraining/VLM/language/scale, NOT via the flow ACTION HEAD's multimodality (regression
heads are known-competitive on LIBERO) — my ablation only tested the head. (b) Real confound
in the closed-loop MEAN test: replan_steps=5 executes only the first few of each 50-step
chunk; the sample spread PEAKS LATE (peak_step ~34-49/50, from serve_mean diversity logs),
so closed-loop discards the branching before executing it. Ran the fix — OPEN-LOOP
(replan_steps=50, full-chunk execution) A vs MEAN on libero_spatial (3x10): A(k1)=0.567,
MEAN(k8)=0.800. **Averaging IMPROVES open-loop (0.57->0.80), does NOT collapse** => the
distribution is UNIMODAL WITH SAMPLING NOISE (averaging denoises; if multimodal, averaging
distinct modes would tank success). The earlier k-means bimodality ~1.7 was a unimodal-blob
artifact (2-means splits any Gaussian at ~that ratio); occasional 3-5 spikes are rare. So a
regressor (=the mean) MATCHES OR BEATS single-sample flow => flow head's stochasticity is
noise not useful multimodality on LIBERO-spatial => **deflation robust closed- AND open-loop.**
CAVEAT keeping the thesis alive: libero_spatial is strongly-conditioned/near-deterministic;
genuinely multimodal tasks (diverse human demos, libero_10 long-horizon sub-goal ordering)
could exercise the flow head. NEXT if pursued: same open-loop A-vs-MEAN on libero_10/goal —
if MEAN collapses there, flow (and source-noise steering into it) has a real regime;
if not, deflation is general and pivot to structure/realization + mode-assignment framing.
Note: reasoning flip-flopped mid-investigation (deflation -> maybe-multimodal -> deflation);
the open-loop averaging-helps test was decisive. All ABL evals in
`experiments/phase1/results/evals/ABL_*.json`; scripts `serve_mean.py`, `eval_mean.sh`.

**GENERATIVE-HEAD AUDIT — deflation GENERALIZES across LIBERO (2026-07-21)**: Denis asked to
recall original goals + pursue interesting aligned ideas. Reframed the project honestly:
we drifted into validating (and deflating) the source-noise MECHANISM; the ALIGNED live
content is (1) the STRUCTURE/REALIZATION factorization = the original "learned geometric
frame" (transfers x-embodiment, interpretable control by mode-assignment, data-efficient —
all shown), and (2) a contrarian empirical AUDIT: does the flow/diffusion ACTION HEAD
actually do multimodal work? Audit table (phase1_A_s42/14999, 3 trials x 10 tasks, A=single
flow sample vs MEAN=avg-8-samples ~= conditional mean ~= regression target): spatial
closed-loop(replan5) A=0.933 MEAN=0.933; spatial open-loop(replan50) A=0.567 MEAN=0.800;
libero_10 open-loop A=0.267 MEAN=0.567. **MEAN >= A EVERYWHERE — averaging never collapses,
and HELPS open-loop (denoises).** => across LIBERO (spatial + long-horizon), the flow head's
stochasticity is NOISE not useful multimodality; the conditional mean (regression) matches
closed-loop and BEATS single-sample open-loop => the generative head is not earning its keep
on these benchmarks. (Multimodality that exists is late-chunk, peak_step ~34-49/50, and
replanning-masked.) This is a clean, publishable, contrarian finding AND aligns with the
original goal's spirit (understand what actions ARE: on these benchmarks action uncertainty
is low-dim & largely unimodal, which is WHY the geometric-frame/structure view works and WHY
the generative machinery is often superfluous). CAVEAT: one checkpoint/seed, 3 trials, pi0
specifically; a trained-deterministic head (not sample-averaging) would nail the claim.
TWO ALIGNED FORKS proposed to Denis: (A) solidify the audit into a "when does the diffusion
action head matter?" study (cheap: more suites object/goal, closed-loop l10, a trained MSE-
head baseline, sample-count sweep) — reframes the project as an honest, contrarian empirical
contribution; (B) test the ORIGINAL goal (b) mechanism-agnostically — does explicit geometric-
STRUCTURE grounding (as auxiliary supervision, NOT hard-command which paid the Phase-1 40pt
obedience tax) improve generalization/sample-efficiency — requires training runs (bigger).
Closed-loop libero_10 A-vs-MEAN launched to complete the audit 2x2.

**ORIGINAL-GOAL TEST — discovered structure IS useful when the policy is the bottleneck,
CONFIRMED at real (robosuite) scale (2026-07-22, `experiments/rung2/collect_obstacle.py`,
`structure_test.py`, `structure_result.json`)**: Denis pushed to test the ONE open regime
(policy-bottlenecked + real/scaled + discovered structure). Chose robosuite over pi0-low-data
because pi0's pretraining is too strong to ever bottleneck the policy (that's WHY LIBERO was
null). Built OBSTACLE-reach on real Panda: OSC-track a planned detour path around a virtual
obstacle (endpoint dwell so the controller settles; over-clear by obr+8.5cm to absorb
tracking lag). Demo ceiling 0.82 (hard-but-solvable; genuinely bimodal side choice). Test
(single embodiment, 120 scenes x 8 demos, discover coherence frame OVER DEMOS via the
demos-as-bodies trick, held-out 20 scenes, obstacle-reach success = reach+clear, 3 seeds):
**F (coherence structure pin + prior) = 0.383, A (scratch flow) = 0.021, Frand (random
frame) = 0.008.** F >> A (+0.36, ~18x) and F >> Frand (learned structure specifically).
Scratch essentially FAILS the hard detour (2%) => the policy IS the bottleneck; discovered
structure rescues it to 38% (below ceiling 0.82 => prior-limited, same T-oracle>>T pattern
as LIBERO/toy). Discovered frame = 3 pins (progress-omega0+mag = endpoint, lateral-omega1/2
modpi = the bend). This is the scaled replication of the toy +17pt, AMPLIFIED, on real arm
kinematics. **VERDICT on original goal (b): finding structure in actions IS useful — when
the policy is the bottleneck — and it holds at real scale, not just toy.** Completes the
conditional answer on BOTH sides: structure helps when policy-bottlenecked (toy +17pt;
robosuite obstacle +36pt/18x; x-embodiment few-shot) and is useless/harmful when
perception/data already solve it (full-data LIBERO -40pt; pi0 never bottlenecked). Caveat:
scratch A~0.02 is a very weak baseline (hard task + small autograd flow + finite data),
which inflates the RELATIVE win; the absolute F=0.38 and F>>Frand are the solid claims.
Offline eval (real-kinematics demos, geometric success), not closed-loop robosuite execution.

**REPLICATION on a HARDER task — two-obstacle SLALOM (2026-07-22, `experiments/rung2/
collect_slalom.py`, `structure_test_slalom.py`, `slalom_result.json`, `SLALOM_STRUCTURE_
FINDING.md`)**: Denis asked to try the methodology on a more complicated task. Built a
two-obstacle slalom (obstacles on OPPOSITE sides of the reach line => a single bend clears
at most one, demo must S-weave; global weave orientation bimodal per scene). Same pipeline
(demos-as-bodies coherence discovery, F/A/Frand, 3 seeds, 100 train/20 held-out). Collection
tuning that mattered: smoothstep longitudinal profile (arrive at ~0 velocity kills dwell
overshoot); EXCLUDE the rear reach cone (|ang|<150deg) + cap radius 0.24 — the ~27% endpoint
failures were a WORKSPACE limit (rearward planar reach is short), NOT the weave; separate
obstacles longitudinally + narrow bumps (w=0.08) to cut opposite-bump cross-talk that pinned
the 2nd-disk clearance. Demo ceiling 0.596 (< single-obstacle 0.82: two clearances + weave).
RESULT: **F=0.294, A=0.021, Frand=0.035** (F-A=+0.273, F-Frand=+0.259). F/ceiling=0.49 ==
single-obstacle ratio 0.47 => structure recovers the SAME fraction of achievable performance
on a harder task. KEY: the discovered frame GREW from 3 pins to **5 pins**, lateral energy
shifted from omega1-dominated to **omega2-dominated (efrac 0.42)** — the second-harmonic
signature of the S-weave — all phase-coherent (gamma~1.0): [prog-w0+mag=endpoint, prog-w1+mag=
smoothstep long. profile, lat-w1+mag=primary bend, lat-w2+mag=S-weave, lat-w3=3rd harmonic].
Coherence discovers a genuinely HIGHER-dim subspace when the task demands it, and F>>A still
holds. NOTE vs the earlier point-robot complexity-crossover (coherence pin-count DEGRADED
5->3->3, collapsed to scratch as OAT won): no conflict — that was coherence-vs-OAT for
X-EMBODIMENT with small/x-separated obstacles spreading FFT energy; this is structure-vs-scratch
(goal b) single-embodiment with clean opposite-side offsets that CONCENTRATE energy at a
coherent omega2. Coherence works when the shared structure is genuinely low-D + phase-coherent;
the slalom is designed to be. Same caveats (weak scratch baseline; offline geometric success).

**LEARNED-BASIS generalization of the factoring — NEGATIVE, direction RETIRED (2026-07-22,
`experiments/toy_embodiment/learned_basis_toy.py`, `basis_lab.py`, `*_result.json`,
`basis_lab_*.json`)**: Denis found the factoring idea (source noise = pass-through pinned
subspace (+) conditional-completion complement) compelling and asked to expand it to higher-D
via a LEARNED orthonormal basis. Key theory (retro-explains why the fixed Fourier pin works):
pass-through holds for ANY orthonormal transform T — orthonormal T preserves both the linear
flow path AND the isotropic Gaussian source, so clamping a T-subspace gives zero velocity there
and the output carries the command exactly. Fourier is one such T; the OAT pin FAILED earlier
because a NONLINEAR bottleneck breaks pass-through (latent not Gaussian, not linearly tied to
output). Fix = constrain the learned transform ORTHONORMAL. Built a controlled comparison where
ONLY the basis U differs (same general-subspace projection pin c=U^T a, same prior, same flow
executor): A scratch / F Fourier top-k by coherence objective (between-scene / within-scene
variance ratio) / L unrestricted top-k generalized eigvecs of (Sb,Sw) orthonormalized / R random
orthonormal. RESULT across every properly-trained + properly-bottlenecked test: **L NEVER beats
F.** (1) Smooth multi-obstacle n_obst=1/2/3 (well-calibrated, pinned near ceiling): L~=F both
>>A (e.g. n3 F=0.617 L=0.60 A=0.20); basis matters only vs random at n3 (R=0.467). (2) reach3d
(Fourier extended to a 3rd action channel z, sphere over/around detour, bottlenecked): F=0.592 >
L=0.50 > A=0.442 > R=0.392 (ceiling 0.963) — F wins, and this confirms the FOURIER factoring
SCALES to a new axis (prior-limited below ceiling, same pattern as robosuite). (3) waypoint2d
(localized diagonal cross-channel atoms, built to be Fourier's worst case): fully trained all
~=ceiling (A=0.96) because obs directly encodes waypoints => NOT policy-bottlenecked => pin/basis
irrelevant; the L>F seen at a 600-iter smoke was an UNDERTRAINING ARTIFACT (pin gives head start,
scratch erases it by convergence). PRINCIPLED CONCLUSION: the pin helps only when structure is
SMOOTH and policy is BOTTLENECKED, and for smooth structure Fourier modes already span the
coherent subspace (Fourier ~= eigenbasis of smooth-trajectory covariance), so a learned basis
adds nothing. The regime that would favor learned (bottlenecked AND non-Fourier AND low-rank)
did not naturally occur: non-Fourier/localized structure was either easy for the MLP (not
bottlenecked) or high-rank (no low-k basis helps). => KEEP FOURIER; for higher-D (6-DOF/drone)
add channels, don't learn the basis. Also reconciled the earlier "Fourier degrades 5->3->3":
that was the adaptive energy-floor GATING dropping pins, NOT the basis — fixed-k Fourier does
not collapse (n3 F=0.617 >> A). Retired the learned-basis idea with evidence rather than
engineering a task to force a win.

**RUNG 3 — Fourier factoring at REAL 6-DOF + LAPLACIAN bases (2026-07-23, `experiments/rung3/`:
`collect_pose6d.py`, `structure_test_pose6d.py`, `pose6d_diag.py`, `laplacian_basis.py`,
`structure_test_pose6d_bases.py`, `glap_sweep.py`, `*_result.json`)**: Denis: scale the Fourier
factoring to a real higher-DOF embodiment in a typical benchmark; if it works, test x-embodiment
transfer of the pinned subspace. Built 6-DOF pose-reach-around-obstacle on real Panda (OSC_POSE,
action = dpos3+daxisangle3; servo6d_test verified pose servo exact: pos 0.9mm, ori 0.7mrad). Chunk
= achieved 6-ch pose-delta (H=32, C=6, D=192): [dpos_canonical(3), dori_world(3)]. Demo ceiling
0.861. Machinery = the C-channel `basis_lab` pin (Fourier top-k by coherence variance-ratio Sb/Sw,
projection pin c=Uᵀa, prior, flow executor) — NOT the 2-ch flow_embod coherence used on planar.
KEY RESULTS: (1) **Pass-through MECHANISM SCALES to 6-DOF**: pin_bind_err ~0.01-0.04 vs |command|
~13 (~0.1-0.3%) — clamping binds exactly at D=192. Strong positive. (2) **The pose task is NOT
strongly policy-bottlenecked** (pose6d_diag capacity sweep: A rises 0.10/0.75/0.95 as HID 128/256/384;
at high capacity A=0.95 and the pin HURTS, F_oracle=0.55<A — obs fully specifies detour dir+ori, so
big MLP solves it alone). Same bottleneck-conditional law: pin helps iff policy bottlenecked. So this
task can't show a large clean pin win the way planar obstacle did (A~0.02, F~0.38). (3) **LAPLACIAN /
basis-family: cross-channel coupling is what matters at 6-DOF, modestly.** Added path-graph Laplacian
(DCT-II, per-channel, free endpoints) and (time x channel) GRID Laplacian (channel edges weight w =
cross-channel coupling) bases — both orthonormal so pass-through holds. In a data-bottleneck (N_TRAIN=30,
HID=256): per-channel bases (Fourier, DCT, random) ALL sit at scratch (~0.44-0.48, A=0.44); only the
cross-channel GRID Laplacian beats scratch. DECISIVE control = coupling-weight sweep (4 seeds, N_HELD=50):
w=0 (no coupling ~ per-channel) = 0.32 at/below A=0.365 (3/4 seeds negative); w=0.5 = 0.435 prior /
0.495 oracle (+0.07/+0.13 over A, all 4 seeds positive for oracle). => cross-channel coupling is the
mechanism (6-DOF has pos-ori coordination per-channel bases can't compactly represent; planar tasks
lacked this). CAVEATS: magnitude MODEST + noisy (initial N_HELD=30 screen over-estimated GLAP=0.60;
tighter eval ~0.43; w-curve non-monotonic, w=1.0 dips); oracle >> prior throughout (prior-limited).
Clean coherence signal throughout: structured bases (F/DCT/GLAP) prior-err ~0.14-0.16 vs random 0.41-0.43.
NET: mechanism scales to 6-DOF; basis family finally mattered (cross-channel > per-channel) but the pose
task is too weakly bottlenecked for a large effect. Decision pending before x-embodiment: strengthen the
6-DOF bottleneck (hide detour dir / bimodal) OR run x-embodiment transfer on the STRONGLY-bottlenecked
planar tasks where the pin effect is large+clean. NOTE: box nohup/disown survives SSH-task kills (the
harness killed my watcher wrappers twice; box python kept running — always re-check pgrep + log).

**RUNG 3b — STRENGTHENED 6-DOF bottleneck + CAUSAL two-sided coupling proof (2026-07-23,
`collect_pose6d_hard.py`, `structure_test_pose6d_hard.py`, `pose6d_hard_{c1,c0}_result.json`)**:
Denis chose to strengthen the bottleneck. Built coupled 6-DOF task: hand must BANK into the detour
(target orientation = rotation about reach axis, sign=detour side, magnitude scales with detour amp;
COUPLE knob). Position(ch0-2) and orientation(ch3-5) mechanically COORDINATED. Orientation NOT in obs
but DETERMINED by observable offset (so prior can predict; bottleneck from low data + hardness of
learning the coupling, not hidden info). Two datasets: c1 COUPLE=1.0 (ceiling 0.787), c0 COUPLE=0.0
decoupled control (ceiling 0.979). Data-bottleneck N_TRAIN=25, HID=256, 4 seeds. Bank/dpos/dori all
CANONICALIZED (rotate by Rz(-phi)) so structure is azimuth-invariant. DECISIVE TWO-SIDED RESULT:
c1 (coupled): GLAP(grid-Laplacian,cross-channel)=0.70 >> R(random,cross-channel)=0.525 > F(Fourier,
per-channel)=0.365 ~ DCT(per-channel)=0.355 ~ A(scratch)=0.315; GLAP top in ALL 4 seeds (0.74/0.72/
0.62/0.72, low var), GLAP_oracle=0.78~ceiling. c0 (decoupled): F=0.935 ~ DCT=0.915 ~ GLAP=0.91 >
R=0.875 > A=0.715 — per-channel Fourier now BEST, GLAP NO advantage, R (needless mixing) now WORST
pinned. INTERPRETATION (clean causal): my Fourier/DCT bases are strictly PER-CHANNEL (each vec in one
of 6 channels); random & grid-Laplacian MIX channels. Ordering "structured-cross-channel > random-
cross-channel > per-channel" holds iff task couples channels; flips off without coupling. => the pin
helps a bottlenecked 6-DOF policy ONLY via a cross-channel basis, and ONLY when the task couples pose
channels. KEY consequence for x-embodiment (Denis raised): the cross-channel COUPLING = body-specific
REALIZATION, NOT transferable task-space structure. So GLAP is a WITHIN-embodiment sample-efficiency
result, not the transfer mechanism. Transfer pin should be the task-space PATH (position trajectory,
canonical frame), complement/coupling relearned per body.

**CORE-IDEA REFRAME + LEARN-U-FOR-TRANSFER direction (2026-07-23)**: Denis refocused on fundamentals:
the interesting core = denoising factors into (bit1) an orthonormal basis U + a low-dim pinned
"instruction" subspace c=Uᵀa (pass-through=identity, interpretable, steerable, transferable) and
(bit2) the generative completion of the complement (body/context realization). Orthonormality is
load-bearing: preserves linear flow path + isotropic Gaussian source => pin passes through as identity
(nonlinear OAT broke this). Goals ranked: steerability > interpretability > x-embodiment bonus; north
star = 1-shot drone-through-gate. Assessment given to Denis: recent basis-family work refined bit1's
internals (largely CLOSED: Fourier suffices for smooth; cross-channel matters within-body but is
realization) but did NOT advance the 3 actual goals. Open high-value questions: Q2 steerability+interp
interface demo; Q3 (highest) does (U,c) transfer across embodiments while bit2 relearned. Denis' new
ask: LEARN the best U (not fixed Fourier) with the hope it transfers across embodiments incl. VERY
DIFFERENT WORKSPACES. Correct objective (differs from the earlier within-body learned-basis NEGATIVE!):
treat BODY as the nuisance. U = top-k generalized eigvecs of (Sigma_scene, Sigma_body), orthonormalized:
Sigma_scene = cov of scene-mean c across scenes (instruction signal, want high); Sigma_body = mean over
scenes of across-body cov at fixed scene (body variation, want low). This searches ALL orthonormal bases
for body-invariance (not just Fourier bins as flow_embod.freeze_frame does) AND auto-assigns body-specific
coupling to the complement (resolves the coupling/transfer tension by construction). Requires canonical
scale-normalization for different workspaces (transfer normalized PATH SHAPE, complement re-injects metric).
RISKS: smooth paths may be low-freq regardless of body => learned U ~ Fourier again; if bodies solve too
differently the invariant subspace may be tiny (gen-eigvals ~1 = no separation) — a real finding. Denis
chose TOY-FIRST testbed (toy_embodiment bodies, CPU): learn U by cross-body objective, freeze on set-A,
transfer to held-out body (prior commands invariant, relearn complement), compare transfer success vs
Fourier/coherence frame vs random-U + diagnose c-invariance across bodies. If learned-U beats Fourier for
TRANSFER, escalate to robosuite multi-arm (Panda/Sawyer/IIWA/Jaco, different workspaces) then drone.

**LEARN-U-FOR-TRANSFER RESULT — NEGATIVE for learning, but per-channel-Fourier is fragile under
embodiment shift (2026-07-23, `experiments/toy_embodiment/learnu_transfer.py`, `learnu_transfer_result.json`)**:
Set-A = arm family {arm2,arm3,arm4,arm5} (reach 1.8-2.5); U learned by (Sigma_scene,Sigma_body) gen-eig
(top gen-eig ~276 => strong invariant subspace exists). Frozen U -> set-A prior -> transfer to held-out
bodies, relearn only the flow executor on the held-out body. Bug found+fixed en route: canonical->world
rotation used angles[:,None] (4D convention) on a 3D generated chunk -> mis-broadcast -> ALL success 0.0;
fix = 1D angles (scratch-on-point went 0.0 -> 0.47). POOLED TRANSFER SUCCESS (3 seeds), held-out body ->
[scratch / Fourier / Urand / Ulearn / Ulearn-oracle]: point (unconstrained=drone analog) 0.55/0.52/0.78/
0.77/0.81; point_drag (inertial) 0.51/0.51/0.76/0.77/0.76; arm_short (in-family arm) 0.28/0.54/0.46/0.46/0.44.
c-invariance (lower=more body-invariant): point Ulearn 0.07 < F 0.098 < Urand 0.156. THREE robust
(all-3-seed) findings: (1) **Learning U did NOT beat RANDOM orthonormal on transfer success** (Ulearn~=Urand
everywhere) — the invariance objective works in COEFFICIENT space (Ulearn most body-invariant) but that does
not convert to success advantage. (2) **No universal basis; split by body similarity**: for bodies very
DIFFERENT from training arms (point/point_drag), any channel-MIXING basis (learned OR random) transfers
(~0.77) while PER-CHANNEL FOURIER FAILS (=scratch ~0.52); for the IN-FAMILY arm (arm_short) it flips —
Fourier best (0.54), mixing bases worse (0.46). (3) **Pin+prior is what transfers, not the basis**: oracle
~= prior (0.81 vs 0.77 on point) => set-A prior already predicts the invariant for held-out bodies; transfer
mechanism sound, basis identity secondary/regime-dependent. UNIFYING THREAD with Rung-3b: per-channel
Fourier is FRAGILE under distribution shift (task channel-coupling OR large embodiment change); a channel-
mixing orthonormal basis is more robust; the specific mixing need NOT be learned (random suffices). NET:
"learn the best U and hope it transfers" NOT supported; leverage is in the pin+prior + using a mixing (not
per-channel) basis for cross-workspace transfer. Consistent with the standing result that the basis is not
where the leverage is.

**STEERABILITY + INTERPRETABILITY DEMONSTRATED AT 6-DOF (2026-07-23, `experiments/rung3/steer6d.py`,
`steer6d_result.json`)**: Denis pushed on generality ("if it doesn't generalize to higher DOF it isn't
good enough"). Reframe given to him: the MECHANISM already generalized to 6-DOF (GLAP pin 0.70 vs scratch
0.315); what didn't generalize is per-channel Fourier + the VALUE of learning the basis. The 3 GOALS are
what must generalize, and steerability+interpretability are UNCONDITIONAL (don't need the bottleneck) —
the most general claim, never yet shown end-to-end. So instead of more basis optimization, DEMONSTRATED
the goals directly at 6-DOF on the coupled data (Panda_c1) with a FIXED grid-Laplacian pin (K=10). Also
argued grid-Laplacian is the DOF-AGNOSTIC general basis (defined by a graph over time x DOF => specified
for any embodiment's channel count without relearning; beats "learn U"). RESULTS: (1) INTERPRETABILITY —
pinned coordinates correlate ~0.97 with detour side and ~1.0 with bank angle (coupled in this task), i.e.
the pinned subspace direction IS the named coupled task quantity. (2) STEERABILITY — the steering handle
is a DIRECTION in the pinned subspace (side encoded redundantly across all 10 coords; sweeping ONE coord
fails because the pin fights itself — must move along the empirical side-direction d_side = mean-c(side>0)
- mean-c(side<0)). Sweeping the command along d_side on a FIXED scene: pass-through slope realized-vs-
commanded = 1.001 (identity holds at D=192); the 6-DOF behavior steers monotonically THROUGH A SIGN FLIP —
detour +0.095->-0.096 and coupled bank +0.539->-0.537, both flipping together at the same command. So a
single interpretable handle steers the full coupled 6-DOF behavior (position detour + orientation bank) as
a unit, by pass-through, unconditionally. First direct end-to-end evidence of the primary goals at high DOF.
NEXT (per Denis' generality concern): 6-DOF cross-embodiment transfer of the task-space subspace (freeze on
one body, relearn complement on a very different one) is the follow-on.

**6-DOF CROSS-EMBODIMENT TRANSFER — POSITIVE, task-space subspace transfers across real arms (2026-07-24,
`experiments/rung3/transfer6d.py`, `collect_pose6d_hard.py` on multiple arms, `transfer6d_result.json`)**:
Collected the coupled 6-DOF pose task on 4 robosuite arms (scenes seeded identically => paired scenes across
arms; demo ceilings Panda 0.787, IIWA 0.654, UR5e 0.810, Jaco 0.738; Sawyer 0.38 excluded as too low).
Set-A = {Panda,IIWA,UR5e}; held-out B = Jaco. Grid-Laplacian subspace (K=12) selected by CROSS-ARM
coherence (Sigma_scene/Sigma_body over set-A arms; top_gen_eig=745894 => strong arm-invariant subspace),
frozen; set-A scene->coeff prior; on Jaco relearn ONLY the flow executor. POOLED (3 seeds) held-out success:
scratch 0.225, Fourier(per-channel) 0.383, random(mixing) 0.30, **GLAP(grid-Laplacian) 0.75**, GLAP-oracle
0.725; Jaco own ceiling 0.738. Cross-arm c-invariance GLAP 0.075 < F 0.096 < Rand 0.133. FINDINGS: (1)
task-space instruction subspace TRANSFERS across real 6-DOF embodiment change — freeze on 3 arms + relearn
complement on Jaco lifts scratch 0.225 -> 0.75 = Jaco's OWN demo ceiling 0.738 (near-complete recovery). (2)
grid-Laplacian is the effective transfer basis: 0.75 vs Fourier 0.383 vs random 0.30 — UNLIKE the toy transfer
(where learned~=random>Fourier), here structured cross-channel > random, and c-invariance ordering PREDICTS
success ordering. Plausible mechanism: the coupled 6-DOF task's structure aligns with the grid-Laplacian's
cross-channel modes, which random mixing does not capture and per-channel Fourier captures only partially;
the toy (2-ch planar, weak coupling) lacked this so the GLAP-vs-random distinction washed out. (3) prior
generalizes across bodies: oracle 0.725 ~= prior 0.75, so the set-A scene->coeff prior predicts the held-out
arm's coefficient without per-arm tuning. NET: all THREE goals now have direct 6-DOF evidence — steerability +
interpretability (steer6d) and cross-embodiment transfer (this). Caveat: offline geometric success on real-
kinematics achieved chunks (not closed-loop robosuite execution); scratch 0.225 is a low-data-bottlenecked
baseline. INFRA NOTE: /tmp gets cleaned + box SSH is flaky (connection timeouts killed nohup children twice);
launch long box jobs with `setsid bash script < /dev/null &` writing logs to the PERSISTENT experiment dir,
and verify the log grows before trusting the launch. Published method-explainer artifact (flow interpolation,
orthonormal invariances, Fourier/Laplacian from first principles, pass-through derivation, Sigma_scene/Sigma_body
objective): https://claude.ai/code/artifact/2cbb1354-4674-4118-8a28-7c3df18adbbf

**CONSOLIDATION DOC + CLOSED-LOOP EXECUTION (2026-07-25)**: (1) Wrote a single reproducible arc document
`experiments/FACTORING_ARC.md` (construction, 8-entry experiment ledger with verified numbers + file
pointers, synthesis, env/repro incl. robosuite pins + setsid/persistent-log operational notes). All numbers
re-verified against saved result JSONs. (2) CLOSED-LOOP EXECUTION retires the standing offline-geometric
caveat. Two-stage: `rung3/gen_cle.py` (autograd env) trains scratch + grid-Laplacian policies on coupled
6-DOF Panda_c1, generates 40 held-out chunks, saves `cle_chunks.npz`; `rung3/exec_cle.py` (robosuite env)
reconstructs absolute pose waypoints from each generated canonical pose-delta chunk (reach azimuth 0 so
canonical=world; integrate dpos to positions, compose axisangle2quat(dori) for orientation), OSC-tracks them
on the real Panda (KP_POS=12,KP_ROT=5, +6 settle steps), measures achieved reach+ori+clear. RESULT
(`cle_result.json`): closed-loop scratch 0.175 / GLAP 0.675 vs offline scratch 0.150 / GLAP 0.700; per-scene
agreement 0.925 for BOTH. => the grid-Laplacian pin's advantage holds under real simulator dynamics (same
~0.5 gap), and the offline geometric metric agreed with closed-loop on 37/40 scenes, so it was a faithful
proxy. Offline-geometric caveat that applied since the planar rung-2 work is now addressed at 6-DOF.

**CLOSED-LOOP EXECUTION ON THE TRANSFER ARMS (2026-07-25, `rung3/gen_cle_transfer.py`, `exec_cle.py`
parameterized by SNMVP_ARM/NPZ/OUT, `cle_result_{Jaco,UR5e}.json`)**: extended CLE to the cross-embodiment
transfer. gen_cle_transfer freezes grid-Laplacian U + set-A prior, relearns only the executor on the held-out
arm, generates held-out chunks; exec_cle executes them in robosuite ON THAT ARM (robots=<HELD>). Two held-out
configs: Jaco (set-A Panda,IIWA,UR5e) and UR5e (set-A Panda,IIWA,Jaco). RESULTS [closed-loop scratch / GLAP-
transfer | offline scratch / GLAP | agreement scratch / GLAP]: Jaco 0.150/0.575 | 0.175/0.675 | 0.925/0.750;
UR5e 0.325/0.575 | 0.275/0.750 | 0.850/0.825. FINDINGS: (1) the transferred grid-Laplacian advantage HOLDS
under real dynamics on both held-out arms (Jaco +0.425, UR5e +0.25 over scratch closed-loop) — transfer
benefit is not an offline-metric artifact. (2) Offline OVERSTATES the transferred policy under execution by a
modest amount (GLAP closed-loop ~0.10-0.18 below offline; agreement 0.75-0.825), a LARGER execution gap than
the single-arm native case (0.925 agreement, near-equal), plausibly because the transferred policy relearned
only the complement so re-executing its trajectory on the held-out arm's dynamics adds tracking mismatch and
some geometrically-passing trajectories fail under execution. NET: both requested tasks done — consolidation
doc `experiments/FACTORING_ARC.md` + closed-loop confirmation of the 6-DOF single-arm AND cross-embodiment
transfer results.

**VARIABLE-DOF TRANSFER — NEGATIVE (2026-07-26, `rung3/collect_pos3.py`, `vardof_transfer.py`,
`run_vardof_pipeline.sh`, `vardof_*_result.json`)**: Denis asked whether the method generalizes to
embodiment changes with a VARIABLE ACTION-DIMENSION (drone north star). Design: pin the instruction in a
SHARED 3-ch end-effector POSITION space, embed it into each embodiment's full action space (position
channels of a 6-ch pose action, or the whole of a 3-ch position action), so one instruction coordinate
applies regardless of action dim. Collected 3-ch position detour-reach with OSC_POSITION (action_dim=4) on
Panda (ceiling 1.0) + UR5e (0.765); reused 6-ch OSC_POSE pose data for the arms. Grid-Laplacian shared
position subspace frozen on set-A, transfer to held-out embodiment of DIFFERENT action dim, success =
position reach+clear. POOLED (3 seeds) [scratch / GLAP-transfer / oracle]: 6->3 Panda 1.00/0.975/0.992
(uninformative, scratch saturated); 6->3 UR5e 0.808/0.625/0.70 (pin HURTS); 3->6 Jaco 0.358/0.142/0.367
(pin HURTS, but set-A only 2 bodies => weak covariance). VERDICT: naive variable-DOF transfer does NOT
generalize — the frozen subspace LOWERS success across an action-dim change, oracle also <= scratch.
PLAUSIBLE MECHANISM (refines the 2.8 positive): the 2.8 cross-arm transfer worked because all arms used the
SAME controller (OSC_POSE) so the stored action was a genuinely shared 6-ch task-space pose; here the 3-ch
and 6-ch embodiments use DIFFERENT controllers (OSC_POSITION vs OSC_POSE), and the controller shapes the
ACHIEVED trajectory, so each produces controller-specific position structure and the frozen subspace imposes
a mismatched command. I.e. we pin the ACHIEVED trajectory, which is partly REALIZATION, not controller-
invariant task structure. IMPLICATION: to transfer across action-dim/controller changes, the shared pinned
object should be the PLANNED/DESIRED task path (controller-independent), not the achieved trajectory — the
natural next experiment. INFRA: the full pipeline (collect + 3 transfer tests) ran autonomously via
`setsid bash run_vardof_pipeline.sh </dev/null >log 2>&1 & disown` and completed through repeated SSH drops
(box connectivity very flaky this session, frequent 'No route to host'); minimal atomic launch commands are
essential (bundling pkill/rm/verify widens the drop window and the launch fails before setsid runs).

**LAPLACE / TRANSFER-FUNCTION TOY (2026-07-26, `rung3/laplace_toy.py`, `laplace_toy_result.json`)**: Denis
asked whether a Laplace transform helps anywhere in the loop. Reasoning given: discrete analog is the
z-transform; it generalizes Fourier by adding DECAY (damped-exponential) modes = the language of LTI
dynamics (transfer function H(z), poles). The task SHAPE is smooth => Fourier already suffices, so Laplace
does NOT help the instruction subspace; the decay structure lives in the REALIZATION (settling, controller
transient, sim-to-real gap = pole change) — exactly where the additive pin failed to transfer (variable-DOF
negative). Key idea: achieved(z) = planned(z)*H(z) (tracker = reference convolved w/ impulse response), a
MULTIPLICATIVE factoring: task=planned, realization=H(z); deconvolving H recovers the controller-invariant
planned path (the 'transfer the planned path not the achieved trajectory' fix), then re-apply target H.
Nonlinear (deconvolution) so it COMPLEMENTS, doesn't slot into, the additive pass-through pin. TOY: shared
planned bump passed through 2nd-order LTI filters (poles = w,zeta) = 'bodies'; reconstruct held-out body's
achieved as its poles leave the training range. RESULT (w=5 fixed, train zeta{1.0,1.2,1.4}, held zeta swept;
first run confounded by Euler instability at w=10/dt=0.12 — fixed w=5,dt=0.08): additive_err vs laplace_err
by held zeta: 1.0:0.001/0.008, 0.8:0.007/0.009, 0.6:0.024/0.011, 0.4:0.067/0.017. => within training range
additive is fine (slightly better, no ID noise); as the pole gap grows additive degrades ~70x (0.001->0.067)
while transfer-function factoring stays ~flat (~2x), ~4x better at the largest gap; filter ID accurate
(recovered zeta within ~0.01). CALIBRATION: idealized LTI toy where the factoring holds BY CONSTRUCTION
(achieved literally = filter(planned), known low-order family, clean ID) => demonstrates the MECHANISM, not
real-dynamics performance; on nonlinear robosuite dynamics the gain would be smaller/possibly absent.
IMPLICATION: argues for a system-ID + deconvolution step on the realization for cross-dynamics transfer
(variable-DOF/sim-to-real); decisive test is on the robosuite sim-to-real data (pipeline still running).

**SIM-TO-REAL TRANSFER — POSITIVE, confirms Denis' hypothesis (2026-07-26, `rung3/collect_dyn.py`,
`simreal_transfer.py`, `run_simreal_pipeline.sh`, `simreal_result.json`)**: Denis' hypothesis: sim-to-real
is cross-embodiment with a FIXED action interface (same controller/DOF) and only the DYNAMICS changing, so
it sits in the Section-2.8 regime that transferred, NOT the variable-DOF regime that failed. Test: coupled
6-DOF Panda task under OSC_POSE (6-ch, fixed) with dynamics set by controller gain/damping/latency. 3 sim
variants (sim1 kp150/d1.0/l0 ceiling 0.771, sim2 kp130/d1.1/l0 0.676, sim3 kp200/d0.85/l1 0.573) + held-out
'real' (kp250/d0.75/l0, ceiling 0.789, outside the sim gain/damping range, verified feasible; severe gaps
like kp300/d0.6/l2 gave ceiling 0.0 = infeasible, so moderate feasible gap used). Grid-Laplacian subspace
frozen on the 3 sim variants, prior on sim invariant, relearn ONLY the executor on the held-out variant over
N held-out scenes. RESULT [scratch / GLAP-transfer / oracle] by N: n10 0.283/0.692/0.750, n25 0.350/0.742/
0.758, n50 0.308/0.700/0.758. FINDINGS: (1) instruction subspace TRANSFERS across the dynamics gap (GLAP
~0.7 vs scratch ~0.3, near the real ceiling 0.789) — same regime as 2.8, opposite of variable-DOF. (2)
DATA-EFFICIENT: GLAP already ~0.69 at N=10 and flat to N=50, while scratch stays ~0.3 (bottlenecked at all N)
=> freeze instruction in sim, relearn realization on scarce real data. (3) sim prior predicts the real
coordinate (oracle ~= prior, +0.02-0.06). CAVEATS: offline geometric (closed-loop slightly lower for a
transferred policy per the CLE result); dynamics gap via controller params not full physics randomization;
one held-out variant. NET: sim-to-real with fixed action interface is the grid-Laplacian's regime and works
with little real data; the Laplace/deconvolution factoring is the candidate fix for the harder case where the
action interface/controller changes enough to degrade the shared subspace. Next: add system-ID+deconvolution
to this sim-to-real setup and test on robosuite dynamics.

**SYSTEM-ID + DECONVOLUTION ON SIM-TO-REAL (2026-07-26, `rung3/simreal_deconv.py`,
`simreal_deconv_result.json`)**: added the deconvolution step. Recompute the DETERMINISTIC planned canonical
reference from the scene recipe; identify a per-channel FIR filter (length 6) mapping planned->achieved by
least squares from n_id held-out demos; reconstruct held-out achieved = FIR(planned); score with HD.success.
POOLED (3 seeds, 40 held-out) [n_id: deconv_success / filter recon_rel_err]: 2:0.975/0.409, 5:1.0/0.395,
10:1.0/0.391, 25:1.0/0.389. vs additive grid-Laplacian transfer 0.692(n10)/0.742(n25); real ceiling 0.789.
=> deconv reaches ~1.0 from 5 demos, above additive (0.69-0.74) and above the demo ceiling, and works from
n_id=2. BUT NOT a controlled comparison: (1) it starts from the recomputed planned reference which is itself
a valid task solution the additive method doesn't use; (2) the FIR is a ROUGH dynamics model (recon_rel_err
~0.39 => does NOT reproduce the real achieved trajectory) — success reflects that the filtered planned path
still reaches target+clears under the lenient geometric measure, which doesn't require matching the real
trajectory; (3) success > demo ceiling because the reconstruction is cleaner than the noisy demos. Consistent
with the toy (factoring out dynamics aids transfer, data-efficient) but the fair reading is it succeeds by
starting from a known planned solution + identified filter, not by modeling the achieved dynamics. Stricter
test: withhold the planned reference / require matching the achieved trajectory. BENCHMARKS (answer to Denis):
Open X-Embodiment (2023, 22 embodiments, ~1M real traj; RT-X/Octo/OpenVLA/CrossFormer) is THE standard
cross-embodiment benchmark but has VARIABLE action spaces (the hard regime) and needs hardware/checkpoints
for closed-loop — an OFFLINE structural transfer study on its trajectories is feasible on our compute and
would test the variable-DOF regime at scale. Sim+real: no single standard; nearest = DROID(2024 real Franka)/
BridgeV2(2023 WidowX) paired with RoboCasa(2024 robosuite sim)/Isaac Lab, but that's fixed-embodiment
sim-to-real (our covered regime). Controlled closed-loop sim we can run w/o hardware: robosuite multi-arm
(used), ManiSkill2. We have NO physical robots => real closed-loop x-embodiment not available; feasible =
controlled sim (robosuite/ManiSkill) + offline OXE analysis.

**OPEN X-EMBODIMENT OFFLINE SUBSPACE-TRANSFER STUDY (2026-07-27, `rung3/oxe_extract.py`, `oxe_transfer.py`,
`run_oxe_pipeline.sh`, `oxe_transfer_result.json`)**: Denis asked to test on OXE with compatible actions.
INFRA: GCS bucket gs://gresearch/robotics/ is PUBLIC + reachable from box; stream via `uv run --with
tensorflow-cpu --with tensorflow-datasets` + tfds.builder_from_directory(gs://.../<ds>/0.1.0), iterate
episodes, extract action = world_vector(3)+rotation_delta(3) = shared 6-D EEF-delta. NOTE: many OXE actions
are DICTs (world_vector/rotation_delta/gripper), some flat/joint-vel/8-D — pick datasets w/ world_vector+
rotation_delta. Slice `train[:N]` errors if <N episodes (viola has 135) — cap in the loop instead. OXE
actions are COMMANDED deltas (no planned-vs-achieved split) so DECONVOLUTION does not apply; and robots do
DIFFERENT TASKS so no shared scenes => can't use the Sigma_scene/Sigma_body coherence objective; study =
reconstruction error of a held-out robot's action chunks (H=16 windows, per-dataset zero-mean unit-RMS) by a
k-dim subspace fit on the other robots. Robots: UR5(berkeley_autolab_ur5,1652 chunks), WidowX(bridge,445),
Franka(toto,5722; viola,8425). RESULT (rel recon err, lower=better, k=8) [GLAP-crosschannel / per-channel /
PCA-otherrobots / PCA-heldout-oracle / random]: UR5 .904/.614/.619/.543/.959; WidowX .924/.871/.887/.625/
.954; toto .871/.038/.740/.024/.968; viola .880/.373/.457/.243/.962. FINDINGS: (1) the CROSS-CHANNEL
grid-Laplacian reconstructs POORLY (~.87-.92, near random) — the pos-ori coupling that helped the robosuite
banking task is NOT present in raw OXE action chunks, so that innovation does not transfer to this data. (2)
the PER-CHANNEL smooth basis captures much more, near the within-robot oracle where actions are smooth (toto
.038 vs oracle .024, viola .373 vs .243) but high for WidowX (.871); fixed => transfers, but coverage varies
a lot by robot. (3) data-driven PCA fit on other robots transfers POORLY (robot-specific directions;
toto .740 vs oracle .024), consistent w/ learned subspaces not transferring across embodiments. MAJOR
CAVEAT: OXE robots do DIFFERENT TASKS, so cross-robot differences CONFOUND embodiment change with task change;
the shared-task transfer premise is not met => this is a structural probe, not a controlled embodiment-
transfer test. Clean test needs the SAME task on different robots (OXE doesn't isolate this). NET: on real
cross-embodiment data as available, the cross-channel innovation does not help, a per-channel smooth basis is
better but only partially transfers, and the result is task-confounded.

**GENERALITY ABLATION ACROSS CROSS-EMBODIMENT SITUATIONS (2026-07-27, `rung3/deconv_eval.py`,
`deconv_*.json`; consolidated in `experiments/FACTORING_ARC.md` section 5)**: Denis' concern — the shared-task
transfer premise is just ONE cross-embodiment use case; make sure the approach is GENERAL, ablate whether
deconvolution helps, before costly real-robot iteration. Ran the deconvolution ablation (system-ID FIR +
planned-reference reconstruction) across held-out bodies and compared to the additive-pin + scratch results.
ABLATION MATRIX (held-out success) [scratch / additive per-channel / additive cross-channel(GLAP) / deconv]:
same-task diff-arm Jaco6 0.225/0.383/0.750/0.14; same-arm changed-dynamics(sim2real) 0.30/-/0.700/0.975;
diff-action-dim UR5e3 0.808/-/0.625/0.758; diff-action-dim Panda3 1.00/-/0.975/0.975. KEY: NO method best
everywhere. GLAP cross-channel best for same-task cross-arm (0.75, deconv FAILS there 0.14). Deconv best for
same-arm dynamics change (0.975) — but that keeps the arm FIXED; deconv does NOT carry to a different arm
(Jaco 0.14) because it starts from a planned reference the held-out arm doesn't follow closely (its earlier
sim2real win was because 'real' was the SAME Panda w/ changed controller). Variable-DOF: nothing beats
scratch on UR5e (additive 0.625<0.808, deconv 0.758 degrades w/ more ID data), Panda3 saturated. => the
strategy is RELIABLE for two situations only: (a) transfer across arms doing the SAME task, (b) sim->real for
the SAME body w/ changed dynamics — both hold the ACTION INTERFACE FIXED (covers the common 'deploy sim
policy on the physical same robot' pattern). It is NOT a general cross-embodiment method: cross-channel GLAP
is specific to a channel-coupling task, deconv is specific to a fixed body w/ changed dynamics, and
variable-action-dim + different-task remain open. Deliverables this round: artifact updated with Section 8
(transfer-function factoring/system-ID/deconvolution, scientific-writing register); FACTORING_ARC.md Section 5
(4 situations, ablation matrix, OXE table, what-holds-what-doesn't) — cleaned to the register (no em dashes/
scoreboard verbs). Honest bottom line for Denis: works for fixed-interface regimes; not yet universal;
variable-DOF and different-task are the open problems to solve before claiming general cross-embodiment.

**PRECISION NOTE — what has vs has NOT been done at VLA/pi0 scale (2026-07-27, Denis flagged)**: only the
BARE source-noise PIN mechanism has ever run inside a real VLA — Phase 1 pinned the movement COMMAND into
pi0's flow source noise (arms A/B/C on LIBERO; established steering viability + obedience tax), plus the
flow-head multimodality audit (serve_mean/averaging deflation). The ENTIRE grid-Laplacian program —
grid-Laplacian basis, structure/realization decomposition, coherence-selected instruction subspace,
cross-embodiment transfer, complement/deconvolution — is TOY + ROBOSUITE only (autograd flow executor,
offline geometric metric), NEVER pi0/real-VLA. So "we already inserted it into a VLA" is true ONLY for the
raw pin; the current METHOD (grid-Laplacian instruction subspace + relearned realization) is untested at VLA
scale. Roadmap consequence: grid-Laplacian only helps when the policy is BOTTLENECKED, and pi0-on-standard-
LIBERO is NOT bottlenecked (why the project moved to robosuite) => "grid-Laplacian in a VLA" specifically
means pi0 pushed into a bottleneck = cross-embodiment FEW-SHOT (fork 2), not vanilla LIBERO (which would
show nothing, per the deflation). The robosuite variable-DOF rung is the last controlled step before that
VLA insertion would be meaningful.

**VARIABLE-DOF GATE RESULT — NEGATIVE, with a structural reframe (2026-07-27, `rung3/collect_vardof_hard.py`,
`collect_vardof_slalom.py`, `vardof_complement.py`, `vardof_complement_result.json`, data_vardof_hard/,
data_vardof_slalom/)**: built the fork-2 gate = the SAME position task under two controllers (set-A hard
task under OSC_POSE 6-ch on Panda/IIWA/UR5e; held-out same task under OSC_POSITION 3-ch on UR5e), to test
whether the instruction subspace transfers across a controller/action-dim change and whether a modeled
realization complement (deconvolution) rescues it where the achieved-pin breaks. Two task variants:
single-obstacle (ceilings 0.90-1.0) and a bottlenecked deterministic slalom (ceilings 0.51-0.66).
CHECKPOINTS confirmed: (a) the pose->position change DOES perturb the achieved-position pinned coordinate
(c-invariance 0.156-0.203 vs ~0.11 for any dynamics gap) = the first setting where the achieved-pin degrades
while the task stays solvable, unlike all dynamics variants; (b) latency and soft/overdamped dynamics could
NOT create a feasible broken regime (they crush the ceiling to ~0), so the achieved-pin is dynamics-robust
whenever the task is feasible. FULL FOUR-WAY STUDY (held-out pos_UR5e, ceiling 0.526, 3 seeds, n=10/25/50)
[S / ACH / PLAN / DECONV]: n10 0.542/0.467/0.0/0.075; n25 0.533/0.575/0.0/0.083; n50 0.492/0.550/0.0/0.075.
VERDICT: **NEGATIVE — the variable-DOF position task is NOT policy-bottlenecked at convergence**, so nothing
helps. CRITICAL METHODOLOGICAL CATCH: my bottleneck checkpoint used 4000 iters (scratch 0.275, looked
bottlenecked) but the full 9000-iter run has scratch 0.49-0.54 = AT/ABOVE the 0.526 ceiling from just n=10 --
the low checkpoint number was an UNDERTRAINING ARTIFACT. Lesson: the bottleneck must be verified AT
CONVERGENCE, not at low iters. ACH ties scratch (neither helps nor hurts on an unbottlenecked task); PLAN
fails outright (0.0, traj err 5-6: planned-DELTA pin clamps an off-manifold target, achieved deltas of even a
dedicated position controller diverge from planned deltas, gap 0.996); DECONV fails (0.075: FIR on position
deltas does not reconstruct a valid slalom). STRUCTURAL REFRAME (the real finding): both position tasks
(single-obstacle AND slalom) are unbottlenecked at convergence, whereas the 6-ch COUPLED task stayed
bottlenecked (scratch ~0.3) at the same iters -- the bottleneck in EVERY positive result comes from the 6-ch
position-orientation COUPLING, which a position-only task lacks. TENSION for the variable-DOF / drone goal:
the part that transfers across an action-dim change is the shared task PATH (position), which is easy to learn
and does not need the pin; the part that is bottlenecked (6-ch coupling) is NOT shared with a lower-DOF body.
=> cannot have both the bottleneck (makes the pin useful) and the action-interface change (makes it
variable-DOF) in this detour-task family. IMPLICATION: fork-2 cross-embodiment WORKS for same-INTERFACE
changes (cross-arm, sim2real -- shown) but the genuine action-dimension change (the drone north star) is not
supported by this method as constructed: the transferable instruction is the easy part. Honest options now:
(1) reframe fork-2 to same-interface cross-embodiment (deploy a fixed-interface policy on a new body/dynamics),
which is solid; (2) pursue fork-1 (steering/interpretability, unconditional) toward a VLA; (3) find a task
family where the SHARED (cross-DOF-transferable) structure is itself bottlenecked (open; not obvious it exists
for position-only actions).

**CONTROLLED TASK-BY-EMBODIMENT DECOMPOSITION (2026-07-27, `rung3/collect_task.py`, `taskembod_study.py`,
`run_taskembod_pipeline.sh`, `taskembod_result.json`)**: Denis: separate embodiment transfer from task
transfer (OXE confounds them). Grid = 3 tasks (bank: lateral detour+roll; vertical: z detour+pitch; slalom:
two-obstacle lateral S-curve, no orientation) x 3 arms (Panda/IIWA/UR5e), SAME 6-ch OSC_POSE interface, 80
scenes x 6 demos each, paired scenes per task. Metric = rel reconstruction error of chunks by a k=10
grid-Laplacian(w=0.5) coherence subspace. CROSS-EMBODIMENT within task [held-out-arm / in-sample / PCA-oracle
/ random]: bank .530/.526/.177/.965; vertical .488/.491/.131/.959; slalom .934/.919/.276/.965. CROSS-TASK
[src->tgt]: bank->{bank .526, vert .533, slal .972}; vert->{bank .539, vert .491, slal .966}; slal->{bank
.955, vert .961, slal .919}. FINDINGS: (1) EMBODIMENT-INVARIANCE HOLDS + GENERALIZES: held-out-arm err ==
in-sample err within noise for bank (.530 vs .526) and vertical (.488 vs .491) => subspace transfers across
arms with NO added error, for BOTH tasks (not just the one tested earlier). (2) SUBSPACE SHARED across detour
DIRECTIONS: bank<->vertical reconstruct each other (~.53) as well as themselves => one grid-Laplacian subspace
represents both single-detour tasks; the direction is in the COORDINATE not the basis (consistent w/ the
intended structure/coordinate split). (3) FAILS on the S-curve: slalom near random whether fit on itself
(.919) or others (.955-.972) while PCA-on-slalom = .276 => fixed low-order coordinated subspace misses
higher-harmonic structure (caveat: slalom demos noisiest, ceiling .30-.46, lowers coherence). (4) PARTIAL
COVERAGE even where transfer holds: grid-Laplacian leaves ~.50 rel err (~75% variance) for bank/vertical vs
PCA .13-.18 (~98%) => fixed transferable basis captures much less than a data-fit basis. NET: the property
enabling cross-body transfer (subspace embodiment-invariance) IS general in this controlled design (holds
across both single-detour tasks + both detour directions), but the fixed grid-Laplacian subspace is a PARTIAL
action representation — embodiment-general but not structure-complete (single-detour ~75%, S-curve fails).

**E1/E2 DIAGNOSTICS — the grid-Laplacian's incompleteness is a MODE-EFFICIENCY limit, not a transfer
limit; fix = fit the basis to training-arm variance (2026-07-27, `rung3/diag_e1e2.py`,
`diag_e1e2_result.json`)**: Denis: "generalized but doesn't represent actions completely; devise experiments
to improve it." Two diagnostics on the existing task x embodiment data (reused normalization + coherence
machinery). E1 (residual-transfer): project onto grid-Lap subspace fit on 2 in-sample arms, fit PCA on the
IN-SAMPLE arms' RESIDUALS, reconstruct the HELD-OUT arm's residual. Result [in_sample/transfer/oracle/random],
m=10: bank .286/.406/.229/.969; vertical .209/.330/.159/.973; slalom .276/.376/.214/.969. => the residual
grid-Lap misses is SHARED across arms (transfer near in_sample+oracle, far from random) for ALL THREE tasks
INCLUDING slalom => a richer transferable basis CAN capture it (Fork A, not body-specific realization).
E2 (coverage-vs-transfer knee): sweep K=2..96. Transfer GAP (held-out minus in-sample rel err) stays ~0 at
EVERY K (bank max +.017, K=96 -.006; vertical ~0 throughout) => NO KNEE: grid-Lap modes are embodiment-
invariant at ALL orders, growing the basis never starts capturing body-specifics. So the feared coverage/
transfer tradeoff DOES NOT EXIST; the real limit is MODE-EFFICIENCY (grid-Lap needs ~3-5x more modes than
PCA for equal coverage: bank K=32 GLAP .284 vs PCA-oracle .072). Slalom is not a transfer failure but
higher-rank (its PCA-oracle is also worse) + grid-Lap's fixed low modes especially inefficient for the
S-curve; it transfers fine as K grows (K=96 in .311/transfer .324). E2b (data-fit transfer, the fix):
PCA fit on the 2 IN-SAMPLE arms, eval on held-out. RESULT [gridlap / pca_insample / coherence_geneig]:
bank K=8 .562/.261/.761, K=32 .284/.151/.749; vertical K=8 .544/.199/.697; slalom K=12 .894/.351/.916,
K=32 .703/.206/.905. => **PCA fit on TRAINING ARMS transfers across embodiment nearly as well as the
held-out oracle AND is 3-5x more mode-efficient than grid-Laplacian** (PCA K=8 beats grid-Lap K=32
everywhere; slalom K=12 PCA .351 ~ grid-Lap needs K>64). WHY it transfers (contra the task-confounded OXE
null): same task + shared scenes + fixed action interface => the DOMINANT action variance IS the shared
task structure (detour side/amp), body differences are the low-variance residual, so PCA top-k naturally
keeps shared + drops body-specific. PCA basis is ORTHONORMAL => pass-through pin still holds exactly (keep
the mechanism). The coherence generalized-eigenbasis (max Sb/Sw over all orthonormal dirs) FAILS (~.75,
near random) — pure invariance ignores energy => near-zero coverage; this REPLICATES + EXPLAINS the old
learn-U negative (invariance-only objective is the wrong objective). NET REFRAME of "improve the
representation": don't hand-design a bigger fixed basis — FIT the instruction subspace to the training-arm
action variance (PCA/variance-fit); it stays embodiment-invariant (verified transfer), is low-dim +
interpretable-enough, keeps pass-through, and fixes slalom. Grid-Laplacian's remaining role = a PRIOR for
the few-arm/few-scene regime where PCA is unreliable to estimate. OPEN (Tier-2/E6 follow-ups, not yet run):
confirm the PCA-pin gives the SAME transfer SUCCESS + steerability as grid-Lap (offline recon != success);
post-hoc interpretability of PCA directions (do they still correlate with side/bank like steer6d's d_side);
robustness of the PCA estimate as #training arms grows.

**#1 PCA-PIN vs GRID-LAPLACIAN-PIN TRANSFER SUCCESS — reconstruction win does NOT convert; invariance,
not coverage, drives transfer (2026-07-27, `rung3/transfer6d_pca.py`, `transfer6d_pca_result.json`)**:
extended transfer6d.py with a data-fit PCA basis (top-K principal dirs of pooled set-A chunks, the E2b
winner) as a frozen pinned subspace; same pipeline (freeze on set-A {Panda,IIWA,UR5e}, set-A scene->coeff
prior, relearn ONLY the executor on held-out Jaco, 10k iters, 3 seeds). POOLED transfer success (Jaco
ceiling 0.738): scratch 0.225, **GLAP 0.750** (oracle 0.725), **PCA 0.342** (oracle 0.475), Fourier 0.383,
random 0.300. c-invariance (||c_Jaco - c_setA||/||c_setA||, lower=more body-invariant): GLAP 0.075 < F 0.096
< Rand 0.133 ~ PCA 0.136. VERDICT: **the E2b reconstruction advantage does NOT convert to transfer success —
grid-Laplacian beats PCA >2x (0.75 vs 0.34)**, and success tracks coordinate BODY-INVARIANCE not coverage
(c-invariance ordering GLAP<F<Rand~PCA PREDICTS success ordering; coverage and transfer are ANTI-correlated,
the coverage-maximizing PCA is the worst-transferring structured basis). MECHANISM (reconciles with E1/E2b):
reconstruction rewards capturing variance INCLUDING the low-variance body-specific component, so PCA's top-K
absorb it (why it reconstructs the held-out arm well); but the PIN needs its coordinate PREDICTED invariantly
by the set-A prior, and a body-specific coordinate makes that prediction wrong -> wrong command injected ->
success drops. Confirmed by the prior gap: PCA oracle 0.475 >> PCA prior 0.342 (big body-mismatch penalty)
while GLAP oracle 0.725 ~= prior 0.750 (no gap; coordinate is invariant so set-A prior nails it). The missing
~25% (E1 "shared residual") is a SHARED SUBSPACE with a BODY-SPECIFIC COORDINATE: shared directions (fit each
arm's own coord -> reconstructs, looks shared) but the coordinate VALUE is body-specific (can't be predicted
across bodies). Grid-Laplacian's coherence selection (max Sigma_scene/Sigma_body) deliberately drops exactly
those directions, keeping only body-invariant-coordinate structure. => grid-Laplacian's INCOMPLETENESS IS
CORRECT for the pin: the uncaptured variance belongs in the relearned complement (realization), NOT the pinned
instruction; "completing" it via variance-fit pulls body-specific coordinates into the pin and HURTS transfer.
So "improve the representation" does NOT mean raise coverage — it means capture more body-INVARIANT-coordinate
structure (raise the coherent Sigma_scene/Sigma_body content), which grid-Laplacian already does and PCA does
the opposite of. Validates the standing "pin+prior transfers, basis secondary, invariance is what matters"
thread. Pass-through steering slope = 1.000 for BOTH bases on the relearned Jaco executor (orthonormal =>
mechanism survives per basis); the behavioral lateral-slope probe read ~0 for both (my raw-channel-mean probe
doesn't capture the coupled-task steering direction — steer6d already showed behavioral steering for GLAP, so
not re-litigated). CAVEAT: single held-out arm (Jaco), offline geometric success; PCA fit on pooled set-A
chunks (a PCA restricted to the invariant/coherent signal would land between PCA and GLAP but by construction
can't beat the coherence selection at invariance). NET for Denis's question: the grid-Laplacian is the right
pinned subspace for cross-embodiment; its "incompleteness" is a feature (body-specific coordinate correctly
excluded), and completeness (PCA) is actively harmful to transfer.

**VLA-SCALING DIRECTION SET — focus cross-embodiment (fork 2), keep steering-interface (fork 1) in
reserve (2026-07-27)**: Denis asked whether the experiments gear toward inserting the method into a VLA.
Honest assessment given: (i) the mechanism was ALREADY inserted into a real VLA in Phase 1 (source-noise
pin in pi0/openpi on LIBERO) — it steers (11x tighter, mm-calibrated) but pays the ~40pt obedience tax AND
the LIBERO averaging audit showed pi0's flow head is unimodal-with-noise so a regressor matches it =>
source-noise machinery adds nothing over regress+assign for SUCCESS on LIBERO. (ii) The recent toy/robosuite
work (grid-Laplacian, complement/deconvolution) is DELIBERATELY small-scale characterization (Denis chose
robosuite because pi0's pretraining is too strong to bottleneck), NOT scaling toward a VLA. (iii) Core
tension: the decomposition helps ONLY when the policy is bottlenecked, and VLAs are engineered NOT to be
bottlenecked — so "drop it into a standard VLA for better LIBERO success" is contradicted by our own
evidence. TWO insertable "things", different readiness: FORK 1 = pin as an interpretable/exactly-steerable
action handle (UNCONDITIONAL, provably scales via pass-through, already shown in pi0, untouched by the
deflation; a control/interpretability capability regardless of success); FORK 2 = structure/realization
DECOMPOSITION as a cross-embodiment FEW-SHOT adaptation architecture (freeze shared instruction, relearn
per-body realization/executor) — cross-embodiment few-shot IS a genuine VLA bottleneck (new body + scarce
data), the regime where the decomposition demonstrably helps, and the drone north star. DECISION (Denis):
both desirable, FOCUS FORK 2 (cross-embodiment). Consequence: the current complement/deconvolution study
feeds fork 2 directly (the realization adapter = the per-embodiment executor relearned on a new body); the
GATE for fork 2 is the variable-DOF/controller regime (still unsolved — the planned-pin + deconvolution work
is the attempt to crack it). Path to a VLA substrate for fork 2 = multi-embodiment data (OXE) with the
freeze-instruction/relearn-realization split, but OXE has variable action spaces (the unsolved regime) and is
task-confounded, so the controlled robosuite variable-DOF result must come first. NOT chasing better LIBERO
success (evidence says it won't come from this).

**BOX GPU CONSTRAINT (2026-07-27, Denis)**: use only ONE GPU for my jobs; Denis reserves the other for his
own tasks. IMPORTANT: which GPU is free CHANGES — as of 2026-07-27 Denis's cosmos-framework job runs on
GPU 0, so the FREE gpu is GPU 1. ALWAYS run `nvidia-smi` before launching a GPU job and pick the empty one;
do NOT assume an index. Pin the job with CUDA_VISIBLE_DEVICES=<free idx> (and MUJOCO_EGL_DEVICE_ID for
robosuite). CRITICAL JAX GOTCHA: JAX/openpi grabs memory on ALL visible GPUs by default, so a JAX job
(even compute_norm_stats, which only needs CPU) will intrude on the reserved GPU unless scoped. For CPU-only
JAX work use JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1; for GPU training use CUDA_VISIBLE_DEVICES=<free idx>.
Robosuite collections use a GPU via egl; autograd/CPU jobs (complement_study, transfer6d) use none.
UPDATE 2026-07-28: Denis LIFTED the GPU reservation ("we can use both gpus"); both GPUs available again.
UPDATE 2026-07-28 (later): Denis REINSTATED one-GPU-at-a-time ("only use one gpu at a time until I tell you
otherwise"). Run everything SERIALLY on a single GPU (currently GPU 0; GPU 1 left free). No parallel-across-GPU
drivers until told otherwise.
Still run nvidia-smi before launching and scope JAX with CUDA_VISIBLE_DEVICES to avoid grabbing both when only one is intended.

**VARIANCE-COLLAPSE CONCERN FROM CSFM, applied to the pin (2026-07-27, Denis flagged; CSFM arXiv:2602.05951)**:
CSFM's central warning is that folding conditioning into the flow SOURCE causes distributional collapse /
instability, fixed by variance regularization (KL sigma->1, mean free) + source-target directional alignment.
Our HARD PIN is the extreme endpoint of that spectrum: it sets the source coordinate on the instruction
subspace U to a deterministic c(obs) = ZERO variance on U (full Gaussian variance kept on the complement), so
it induces MAXIMAL variance collapse on U by construction. A variance-collapsed source cannot reproduce the
target's spread along U. Whether this matters depends on the TARGET CONDITIONAL VARIANCE along U: benign if
p(action|obs) has little spread on U (coarse structure ~determined by obs; consistent with LIBERO's near-
unimodality = why the hard pin didn't obviously hurt at pi0 scale), harmful if U carries genuine spread/
multimodality (then the pin underfits the distribution). This is the SAME failure as the Phase-1 obedience
tax, re-explained: a variance-collapsed source represents only the dictated coordinate, not the distribution
of valid actions, so it executes infeasible commands verbatim. STEERING vs FEW-SHOT SUCCESS pull opposite
ways: collapse is DESIRED for exact steering (why hard pin > CSFM for steering precision) but a LIABILITY for
few-shot distribution matching (where CSFM's mean-shift-with-retained-variance is the right instrument).
CONSEQUENCES for the VLA few-shot run: (1) a null/negative hard-pin result is AMBIGUOUS (decomposition useless
vs variance collapse discarded target spread) -- don't read a null as the former without ruling out the
latter; (2) cheap PRE-CHECK before training: fit obs->c on LIBERO data, measure RESIDUAL variance per U-mode
(part of c not explained by obs) -- low = safe to pin, high = pinning collapses real spread; tells us which
grid-Laplacian modes are safe and whether the hard pin fits this data; (3) keep a CSFM-style SOFT-SOURCE
variant (learn mean shift on U, regularize variance toward base not zero) as the designed fallback if the
hard pin shows collapse symptoms (underperforms scratch specifically on tasks with action spread).

**VARIANCE-COLLAPSE PRE-CHECK ON LIBERO — grid-Laplacian is a POOR instruction basis here (2026-07-27,
`rung3/vc_precheck.py`)**: ran the pre-check on real LIBERO lerobot actions (500 episodes, 3903 H=50 chunks,
20 tasks) BEFORE any VLA training. Decomposed c=U^T a variance into between-task vs within-task per
grid-Laplacian mode (K=16 over 50x7), + energy coverage. RESULTS: (1) top-16 low grid-Laplacian modes capture
only **0.197** of total action-chunk variance (vs ~0.9 for smooth robosuite detours) -- LIBERO actions are
contact-rich + bang-bang gripper + high-freq, NOT smooth-low-rank, so the pinnable smooth subspace is a
MINORITY of the action. (2) within_task fraction ~**0.99 across all modes** (mean 0.987; coarsest mode 0.94)
-- the language instruction explains only 1-6% of c; the pinned coordinate is per-episode detail, NOT a
task-level quantity. INTERPRETATION (calibrated): the strict variance-collapse risk is probably BENIGN
(serve_mean showed near-unimodality => within-task variance is most likely initial-state-explained = in the
obs, not stochastic), so a collapse would not discard much irreproducible spread. The REAL problems the
pre-check surfaced are (a) low COVERAGE (pin touches ~20% of action variance) and (b) c is NOT task-aligned
(so the obs->c prior is not a simple few-shot-friendly task map, and the pin injects mostly non-task detail).
NET: the grid-Laplacian basis that fit smooth robosuite detours is a POOR FIT for LIBERO action structure;
caught before spending training compute. OPTIONS before training: (a) data-fit (PCA) instruction subspace on
source LIBERO actions (captures more energy; PCA-transfers-across-TASK is untested and differs from the
cross-EMBODIMENT PCA-null); (b) accept LIBERO actions aren't smooth-low-rank => grid-Laplacian is the wrong
instrument here; (c) proceed with grid-Laplacian as a weak first baseline anyway. Refinement available if
wanted: regress c on the chunk's initial STATE (parquet has it) to split within-task variance into
obs-explained (safe) vs stochastic (collapse) and definitively close the collapse question.

**VLA PIN INFRA WORKING — data-fit PCA pin trains in pi0/LoRA (2026-07-27)**: Denis chose the data-fit
subspace. Confirmed PCA >> grid-Laplacian for LIBERO (`rung3/vc_pca_check.py`, 40-task split): PCA-16
coverage 0.94 + cross-TASK transfer 0.24 rel err (held-out tasks) vs grid-Lap 0.18 / 0.87(~random). Built the
PCA instruction subspace in pi0's EXACT training action space via the full data loader (`openpi/make_u_pca.py`,
CPU-only): actions are normalized+padded (50x32=1600), U=(1600,16) orthonormal, top-16 coverage 0.956, saved
`rung3/pin_U_pca.npy`. Inserted an env-gated source-noise pin into pi0 training (`rung3/patch_pi0_pin.py`
patches `openpi/src/openpi/models/pi0.py` compute_loss; backup pi0.py.snmvp_bak): when SNMVP_PIN_U is set it
replaces noise's U-coordinate with the action's (u_t=0 on U => pass-through); no-op otherwise; inference needs
no edit (sample_actions takes a `noise` arg). GOTCHA fixed: keep _PIN_U as a NUMPY constant loaded at import
and do jnp.asarray INSIDE compute_loss (a module-level jnp array escaped the jit as a tracer -> UnexpectedTracerError).
SMOKE (10-step LoRA `pi0_libero_low_mem_finetune`, GPU 1, WANDB_MODE=disabled): exit 0, "SNMVP pin enabled
shape (1600,16)", Step0 loss 0.702 grad_norm 5.87, no OOM/tracer. Launch pattern: CUDA_VISIBLE_DEVICES=1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled SNMVP_PIN_U=<npy> uv run scripts/train.py
pi0_libero_low_mem_finetune --exp-name=... --num-train-steps=N --overwrite. Data re-downloaded (~20GB lerobot
physical-intelligence/libero), norm-stats computed. NEXT (few-shot study, not yet built): short source LoRA
train (pin vs scratch), obs->c prior (c is ~92% within-task so prior must read full obs not just language;
learnability from few demos is the crux), few-shot adapt to held-out tasks, closed-loop LIBERO eval pinning
prior(obs) into the PCA source subspace (adapt Phase-1 serve_snmvp_policy). CAVEAT still live: PCA pins the
HIGHEST-variance directions so CSFM variance-collapse risk is more acute than grid-Lap; keep CSFM-soft fallback.

**FULL CLOSED-LOOP FEW-SHOT STUDY — pipeline in progress (2026-07-27, Denis chose full closed-loop)**:
GATE CHECK (`state_c_check.py`, CPU): can c=U^T a be predicted online (no action at inference)? Linear
state->c held-out R2 = 0.672 overall; per PCA mode [0.97,0.99,0.64,0.87,0.67, 0.44,0.14,...] -- the TOP,
highest-variance modes are strongly obs-predictable, the low-variance tail is not. => closed-loop pin IS
drivable, and the principled pinned subspace = the TOP PREDICTABLE modes (high variance + obs-determined,
avoids collapsing spread the prior can't recover). Refined pin to K=5 (`pin_U_pca_k5.npy`, modes 0-4,
R2 0.64-0.99). TASK HOLDOUT (`make_task_split.py`): 8 of 40 tasks held out for few-shot [0,1,2,9,11,17,21,28];
1373 source / 320 held-out episodes; `source_episodes.json`, `heldout_tasks.json`. Env-gated episode filter
patched into `openpi/.../data_loader.py` create_torch_dataset (SNMVP_EPISODES=json list -> LeRobotDataset
episodes=; backup .snmvp_bak). SOURCE TRAINING LAUNCHED (`run_src_train.sh`, GPU 1, sequential): pin
(SNMVP_PIN_U=k5, exp snmvp_src_pin) then scratch (exp snmvp_src_scratch), each LoRA pi0_libero_low_mem_finetune
5000 steps save-interval 2500, holding out the 8 tasks; checkpoints -> openpi/checkpoints/pi0_libero_low_mem_finetune/.
EVAL HARNESS REUSABLE from Phase 1: `scripts/serve_snmvp_policy.py` = SnmvpNoisePolicy wrapper that runs a
PRIOR on obs -> invariant, converts via `snmvp.openpi_adapter.make_calibrated_noise` -> policy.infer(obs,
noise=); `libero_eval_client.py` drives the LIBERO sim. Closed-loop pinning pipeline already exists; adaptation
needed = retarget make_calibrated_noise to pin the PCA subspace U (noise = g - (g@U)@U^T + c@U^T) and retrain
the prior for c=U^T a (K=5; state->c works). REMAINING (next phase): (1) retarget serving-side noise to the
PCA-U pin + train obs->c prior; (2) few-shot LoRA-adapt each source ckpt (pin, scratch) on k demos of each
held-out task; (3) closed-loop LIBERO eval, compare pin+prior vs scratch success vs k. This is a multi-hour
multi-run phase. Milestone reached: pin trains in a VLA with a LIBERO-fitting basis + gate check positive.

**2500-CHECKPOINT PASS-THROUGH CHECK — POSITIVE (2026-07-28, `openpi/probe_pin.py`, `probe_pin.log`)**: Denis
asked to check the step-2500 pin source checkpoint before finishing. Offline probe in the model's normalized
action space on HELD-OUT-task chunks (192 chunks, GPU 0): for each, set the source-noise U-coordinate to the
oracle c=U^T a and measure the produced action's U-coordinate vs c. RESULT: PIN pass-through rel err mean
0.136 / median 0.110; UNPINNED (fresh Gaussian) 0.889. => the data-fit PCA pin trained into pi0 (LoRA, 2500
steps) and PASSES THROUGH on held-out tasks (~0.11-0.14, will tighten at 5000 steps), vs ~0.89 unpinned.
Confirms the mechanism + PCA basis + pinned-noise inference path at VLA scale, generalizing to unseen tasks.
NOTE: make_calibrated_noise (Phase 1) pins raw leading action DIMS, not a subspace; the probe built PCA-U
pinned noise directly (noise = g - (g@U)U^T + c@U^T) and called model.sample_actions(noise=) via
policy._model. GPU RESERVATION LIFTED (both GPUs usable). Source pin run finishing (4.75k/5k); scratch next.
CLOSED-LOOP INFRA EXISTS from Phase 1: separate LIBERO client venv `openpi/examples/libero/.venv` + package
`openpi/third_party/libero` + `scripts/libero_eval_client.py` (drives sim, websocket to serve_snmvp_policy;
supports a per-call pin via obs key). Closed-loop few-shot remaining: retarget serve_snmvp_policy to build
PCA-U pinned noise + an obs->c prior (K=5; state->c linear R2~0.87-0.99 top modes), finish scratch source,
few-shot LoRA-adapt each source ckpt on the 8 held-out tasks (16 runs, parallelize across both GPUs), serve +
libero_eval_client closed-loop success pin+prior vs scratch vs #demos.

**CLOSED-LOOP FIRST CUT — HARNESS WORKS, but the tasks floored both arms (2026-07-28)**: completed the full
pipeline. Source models: pin/4999 + scratch/4999 (LoRA pi0_libero_low_mem_finetune, 5000 steps, 8 tasks held
out; fsdp2 deadlocked on NCCL clique -> single-GPU). Few-shot: k=10 LoRA-adapt of each source on held-out
tasks 0,1 (init-from-ckpt via env-gated SNMVP_INIT_CKPT patch to get_config; pin arm SNMVP_PIN_U on, scratch
off) -> fs_{pin,scratch}_t{0,1}/799. Priors: state->c ridge R2 0.759 (t0) / 0.815 (t1). Task mapping
(`task_map.json`, built in LIBERO client venv w/ PYTHONPATH=third_party/libero LIBERO_CONFIG_PATH): our 40
lerobot tasks span libero_10/90/goal/object; held-out 0,1 -> libero_10 tasks 4,6 (LONG-HORIZON = hardest).
Harness built + WORKS: `serve_pca_pin.py` (PCA-U pinned noise from prior, `--prior` off = unpinned baseline),
`pca_pin.py`, `make_prior.py`, single-task client filter (SNMVP_TASK_ID patch), `run_cl_eval.sh`/`run_cl_all.sh`.
Serve pattern: CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G PYTHONPATH=.../third_party/libero
LIBERO_CONFIG_PATH=~/code/libero-config; client examples/libero/.venv/bin/python libero_eval_client.py.
RESULT (10 trials each, real ~17s rollouts, verified not a crash): pin_t0/scratch_t0/pin_t1/scratch_t1 ALL
0.0. => NO HEADROOM: libero_10 long-horizon + k=10 few-shot from a 5000-step LoRA source floors BOTH arms at
0, so pin-vs-scratch can't be distinguished (analog of the toy scratch=ceiling null, at the floor). NOT a pin
failure (pin didn't underperform scratch; both 0). FIX for a meaningful cut: eval EASIER held-out tasks
(short-horizon) where scratch is partial: held-out 21,28 -> libero_object 4,5 (easiest); 11,17 -> libero_goal
1,9. Also worth: check SOURCE competence (eval scratch/4999 zero-shot on an easy task; if ~0 the 5000-step
LoRA source is undertrained vs the standard 30k full finetune and needs more source training). Held-out task
indices are [0,1,2,9,11,17,21,28]; only 0,1 have few-shot adapts+priors so far.

**EASY-TASK RE-CUT ALSO FLOORS AT 0 — the SOURCE doesn't generalize to held-out tasks (2026-07-28)**: redid
the few-shot cut on the EASY held-out libero_object tasks 21,28 (-> object tasks 4,5; short-horizon).
Source-competence check first: scratch/4999 on an IN-SOURCE object task (task 9) = 0.8 (source IS competent
at trained tasks). Priors R2 0.90/0.91. Few-shot k=10 (pin+scratch) + closed-loop: ALL FOUR = 0.0
(cl_{pin,scratch}_t{21,28}). DIAGNOSTIC (source zero-shot on the HELD-OUT object tasks): scratch/4999 ->
object4 0.0, object5 0.1. => ROOT CAUSE is the SOURCE, not the pin: a 5000-step LoRA on 32 specific LIBERO
tasks learns THOSE (0.8 in-source) but does NOT transfer to a held-out task (0-0.1), and k=10 LoRA can't
teach a new LIBERO task from ~0. Both arms floor => NO HEADROOM for a pin-vs-scratch comparison. Third
no-headroom outcome at VLA scale (LIBERO averaging deflation; libero_10 floor; now held-out-object floor).
The recurring obstacle: constructing a regime where the VLA is PARTIALLY competent (headroom) not
saturated/floored. WHAT IS SOLID at VLA scale: the pin MECHANISM (grid-Lap probe pass-through 0.11-0.14 on
held-out tasks; PCA-pin trains + passes through) + the closed-loop harness (serve_pca_pin + prior + LIBERO
client, all working). What is NOT achievable in this budget: a clean few-shot pin-vs-scratch SUCCESS number,
because held-out LIBERO tasks are too distinct for a weak-LoRA source to few-shot. OPTIONS: (a) stronger
source (full finetune / much longer LoRA on the 32-task split, ~1 day) so it nears held-out tasks -> few-shot
gets a foothold; (b) change few-shot target to a source-KNOWN task with limited data (data-efficiency, not
new-task) for headroom; (c) bigger k (25/50) [likely still floors given source 0 zero-shot]; (d) accept the
robosuite bottlenecked results as the defensible "pin helps when bottlenecked" evidence and frame the VLA
work as mechanism-validated-but-headroom-not-achievable. Denis's call.

**COMPLEMENT STUDY + BROKEN-ACH STRESS TEST (2026-07-27, `rung3/complement_study.py`, `complement_result.json`,
`data_dyn/{lat3,lat5,stiff,soft}.npz`)**: fork-2 push on the realization/complement side. (1) CONTROLLED
FOUR-WAY on sim2real (set-A sim1/2/3, held-out 'real' kp250/d0.75, ceiling 0.789, 3 seeds, held-scene sweep
n=10/25/50) [S scratch / ACH achieved-pin+prior+relearned-exec / PLAN planned-path-pin+exec / DECONV
planned+FIR realization]: S 0.28/0.35/0.31; ACH 0.69/0.74/0.70 (near ceiling, data-efficient); PLAN
0.17/0.18/0.38 (FAILS, below scratch); DECONV 1.0 (soft — fir_err 0.39, reaches under the lenient geometric
metric from the deterministic planned reference, does NOT reproduce the real trajectory). KEY LESSONS: (a)
the achieved-pin transfer works + is data-efficient on a MODERATE dynamics gap (re-confirms 2.8). (b) NAIVE
PLANNED-PIN FAILS: pass-through CLAMPS the achieved coordinate to the planned value, but real tracking
deviates from plan (achieved-vs-planned gap 0.54 in the pinned modes) => clamps a false target, executor
can't override a hard clamp. LESSON: cannot pin the planned path directly; the plan->achieved realization
map must be MODELED (deconvolution), not clamped. (c) DECONV's win is soft (lenient metric + starts from a
valid plan). (2) BROKEN-ACH STRESS TEST (cheap latency/dynamics proxy, Denis chose latency-first go/no-go):
tried to create a regime where ACH BREAKS but task stays solvable. Latency lat3/lat5 -> ceiling 0.06/0.00
(INFEASIBLE: high latency on a fixed-horizon tracking task makes the endpoint unreachable). Aggressive
FEASIBLE gain/damping: stiff kp330/d0.6 -> ceiling 0.818 FEASIBLE but ACH c-invariance only 0.109->0.161
(NO break); soft kp95/d1.45 -> ceiling 0.097 INFEASIBLE, c-inv 0.183. VERDICT: **no feasible-but-broken-ACH
regime exists via dynamics** — the low-freq pinned coordinate is dynamics-ROBUST whenever the task is
solvable (ringing/transient is high-freq, doesn't touch the low-freq grid-Laplacian coordinate that any
feasible controller must produce); the only perturbations that break the coordinate also kill feasibility.
=> the achieved-pin robustness is a POSITIVE result, and the broken-but-solvable regime that needs a modeled
realization complement requires a genuine CONTROLLER/DOF change (different action semantics), i.e. the real
fork-2 variable-DOF gate — cannot be proxied by dynamics. Added a stricter trajectory-match diagnostic
(*_traj = rel err of each method's output vs the held-out body's OWN achieved demos) to complement_study.py
to quantify DECONV's off-manifold leniency; not yet exercised on a broken regime. NEXT: commit to the
variable-DOF collection (bottlenecked hard position task solvable by BOTH a 3-ch OSC_POSITION and 6-ch
OSC_POSE controller) = the actual gate; pin the shared PLANNED task path, model the per-controller realization
(deconvolution), test transfer across the action-dim/controller change where ACH is expected to break.

**KEY FINDING — the pin-trained VLA is HOSTAGE to the prior at inference (2026-07-28)**: parity check served
the PIN source (snmvp_src_pin/4999) UNPINNED (fresh Gaussian, --prior NONE) on an IN-SOURCE object task
(task9): 0.0, vs SCRATCH source 0.8 on the same task. NOT a broken ckpt -- pass-through working: fresh
Gaussian => RANDOM U-subspace coordinate, and the pin-trained model faithfully passes that random
"instruction" through into the action, wrecking it; scratch has no pinned subspace so generates freely and
succeeds (0.8). CONSEQUENCE: the pin-trained policy REQUIRES a meaningful c at inference (a good prior); a
wrong/random c forces a bad action, whereas scratch is unconstrained. The pin's closed-loop success is GATED
BY PRIOR QUALITY = variance-collapse/obedience-tax at VLA scale, closed-loop. Implications: (1) every pin-arm
eval MUST serve WITH a good prior (few-shot evals do). (2) Clean closed-loop confirmation of pass-through at
VLA scale: random U-coord => 0.0. (3) Fair parity = serve pin source PINNED (prior on the in-source task);
expected ~0.8 if mechanism sound. DEEPER POINT: the hard pin trades the policy's own free generation for
obedience to c, so at VLA scale it can only match/beat scratch if the prior supplies c BETTER than the free
policy would -- a high bar; consistent with CSFM soft-source (keep variance) for generation vs hard pin for
exact steering. Running fewshot_v2 (full-demo few-shot task21, pin served pinned w/ prior_t21) for
learnability. This reframes the VLA few-shot question: it is really "can a prior supply c well enough to beat
free generation," and the honest read so far is the hard pin is the wrong instrument for few-shot SUCCESS
(great for exact steering, not for beating a free generative policy).

**FIRST POSITIVE VLA CLOSED-LOOP RESULT — pin+prior BEATS free scratch on full-demo few-shot (2026-07-28,
`run_fewshot_v2.sh`, `cl_{pin,scratch}_t21_full.json`)**: the k=10 floors were too few demos (both arms 0).
With FULL demos (45) + 3000-step few-shot on held-out libero_object task 21, 10 trials: **PIN+prior = 0.80
(8/10) vs SCRATCH (free) = 0.30 (3/10)**. Pin arm reaches the source's own in-source competence (0.8); free
fine-tuning only gets 0.3 from the SAME 45 demos. => the decomposition (predict low-dim c from a good prior,
R2 0.90, + generate the completion around the pinned c) extracts MORE from the same demos than free
fine-tuning. This CORRECTS the pessimistic "hard pin is the wrong instrument for few-shot" read: with a GOOD
prior the pin clears the bar (prior's c beats free generation). Reconciles the hostage-to-prior finding:
random c -> 0.0, good-prior c -> 0.80. This is the tangible few-shot improvement Denis wanted, first evidence
at real VLA scale, closed-loop. CAVEATS (all addressable): 1 task, 1 seed, 10 trials; 45 demos = full task
set, not small-k yet; pin uses the demos both to adapt the executor AND fit the prior (that dual use IS the
method; both arms see the same 45 demos). CONFIRM NEXT: (1) replicate on held-out task 28; (2) k-sweep
(45->25->15->10) to map the data-efficiency curve and find where pin's advantage is largest / where both
floor; (3) more trials/seeds for tighter estimates. All on GPU 0 (GPU 1 = Denis's).

**REPLICATED on task 28 (2026-07-28, `run_replicate_t28.sh` / `run_rep28_par.sh`, `cl_*_t28_full.json`)**:
full-demo (42) 3000-step few-shot, 10 trials. TASK21: pin+prior 0.80 (8/10) vs scratch 0.30 (3/10), +0.50.
TASK28: pin+prior 0.80 (8/10) vs scratch 0.20 (2/10), +0.60. => the positive result HOLDS across two
independent held-out libero_object tasks: pin+prior reliably reaches ~0.80 (= source in-source competence)
while free scratch fine-tuning gets only 0.20-0.30 from the SAME demos. Solid, replicated, real-VLA-scale,
closed-loop: the pin+prior decomposition is substantially more demo-efficient than free fine-tuning. NOTE:
"full demos" = 45/42 (the whole held-out task demo set), 3000 steps; this is demo-efficiency at the full set,
not yet small-k. NEXT: k-sweep (45->25->15) to map where pin's advantage is largest and where both floor
(k=10 earlier floored both at 0). GPU rule volatile this session: Denis toggled reserve/release GPU 1 several
times (kill scratch on GPU1, keep pin on GPU0, then re-grant both) -- always nvidia-smi + be ready to free
GPU1 on request; keep runs as self-contained setsid scripts with status files (SSH drops frequently, launcher
returns 255 but detached jobs run; watchers get culled -- rely on status files + Denis pings).

**STEP-SWEEP + K-SWEEP — the 0.8-vs-0.3 headline was NOISE-INFLATED; robust signal = pin learns FASTER,
converges similar (2026-07-29, `run_stepsweep_t21.sh`, `run_ksweep_t21.sh`, `cl_*_t21_s*.json`,
`cl_*_t21_k{15,25}.json`)**. All task21, libero_object task4, 10 trials each. STEP-SWEEP (full 45 demos, vs
training steps) [pin/scratch]: s500 0/0; s1000 .1/0; s1500 .4/.1; s2000 1.0/.7; s2500 1.0/.7; s2999 .9/.8.
=> CLEAN, informative: both need ~1500-2000 steps to leave the floor; PIN LEADS during the rise (s1500-2500
pin .4-1.0 vs scratch .1-.7, reaching 1.0 at s2000 while scratch .7), then they CONVERGE by s3000 (.9 vs .8).
Supports "the decomposition accelerates learning" (pin competent in fewer steps). K-SWEEP (3000 steps, vs
#demos) [pin/scratch]: k10 0/0; k15 .4/0; k25 .2/.4; k45 .8/.3(earlier). => NOISY / NON-MONOTONIC (pin wins
k15,k45; scratch wins k25) -- 10-trial noise dominates. RUN VARIANCE is real: scratch at ~3000 steps was 0.30
(earlier full-demo run) vs 0.80 (step-sweep s2999) -- SAME config, different run, big swing. HONEST CORRECTION
to the earlier "pin 0.80 >> scratch 0.30" excitement: that gap is within the noise band; the reproducible
claim is the WEAKER "pin reaches competence in fewer training steps, converging to ~similar final success,"
NOT a large sustained success gap. 10 trials is too few (+-0.15-0.2). NEXT to firm up: many more trials
(20-50) at the key points (esp. the s1500-2500 rise where the pin lead is); the step-sweep speed-of-learning
signal is the cleaner claim than the k-sweep. Overall status: the pin MECHANISM + few-shot pipeline work at
VLA scale; the pin gives a modest, noisy learning-speed advantage that converges -- not the big gap first
seen. Consistent with the whole arc: pin's value is real but modest/regime-dependent, easy to over-read from
few trials.

**DEFINITIVE 50-TRIAL STEP-SWEEP — pin PEAKS EARLY+HIGH then OVERTRAINS; scratch slower but ends higher
(2026-07-29, `run_stepsweep_moretrials.sh`, `cl_*_t21_s*_n50.json`)**: re-evaluated the existing step
checkpoints at 50 trials (full libero_object init-state set, +-~0.07). Task21, success vs training steps
[pin/scratch]: s500 .08/0; s1000 .24/.16; s1500 .22/.06; s2000 .98/.62; s2500 .84/.56; s2999 .66/.80.
CLEAN READ (tighter than 10-trial): (1) PIN LEARNS FASTER + reaches a HIGH PEAK EARLY -- 0.98 at s2000 vs
scratch 0.62 (+0.36), leads at every step through s2500. (2) PIN OVERTRAINS past its peak: monotone decline
0.98(s2000)->0.84(s2500)->0.66(s2999) -- exceeds noise, looks real. (3) SCRATCH slower but keeps improving,
ends at 0.80 > pin's final 0.66. NET: the pin's benefit is SPEED/compute-efficiency (fast high peak) but it
REQUIRES EARLY STOPPING; trained to convergence it underperforms free scratch. This REVISES the 10-trial
"converge to similar" read (that was noise: 10-trial gave pin .9/scratch .8 at s2999; 50-trial gives pin
.66/scratch .80 -- scratch ahead at the end). Practical implication: pin + early-stop at ~s2000 gives 0.98,
far better+faster than scratch's best (0.80); without early stopping the pin loses. Mechanism (plausible):
the forced pinned coordinate accelerates initial learning (hands the model the coarse structure) but the
flow adapter overtrains the completion / the fixed pin becomes a constraint that free scratch eventually
beats by generating freely. CAVEAT: single training run per arm; the overtraining decline should be confirmed
with multiple training seeds (50 trials tightens EVAL noise, not TRAIN-seed variance). This is the honest,
well-supported headline: pin = faster learning to a higher early peak + overtrains; scratch = slower, higher
at convergence. Reconciles with steering-vs-generation theme: the hard pin helps early (structure handed for
free) but its rigidity caps the ceiling that a free generative policy eventually surpasses.

## Addressing the late-step decline (2026-07-27/29)
Two "intelligent fix" attempts on the pin step-sweep decline (pin peaks ~0.98 @step2000 then falls to ~0.66 @3000).

ATTEMPT 1 (refit prior) - FAILED. Hypothesis: prior-error amplification (prior fit on k=10 demos, model
adapted on 45). Refit prior on all 45 demos (R^2 0.851, LOWER than the k=10 prior's 0.902 - the k=10 R^2 was
inflated by a small held split), re-evaluated the SAME step checkpoints (30 trials). Curve kept the same
peak-then-decline shape: step500 0.067, 1000 0.333, 1500 0.133, 2000 0.87, 2500 0.70, 2999 0.57 (vs orig
0.08/0.24/0.22/0.98/0.84/0.66). Conclusion: the decline lives in the MODEL CHECKPOINTS, not the prior (both
curves use the same checkpoints, differ only in prior). Prior-error amplification refuted.

CONFIG FINDING: the LoRA adapt ran with ~zero regularization - AdamW weight_decay=1e-10 default and EMA OFF
(pi0_libero_low_mem_finetune sets ema_decay=None). Supports an overfitting story: pin hands coarse structure
via c, adapter converges fast then overtrains the completion on 45 demos with no wd/no weight-averaging.

ATTEMPT 2 (regularization) - RUNNING. Env-gated override added to config.get_config: SNMVP_WD sets AdamW
weight_decay, SNMVP_EMA sets ema_decay (patch_wd_ema.py; verified wd=0.001 ema=0.99). Retrain fs_pin_t21_reg,
SAME seed+data (isolates reg effect), wd=1e-3 + ema=0.99, save every 500 to step 3000, then re-eval curve
(prior_t21, 30 trials) -> cl_pin_t21_reg_s{S}.json. NOTE openpi checkpoints.py:146 saves ema_params AS the
served params when EMA on, so the eval genuinely serves the EMA-averaged weights (not a no-op). If the curve
flattens -> regularization addresses the decline (better than early stopping); if it still declines with same
seed+data -> the decline is robust to this reg and likely genuine overtraining needing stronger measures or a
second-seed confirmation. GPU0, ~2.5h.

## Reg fix result + PCA-component sweep launch (2026-07-29)
ATTEMPT 2 RESULT (wd=1e-3 + EMA=0.99, same seed+data, isolates reg) - DID NOT FIX the decline.
Curve (30 trials): s500 0.00, s1000 0.167, s1500 0.067, s2000 0.867, s2500 0.933, s2999 0.30 (vs orig 50-trial
0.08/0.24/0.22/0.98/0.84/0.66). Regularization shifted the peak later (2000->2500) and kept it high (0.93) but
the collapse persists and is steeper (0.93->0.30). So the decline is NOT ordinary weight-norm overfitting.
Two runs (orig unregularized, reg) both peak-then-decline, but share a seed. Leading remaining hypothesis:
served-distribution narrowing / variance collapse from the deterministic pinned source (the CSFM concern),
which wd/EMA would not touch. Verified openpi checkpoints.py:146 serves ema_params when EMA on (not a no-op).

DECISIVE CONTROL LAUNCHED: seed2 step-sweep (fs_pin_t21_seed2, --seed=1, else identical, no extra reg) on GPU0
-> cl_pin_t21_seed2_s{S}.json. If it also peaks-then-declines -> real seed-independent pin+flow overtraining
(benign, early-stop handles it); if not -> orig was single-run noise. This is the LAST decline experiment
(user agreed peak ~0.98 is the real result; decline only matters as benign-vs-mechanism-instability).

PCA-COMPONENT SWEEP LAUNCHED (user's "what does the pin represent / more components?" question). NOTE the
existing run_ksweep_t21.sh is a DEMO-COUNT sweep (k15/k25), a different axis. New driver run_pca_pipeline.sh
(args K GPU PORT [WAITFILE WAITTOKEN]) runs per-K full pipeline: source-pin pretrain 5000 (snmvp_src_pin_kK)
-> few-shot adapt 3000 step-checkpointed (fs_pin_t21_kK) -> K-dim prior (make_prior.py PATCHED to read
SNMVP_PIN_U; was hardcoded to k5) -> eval {1500,2000,2500,2999} 30 trials -> cl_pin_t21_kK_s{S}.json, deleting
8.7G step ckpts after eval. U's: make_u_pca.py SNMVP_K env; coverage K10=0.890, K16=0.956 (K5 partial).
Running: K16 on GPU1 now, K10 on GPU0 after SEED2_DONE. Interpretation: K16 raises peak => more instruction
dims add representable action content; same peak but moved/steeper decline => 5 dims already carry the useful
instruction, extra dims only add overfit capacity. All robust via setsid; status files pca_k{K}.status.

## seed2 decline verdict + k10 handoff bugfix (2026-07-29)
SEED2 (--seed=1, else identical to orig step-sweep, no extra reg), 30 trials:
s500 0.00, s1000 0.033, s1500 0.367, s2000 0.867, s2500 0.90, s2999 0.767.
Three-run peak->end drops: orig 0.98->0.66 (-0.32, 50 trials), reg 0.93->0.30 (-0.63), seed2 0.90->0.767 (-0.13).
VERDICT: the late decline is REAL and seed-independent in DIRECTION (all 3 runs end below their own peak) but
MILD and highly variable in MAGNITUDE (-0.13 to -0.63), and the peak LOCATION wanders (step 2000 vs 2500).
Reads as a genuine but modest overtraining tendency on top of substantial train-trajectory + eval variance -
NOT mechanism instability. Benign: early-stop in the 2000-2500 window captures the pin advantage; pin's value
is fast-to-high-peak (scratch only catches up near convergence ~0.80). Decline question CLOSED. seed2 at n=30
SE~0.07 so its -0.13 drop is ~1.3 SE (borderline); orig -0.32 is real (>4 SE). Don't spend more on "fixing" it.

BUGFIX: run_pca_pipeline.sh waiter greps a RELATIVE path but the driver cd's to ~/code/openpi first, so the
GPU-handoff wait never sees SEED2_DONE (loops uselessly). K10 was stuck; killed and relaunched directly on the
freed GPU0 with no waiter. If reusing the waiter, pass an ABSOLUTE WAITFILE path. Both K10 (GPU0) and K16
(GPU1) source pretrains now running; full K={5,10,16} sweep ETA ~4-5h.

## Sim->real (Bridge) faithful pipeline (2026-07-30)
GOAL (user): train pi0 flow on sim (LIBERO), then on real (Bridge) learn ONLY the pin (state->c
prior refit; flow frozen) and see if it produces correct real actions.

DATA: bridge_extract_raw.py streams 300 Bridge episodes from gs://gresearch/robotics/bridge/0.1.0
(tfds via `uv run --with tensorflow-cpu --with tensorflow-datasets`) -> data_bridge_raw/ep_*.npz
(image 256, state 7, action 7 = world_vector3+rotation_delta3+open_gripper1; lang in
observation["natural_language_instruction"]). Median episode 34 steps (<H=50). LIBERO control set:
libero_extract_raw.py -> data_libero_raw (38 single-task eps, image+wrist+state+action).

KEY BUGS FOUND (via a LIBERO control on the offline evaluator - essential; the shortcut looked
plausible but was wrong):
1) pi0_libero_low_mem_finetune has extra_delta_transform=True, so model action space != normalized
   raw deltas. Manual c=U^T(normalize(raw)) mismatched -> injecting the pin BLEW UP predictions 4x
   on LIBERO (oracle should help). Fix: pi0_libero_shared config = copy with
   extra_delta_transform=False (registered statically in _CONFIGS_DICT via patch_shared_config2.py
   so tyro cli() accepts it; cli builds from _CONFIGS_DICT).
2) "shared space" via COMBINED LIBERO+Bridge norm stats is wrong: action scales differ ~30x
   (LIBERO pos-delta std ~0.34 vs Bridge ~0.01) and 273k vs 10k frames -> combined dominated by
   LIBERO, crushes Bridge to ~0. Correct shared space = PER-DOMAIN standardization (each by own
   raw-delta stats -> both unit) with ONE pin U as cross-domain basis. build_shared_norm.py places
   LIBERO-own raw-delta stats at assets/pi0_libero_shared/...; Bridge uses its own bridge_norm.

FAITHFUL SETUP (running): shared-space U pin_U_pca_k5_shared.npy (K=5, coverage 0.468 in no-delta
LIBERO-standardized space). Retrain source-pin flow snmvp_src_pin_shared under pi0_libero_shared
(SNMVP_PIN_U=shared U, source_episodes, 5000 steps, GPU0 ~2.7h). Then run_simreal_eval.sh runs
eval_offline_action.py: (1) LIBERO control (validate: pass-through low, oracle>>no_pin), (2) Bridge
arms no_pin/mean_c/real_prior/oracle. Decomposed metrics: pass-through relerr (does frozen flow
pass the pinned coord through on real?), subspace R^2 (pin CHANNEL re-grounds?), full-action R^2
(complement included). create_trained_policy takes norm_stats= (inject per-domain) and infer(noise=)
(inject pin). NOTE Bridge is multi-task so state->c prior may be weak (instruction is in language,
not proprioceptive state) - a known risk for the state-prior pin across many tasks.

## Sim->real (Bridge) RESULTS + evaluator validated (2026-07-30)
Faithful pipeline finished. Evaluator VALIDATED on LIBERO control (in-dist, snmvp_src_pin_shared,
pi0_libero_shared no-delta, shared U): no blowup (pred_mag 0.22 vs gt 0.22); oracle passthru_err
0.19, oracle subspace_R2 0.95 (flow passes injected coord through); ordering no_pin(-0.85)/
mean_c(-1.29) < real_prior(0.54) < oracle(0.95). So the fix (extra_delta_transform=False +
per-domain standardization) works and the metric is trustworthy. Note full_R2 negative even for
LIBERO oracle (-0.39) because open-loop 50-step chunk R2 is a weak proxy (that flow is ~0.98
closed-loop); trust subspace_R2 + passthru, not absolute full_R2.

BRIDGE (160 held-out samples): prior state->c R2 = 0.098 (!). Arms subspace_R2: no_pin -0.11,
mean_c -0.48, real_prior -0.41, oracle 0.36; passthru oracle 0.37; full_R2 ~-0.5 all arms;
pred_mag no_pin 0.14 vs gt 0.12 (pinned arms 0.07). VERDICT: hypothesis fails on Bridge but for a
diagnosable reason, not because the pin is wrong: (1) pin channel degrades gracefully OOD not
breaks (oracle passthru 0.19->0.37, subspace 0.95->0.36 vs LIBERO); (2) real_prior ~ no_pin
because state->c prior is ~useless (R2 0.1) - Bridge is MULTI-TASK so the instruction is in
LANGUAGE not proprioceptive state; (3) full-action transfer poor for all arms (frozen sim
complement OOD on WidowX). Both failure causes = consequences of choosing the hardest gap.
IMPLICATIONS for digital-twin->real (the real target): use a c-predictor on the instruction signal
(language/image), NOT proprioceptive state, for multi-task; keep embodiment matched so the
complement transfers (twin does, Bridge doesn't). Oracle's partial survival (0.36) across the
Bridge chasm suggests the pin channel should transfer well in a matched-embodiment twin.
Confirms the refined hypothesis: PREDICTABILITY of c from the conditioning is the binding criterion.

## Language steerability of the pin (2026-07-30)
E1 (LIBERO single-scene suites goal/object/spatial, c=U^T a in no-delta shared space): the PCA pin
coordinate is mostly STATE/motion-phase, NOT language. between-instruction (language) share of
Var(c): goal 23%, object 1%, spatial 5%, all 15%; within-instruction (state) 77-99%. Because in
these suites language changes the TARGET not the gross MOTION (pick ketchup vs milk = same
trajectory). Selecting U for relevance helps but hits a ceiling: between-PCA U 27%, LDA
(max between/within) 43% but in a ~2%-variance slice. => an action-derived pin can only be
language-steered to the extent the ACTION depends on language, which is small here.

Denis's diagnosis: need language to change the MOTION (navigation "go left vs right to same
target"). Built toy_pin_nav.py: 2D reach-to-target where instruction = lateral bow (left..right,
5 levels), same start/target, different path. Small torch flow-matching MLP + the pin applied
EXACTLY as pi0 (noise=noise-(noise@U)@U.T+(a@U)@U.T; x_t=t*noise+(1-t)*a; u_t=noise-a); Euler
sample from pinned noise. (torch has no CUDA kernels for this GPU -> run CUDA_VISIBLE_DEVICES="".)
RESULT (POSITIVE, 1500 held-out): E1' between/language=87.8% (vs LIBERO 1-15%). E2' state->c
R^2=0.077, language->c R^2=0.872, (state+language)->c R^2=0.951. E3' pass-through relerr(oracle)
=0.022; action R^2: no_pin=-74, language-prior pin=0.93, oracle=0.98. E4' fix target, swap
instruction -> path midpoint y tracks commanded bow (-0.47,-0.23,0.01,0.25,0.50); interpolating c
morphs the route smoothly/monotonically. So the pin IS a faithful continuous language steering
handle WHEN language drives the motion. Artifact: https://claude.ai/code/artifact/e4fc6a5c-1ba1-40e7-9d14-18394ea6e8b9
CONCLUSION: pin steerable-by-language iff the action depends on language. Next: find/build a
manipulation (or nav) task with genuine path-language coupling to show this at VLA scale.

## Basis gate: LDA loses, RRR wins (both cases) (2026-07-30)
Metric V_useful(U,predictor) = fraction of TOTAL action variance the pin can SET CORRECTLY from the
conditioning = sum_k Var(c_k)*R2_k / Var(a)_total (coverage x predictability). Over LIBERO single-
scene suites, with a (state+language) predictor:
  object(state-based):  PCA 16.8  LDA 0.1  RRR 19.5
  spatial(state-based): PCA 17.3  LDA 0.1  RRR 21.2
  goal(language-ish):   PCA 32.6  LDA 0.9  RRR 37.0
  all:                  PCA 19.7  LDA 0.4  RRR 24.4
LDA is USELESS as a pin basis: coverage ~0.3-1.4% (it finds discriminative directions carrying
almost no action), so despite high between-instruction ratio it sets ~none of the action. My earlier
"LDA 43% between" was 43% of a 2%-variance sliver. So the literal gate (LDA>PCA) FAILS. But RRR
(reduced-rank regression: action subspace predictable from state+language jointly, top-K of the
fitted predictor's output covariance; pin_U_rrr_k5_shared.npy) BEATS PCA in EVERY regime incl. the
old state-based examples, keeping PCA-like coverage (32-56%). And a (state+language) prior handles
BOTH cases: state-based runs off state (goal state-alone 13 vs +lang 33 => language more than
doubles useful pin content for language-ish tasks). ADOPT RRR + (state+language) prior, not LDA.
NOW RUNNING: retrain source-pin flow with RRR U (snmvp_src_pin_rrr, pi0_libero_shared, GPU0 ~2.8h,
run_rrr_srcpin.sh). Next: build (state+language)->c prior; both-cases closed-loop (state-based task
21 RRR-vs-PCA-vs-nopin; language-based goal tasks RRR+combined-vs-PCA+state-vs-nopin); then combine
with digital-twin->real. analyze_bases.py has the V_useful analysis; toy_pin_nav.py the toy positive.

## VLA both-cases result: (state+language) prior + RRR pin (2026-07-31)
Offline both-cases eval (eval_offline_lang.py) on multi-task LIBERO (goal 10-19 lang-driven, object
20-29 state-driven; 200 eps 10/task; episodes held out; language=task onehot, in-distribution).
Frozen no-delta flows: snmvp_src_pin_rrr (RRR) and snmvp_src_pin_shared (PCA). Metric subspace_R2
(does GENERATED action carry the right c: c_pred vs c_gt).
RRR flow:  goal  no_pin -0.25 / state +0.05 / state_lang +0.73 / oracle +0.99
           object no_pin -0.28 / state -0.41 / state_lang +0.52 / oracle +0.96
PCA flow:  goal state_lang +0.59 ; object state_lang +0.42  (RRR >= PCA both suites)
full_R2: state_lang is the ONLY arm positive (goal +0.30, object +0.04 for RRR); no_pin/state negative.
FINDINGS: (1) (state+language) prior handles BOTH regimes - top arm in both, only arm beating
no-pin. (2) language needed even on OBJECT (state alone -0.41): proprio state can't identify WHICH
object across tasks, the instruction names it -> "state-based" means motion is state-driven, but
task identity still needs language. (3) RRR >= PCA both suites. (4) oracle pass-through 0.96-0.99:
flow faithfully carries injected instruction at VLA scale, both regimes. CAVEATS: offline (action R2
not closed-loop), in-distribution language (onehot), episode-held-out not task-held-out. NEXT: closed-
loop confirmation; generalization (text embedding instead of onehot, held-out tasks); then digital-
twin->real. Scripts: eval_offline_lang.py, data_libero_multi/. Earlier offline RRR-vs-PCA on task0:
RRR real_prior 0.585 vs PCA 0.539 (RRR doesn't break state-based old examples).

## Gate sim/real dataset + pipeline (2026-08-01)
Real data: gate_scenes_all_no_3pov.zip (13.5G, gdown id 17M2Jk_hy-uMAezt3TDQV9LzseRBfMoib) unzipped to
~/gate_ds/gate_scenes_all_no_3pov. LeRobot v3.0, 300 eps / 79625 frames, 10fps. Task = WHICH GATE
(task_index 0-3: left/right/center-from-left/center-from-right, "hover over the stuffed animal") ->
LANGUAGE DRIVES MOTION (pin steerability sweet spot). Features observation.images.image[256],
wrist_image[256], state[7], action[7] (3pov stripped). IMAGES ARE LEGACY BGR fisheye -> swap BGR->RGB
on decode. Split (classifier: synth=MPC fixed 241/301 frames, real=variable teleop; EP 53 forced REAL
per Denis) = 100 real + 200 synthetic (left 50/50, right 50/50, center 100 synth 0 real).
CONVERTED to pi0-loadable: gate_v3_to_lerobot.py -> local/gate_nav (v2.0, BGR->RGB, keys image/
wrist_image/state/actions/task); splits gate_synth_eps.json(200)/gate_real_eps.json(100). Config
pi0_gate registered (patch_gate_config.py; copy of pi0_libero_shared repo_id=local/gate_nav, no-delta).
MASTER run_gate_all.sh (GPU1, idempotent/resumable, SSH-drop-safe): norm stats -> gate U(K=5) ->
split real 80train/20held -> train patterns gate_synth_pin (synth+pin), gate_synth_scratch (synth no
pin), gate_both_pin (synth+real_train+pin). RUNNING. Head-to-head (LIBERO pin-vs-scratch adapted
closed-loop) still on GPU0 -> hh_{pin,scratch}_t{11,21}.json. TODO: gate offline evaluator (held-out
real action R2 + language steerability swap-gate-reroute) + "synth-flow pin-only-on-real" arm (freeze
synth flow, refit prior on real). Experiments: can synth->real work; synth+real cotrain; synth-flow+
pin-only-on-real. Later: add variance to synthetic (better at generation source).

## Stronger-c-predictor sweep + MLP prior in eval (2026-08-01)
The pin's prior (state->c) was THE bottleneck (instruction lives in language/vision, not proprioceptive
state). Fix = NONLINEARITY, not vision. Eval (eval_offline_lang.py) gained an `mlp` arm (256-SiLU MLP on
[model-state, lang-onehot]->c, 3000 steps) alongside no_pin/state/state_lang(linear)/oracle. RESULT --
pin-channel subspace R2 flips NEGATIVE->POSITIVE with the MLP prior: gate_synth_pin state_lang(linear)
-0.25 -> mlp +0.44 (oracle 0.83); gate_real_pin -0.20 -> mlp +0.65 (oracle 0.97). So pass-through works;
the linear prior was the whole problem (flow faithfully carried a BAD c). full_r2 stays ~0 (open-loop,
no drone sim; realization dominates) -- subspace R2 (the pin channel) is the clean signal.
C-PREDICTOR SWEEP on held-out real (data_gate_real, episode split seed0, 5964 tr/2539 te frames, K=5):
  linear(state,lang)            0.46
  MLP(state,lang)               0.66   <- WINNER, locked in
  frozen ImageNet resnet18 +state+lang  0.54  (train_c_predictor_v2.py; vision-alone 0.54, REDUNDANT with
                                              state+lang, dilutes it; ImageNet=domain-mismatch for drone cam)
  pi0 flow-embed (embed_prefix pooled 2048) 0.556; +state+lang 0.563 (train_c_predictor_flowemb.py, GPU;
      embed_prefix=images+language only, state enters embed_suffix; masked-mean-pool; cache flowemb_cache.npz)
  temporal: state-hist(W=8)+vel 0.65; +action-hist(momentum) 0.63 (train_c_predictor_temporal.py)
CONCLUSION: ~0.66 is the FEEDFORWARD CEILING on real gate data, NOT feature-limited -- vision (ImageNet &
pi0's own), history, and momentum ALL fail to beat single-frame state+lang, richer features overfit worse.
Gap to oracle 0.97 = irreducible teleop execution variance (human pilot jitter the oracle reads off the
true future action; unforecastable). Recommendation: use MLP(state,lang) prior. Drone control = x,y,z,yaw
only (indices 0-3; dims 4-6 exactly 0 std; gripper claimed at idx7 but action is 7-dim here with no gripper
signal) -> RRR auto-excludes dead dims, U_gate energy [1.20,1.12,1.49,1.19,0,0,0], sparse control unhurt.
Recipe for #3 embed extraction: policy._input_transform(raw)->tokenizes; jax.tree.map stack batch axis;
_model.Observation.from_dict; _model.preprocess_observation(None,obs,train=False); policy._model.embed_prefix
-> (tokens[b,s,2048],mask,ar_mask); masked-mean-pool. NEXT: back to digital-twin->real training patterns.

## Digital-twin->real training-pattern comparison (2026-08-01) -- POSITIVE
KEY INSIGHT: "train on synth, pin only on real" needs NO flow finetuning -- the pin IS the prior, so
freezing the synth-trained flow and fitting the MLP prior on real = simply eval_offline_lang.py on
gate_synth_pin. So the 3 patterns are an EVAL comparison across the 3 pin checkpoints, all with the SAME
real-fit MLP(state,lang) prior (identical across checkpoints: same real train split, same U -> ~0.66
c-predictability), so differences isolate the FLOW's pass-through fidelity on real, tracked by oracle arm.
RESULT (goal(lang) suite, pin-channel subspace R2, MLP prior / oracle; held-out real, n=60 chunks, K=5):
  synth-only flow + real pin (gate_synth_pin):  0.435 / 0.832   <- DIGITAL-TWIN->REAL, frozen synth flow
  cotrain synth+real     (gate_both_pin):        0.512 / 0.954
  real-only flow         (gate_real_pin):        0.646 / 0.971  <- best (ceiling for this data)
All 3 positive & beat no_pin (-0.42/-0.26/-0.24). full_r2 ~0 (open-loop, realization-dominated; subspace
is the clean signal). CONCLUSIONS: (1) digital-twin->real HOLDS -- frozen synth flow + low-dim pin
re-grounded on real carries the real instruction at 0.44 (mechanism transfers, oracle 0.83). (2) more
real data in flow training monotonically lifts pass-through fidelity (oracle 0.83->0.95->0.97) and the
achieved channel (0.44->0.51->0.65); the PRIOR is constant so the gradient is the FLOW not the prior.
(3) both "success" questions = yes (cotrain 0.51, synth->pin-on-real 0.44). CAVEAT: offline proxy (no
drone sim -> no closed-loop for gate); n=60. Remaining: real closed-loop needs a drone; add variance to
synthetic at generation source (may lift synth-only pattern). run_gate_eval.sh 0 <ckpt>; jsons
gate_eval_gate_{synth,both,real}_pin.json (real_pin on joeec2, others ec2).

## Zero-shot sim->real: NOT FEASIBLE with current MPC twin -- execution-style gap (2026-08-01)
Extracted data_gate_synth (200 eps, EPS=gate_synth_eps.json). Diagnostics gate_zeroshot_diag.py +
gate_zeroshot_v2.py (data_gate_synth vs data_gate_real, gate norm+U, shared instr left/right):
c is NOT domain-invariant per instruction: |c_sim-c_real| mean 3.3 vs left-right sep 0.95 (ratio 3.5).
DEEPER: c is not per-instruction constant, it EVOLVES along trajectory -> language-alone R2=0.002 even
in-domain; c~f(phase,gate). In-domain MLP(progress,lang)->real=0.40, MLP(state,lang)->real=0.70 (phase
carries most). But ALL sim->real transfers NEGATIVE: Z_state(state,lang fit SIM->REAL)=-0.78, Z_prog
(progress,lang SIM->REAL)=-0.17, Z_sp=-0.73. Progress has identical [0,1] support so its failure is not
extrapolation -> c-at-a-phase genuinely differs sim<->real (per-decile shape gap 6-8 vs sep 0.95).
RULED OUT convention bug: raw action means same sign+scale-order both domains ([0.006,-0.002,0,...]).
ROOT CAUSE = execution style: real teleop action std 0.022/0.021 on x/y vs sim MPC 0.008/0.013 (~2.7x);
MPC smooth+low-mag, teleop aggressive; c projects the 50-step temporal PROFILE so smooth-ramp vs bang-
bang land near-opposite in U-space. State coverage also differs (pose mean/std diverge). So the pin
coordinate c is DOMAIN-SPECIFIC; no domain-invariant input recovers real c from a sim-fit prior. FIX is
at synthetic GENERATION SOURCE: match real action distribution (more variance, teleop-like, broader state
coverage) -> then c(sim)~c(real) at matched phase and zero-shot opens. Explains why FEW-shot digital-twin
->real works (0.44): a little real re-grounds the prior in real's execution style, which zero-shot can't.
Text embedding (task#12) still worth doing for generalization/deployable key but WON'T fix zero-shot (the
obstacle is the c-target domain gap, not language encoding).

## Text-embedding language prior (2026-08-01): FAILS for left/right minimal pair -- encoder washes it out
gate_text_embed.py: replace 2-dim one-hot with frozen all-MiniLM-L6-v2 (transformers 4.53.2, mean-pool
+ L2-norm, 384-dim; no sentence_transformers on box). MLP([state,emb])->c on real. RESULTS: (a) trained-
string held R2=0.68 (matches one-hot, no regression). (b) UNSEEN-paraphrase held R2: naive=-14028
(explodes), +paraphrase-augmentation(6 phrasings/gate, hold 2)=-2.9 (controlled but still NEGATIVE). (c)
unseen-paraphrase steering cos vs trained = 0.16 mean (inconsistent 0.38/-0.06). ROOT CAUSE (geometry):
cos(orig-left,orig-right)=0.998 -> between-gate distance 0.002, but within-gate paraphrase distance
0.15-0.27 (cos 0.71-0.87) -> PHRASING VARIATION ~100x THE LEFT/RIGHT SIGNAL. Mean-pooled sentence
embedding drowns the one discriminative word in shared content; not fixable by augmentation (task bit is
sub-threshold in encoder output). CONCLUSION: for a minimal-pair SPATIAL contrast (left/right), sentence
embedding is the WRONG key -- one-hot or a targeted keyword/token feature is better AND robust. Text
embeddings pay off only when instructions vary SEMANTICALLY (objects/goals/destinations) where the
content axis = the task axis. Deployable recommendation for gate: keep one-hot / keyword feature.

## Trajectory deep-dive: zero-shot gap is MULTI-FACTOR & fixable, not fundamental (2026-08-01)
Denis pushed back: sim/real trajectories look basically the same, so method should be robust. Diagnostics
gate_traj_compare.py, gate_traj_fix.py, gate_shape_align.py -> he's substantially RIGHT; earlier "execution-
style gap" framing was too simple. FINDINGS: (1) action = pose-delta (corr 0.98, std ratio ~1.0) in BOTH
domains -> no control-representation mismatch. (2) episode lengths EQUAL (real ~247, sim 241) -> sampling-
rate hypothesis WRONG. (3) whole-path SHAPE (progress-resampled pose) is domain-INVARIANT: within-
instruction cross-domain gap ~0.5x left-right separation, all alignments (raw/start-relative/affine) -> sim-
left closer to real-left than to right paths. BUT the pin's c has domain/task ratio 2.7-3.5 (domain-
DOMINATED) and every sim->real transfer is NEGATIVE (Z_state -1.7, Z_prog -0.49; progress-displacement-c
Z_prog -0.49). Removing domain mean-offset (|mean_sim-mean_real|=1.94) only partly helps (Z_prog -0.49->
-0.23, still neg) -> NOT a single offset. THREE compounding, individually-fixable factors: (A) teleop JITTER
- real per-step delta std ~2.7x sim (high-freq, cancels over route but corrupts the 50-step chunk c) -> fix:
low-pass/integrated displacement c. (B) genuine MID-PATH route diff - at progress 0.5 sim x~1.82 vs real
x~1.05 (MPC arcs wider than human) -> fix at generator: match human route (more specific than "add
variance"). (C) domain-SENSITIVE pin basis - raw shape ratio 0.5 but c ratio 2.7 because variance-max
RRR/PCA U latches onto sim-vs-real variance -> fix ours: build U domain-INVARIANT (capture instruction
axis, project out domain axis). Zero-shot needs A+C (ours) + B (generation); no single fix flips it positive.
Text-embed translation options (Denis's read confirmed): structured slots {direction,target} > discriminative-
span embedding > learned instruction encoder -> c-relevant features. Scripts in $RD.

## VLM-grounded GENERAL pin: internal VLM defines c + generalizes to UNSEEN tasks (2026-08-01)
Denis: use pi0's internal VLM for task-general instructions (slots don't generalize); build c to generalize
beyond drone. RECIPE (vlm_rrr_libero.py): keep c=U^T a (real action coord, keeps binding advantage) but
DEFINE U as the VLM-predictable action subspace -- RRR with pi0's PaliGemma representation as predictor
(U=top-K eigvecs of Cov(Yhat), Yhat=OLS(VLM_feat -> flat normalized action chunk 1600-dim)); prior=MLP(VLM)
->c. No hand-crafted onehot. Feature backends: prefix=pre-fusion embed_prefix mean-pool; context=fused post-
transformer prefix via policy._model.PaliGemma.llm([tokens,None],mask=make_attn_mask(mask,ar_mask),positions=
cumsum(mask,1)-1) -> outputs[0] [b,s,2048] mean-pool (agent-verified call). Testbed data_libero_multi (20
tasks; goal 10-19 lang-driven, object 20-29 state-driven; ckpt snmvp_src_pin_rrr/4999, norm_shared_libero,
3483 frames stride8, cached vlm_feat_{prefix,context}.npz). RESULTS (held c-R^2 goal/object/all):
  IN-DISTRIBUTION (all tasks in train): VLM prefix .833/.786/.823, context .829/.783/.820, state+onehot
    .831/.727/.808 -> VLM ties on goal, BEATS on object, NO onehot; fusion doesn't add (memorization suffices).
  TASK-HELDOUT (tasks 18,19,28,29 unseen): VLM context +.063/+.656/+.400, prefix -.167/+.700/+.303,
    state+onehot -.826/-4.619/-1.905. => onehot COLLAPSES on unseen tasks (can't generalize, no learned
    column); VLM GENERALIZES (+0.40 on brand-new tasks), strong on state-driven object (+0.66), modest on
    unseen language goals (+0.06). FUSION HELPS out-of-distribution (context>prefix, esp unseen lang goals
    -.17->+.06) though not in-distribution -> use the internal CONTEXTUALIZED VLM, justified for generalization.
CONCLUSION: VLM-grounded pin = task-general instruction mechanism; defining c as VLM-predictable action
subspace makes coord+prior transfer to new scenarios. CAVEATS: offline c-R^2 not closed-loop; 4 held tasks
n~300-400/suite; unseen-language-goal generalization is the frontier (+0.06).
Saved pin_U_vlm_ctx_k5_shared.npy (1600,5). LAUNCHED closed-loop retrain snmvp_src_pin_vlm (pi0_libero_shared,
SNMVP_PIN_U=that, 5000 steps GPU0, log src_pin_vlm.log) -- pin enabled confirmed. NEXT after: closed-loop
steerability eval (reuse run_hh_eval w/ U=vlm); VLM-prior-at-serve for unseen-task generalization needs serve
infra (extract VLM feat + prior head at serve).

## DRONE sim->real with generalist pin (2026-08-01): near break-even zero-shot; blocker = route mismatch
vlm_sim2real_drone.py (pi0_gate, gate_synth_pin/4999 VLM, data_gate_synth+real left/right, U built on SIM,
zero-shot=prior fit SIM eval REAL; cache drone_vlm_feat.npz; contextualized VLM feat + smoothed-c (low-pass
window7) + VLM-RRR U). RESULTS (zero-shot / few-shot c-R^2, c-domain-ratio):
  state/raw/state (failed baseline):  -0.826 / +0.673  ratio 3.87
  vlm/raw/vlm (C only):               -0.272 / +0.613  ratio 4.05
  state/smooth/state (A only):        -0.871 / +0.671  ratio 3.87
  vlm/smooth/vlm (A+C combined):      -0.123 / +0.590  ratio 4.04  <- best zero-shot, ~break-even
  vlm/smooth/state (A+C basis,state prior): -2.246 / +0.648        <- prior is what matters
DIAGNOSIS: VLM grounding lifts zero-shot -0.83 -> -0.12 (near break-even) but does NOT cross zero. The win is
the VLM PRIOR (grounds real obs like sim obs, so it transfers), NOT the basis (VLM-RRR ratio 4.05 >= state
3.87 -> NOT more domain-invariant; the VLM sees different sim/real images). c-domain-ratio stays ~4 in ALL
rows => real c genuinely differs from sim c (routes differ); a perfect prior predicting sim-c is still off by
the domain gap. So zero-shot is capped by FACTOR B (MPC twin flies different route than human teleop), a
GENERATION-side fix. Factor A (smoothing) marginal (-0.27->-0.12); factor C (VLM basis invariance) did NOT
pan out. Few-shot solid ~0.6 everywhere. BOTTOM LINE: generalist VLM pin is the right mechanism (near break-
even) but zero-shot needs the synthetic twin to fly real-like ROUTES; few-shot works today. NEXT: match twin
routes at generation source -> then zero-shot should cross positive.

## Domain-invariant U -- solve sim->real on OUR side, no generation access (2026-08-01)
Denis reframed (CORRECT): U should capture the shared SUBSTANCE of the action (translates sim<->real), the
domain gap is about HOW U is built not the data; assume no generation access. gate_invariant_U.py (cached
drone_vlm_feat.npz + smoothed action chunks; real used ONLY to align subspace, zero-shot prior fit on SIM).
Methods & zero-shot c-R^2 / c-domain-ratio / L-R sep:
  base (variance PCA):        -0.19 / 4.06 / 0.80
  proj_domain (remove dom dirs): -0.27 / 4.23 / 0.49
  gen_eig (max instr/domain scatter): -0.00 / 1.04 / 0.08  <- achieves invariance but KILLS instruction
  base + affine sim->real translate:  +0.04 / 4.06 / 0.80  <- CROSSES POSITIVE (marginal, +-0.05 noise)
  disp (route = cumsum actions, "where it goes"): -0.08 / 2.71 / 0.44  <- halves domain gap, KEEPS instruction
  disp + affineT: -5.99 (affine unstable on noisier disp binned means)
KEY FINDINGS: (1) it IS how U is built -- route/displacement basis captures invariant substance (ratio 4.06
->2.71, keeps L-R sep) vs variance basis latching onto domain. (2) In RAW action chunk, instruction & domain
are ENTANGLED + domain dominates 4:1 -> gen_eig can zero the domain only by zeroing instruction (sep 0.08).
Local action window = execution texture (domain); integrated route = instruction (whole-path shape ratio was
0.5 earlier). (3) A minimal-effort AFFINE sim->real translation on c (learned once from matched instr+phase
pairs, a little real data; GLOBAL so generalizes across instructions) bridges zero-shot to break-even/+0.04
WITHOUT generation fix. Linear methods CAP at break-even; residual sim<->real diff is nonlinear/instruction-
dependent. RECIPE (our-side sim->real): route/displacement U + learned sim->real c-translation. NEXT: NONLINEAR
translation (small MLP sim-c->real-c on a few more matched pairs, still few-shot, learned once) = path to
robustly-positive zero-shot; no generation access needed. Supersedes earlier "must fix generation-side".

## Nonlinear sim->real translation + scalability (2026-08-01)
gate_nonlinear_translate.py: prior P fit on SIM (VLM->sim-c); translation T maps P(VLM(real))=sim-c -> real-c,
T learned from a little real data. Base PCA c (K=5). RESULTS (zero-shot c-R^2):
  IN-DISTRIBUTION (T fit on 70% real both instr): P-alone -0.24 | P+affine +0.35 | P+MLP +0.54 | few-shot 0.60
  CROSS-INSTR fit LEFT->test RIGHT: P-alone -0.32 | affine -0.43 | MLP -0.67 | few-shot -0.72
  CROSS-INSTR fit RIGHT->test LEFT: P-alone -0.08 | affine +0.19 | MLP -0.09 | few-shot -1.36
FINDINGS: (1) IN-DIST nonlinear MLP translation ~RECOVERS few-shot (+0.54 vs 0.60) >> affine 0.35 >> P-alone
-0.24 => sim->real shift on c is NONLINEAR and LEARNABLE; for any task with a little real data, route-U + sim
prior + MLP translation ~= few-shot. (2) Does NOT transfer cross-instruction -- BUT few-shot ALSO fails cross-
instr (-0.72,-1.36) => not a translation weakness; two OPPOSITE tasks (left/right) can't teach each other's
domain shift, no diversity to generalize from. MLP overfits its one task; lower-capacity affine transferred
partially one direction (+0.19) (the tell). SCALING DESIGN (answer to "scale across tasks/embodiments"):
(a) c = SHARED semantic coord across embodiments (VLM-grounded route/substance) so 'left turn' ~ same c on
drone or arm; (b) translation CONDITIONED on task/embodiment context T(sim-c, VLM-context)->real-c, NOT global
(shift is task-dependent) NOR context-free-per-task (needs real every task); (c) META-TRAIN T across many
diverse (sim,real) task pairs -> learns sim->real gap as fn of context -> generalizes to a NEW task's sim
zero-shot. 2-task drone CANNOT show cross-task generalization -> OXE now genuinely motivated as the testbed
(not breadth-for-its-own-sake) where the context-conditioned translation can be learned & its cross-task
transfer demonstrated. Path: VLM-grounded route c (shared coord) + VLM-context-conditioned sim->real
translation meta-trained on multi-embodiment corpus.

## "vla^2" / little-OT-flow reconstruction + OXE setup (2026-08-01)
Denis's framing: adapt to a NEW EMBODIMENT by learning one small OPTIMAL-TRANSPORT flow that moves its c-
distribution into the shared coord; big flow-matching VLA stays frozen. It's flow-matching on the pin
(source) coordinate of a flow-matching VLA = flow-on-flow. Reconstructed as conditional flow-matching
transport v(c_t,t,VLM-ctx): x0=sim-prior-guess -> x1=real-c (gate_flow_transport.py). DRONE RESULT (base c):
in-dist P-alone -0.31 | P+MLP +0.51 | P+flow +0.40 | few-shot 0.60; cross-instr all worse (flow -4.65/-1.42).
=> in the PAIRED regime the flow does NOT beat the MLP (flows model DISTRIBUTIONS, regressors win at POINT
estimates; Euler integ compounds OOD error). Cross-task fails for ALL methods (2 opposite tasks = no
diversity). KEY INSIGHT: OT/flow's value is the UNPAIRED regime (new embodiment: sim & real distributions,
NO per-sample correspondence -> can't regress, OT/flow is the ONLY tool). Drone is PAIRED (has real-c labels)
so can't show it. => build the little OT flow as UNPAIRED c-distribution alignment on OXE, not drone.
OXE STATUS: oxe_extract_full.py (adds image+language to shared 6-D EE-delta), all 4 embodiments extracted
full-obs: data_oxe_full/{bridge,berkeley_autolab_ur5,toto,viola}.npz (bridge 361 frames 111 uniq langs).
LIBERO VLM-RRR retrain DONE (snmvp_src_pin_vlm/4999) -> NEXT run closed-loop steerability eval. Scope doc:
OXE_META_TRANSLATION_SCOPE.md. NEXT: OXE VLM features (GPU) -> shared-c coherence across embodiments (ladder
step1) -> unpaired OT-flow alignment.

## OXE ladder overnight results (2026-08-02) -- little-transport vision SUPPORTED
(Ran autonomously; VPN dropped mid-run, jobs survived under nohup/setsid.) OXE VLM feats cached for 4
embodiments (bridge 361, ur5 2753, toto 9638, viola 8425). Shared VLM-RRR U (K=5) on 4 robots, 6-D EE-delta
per-embodiment RMS-normed. STEP1 shared-c coherence (oxe_shared_c.py): (a) POOLED prior held R2 within-
embodiment: toto +0.94, viola +0.68, ur5 +0.46, bridge -0.26 (bridge thin/diverse). (b) HELD-OUT EMBODIMENT
zero-shot (prior on other 3): ALL NEG (bridge -1.27, ur5 -0.19, toto -0.20, viola -0.15) -> one shared prior
does NOT transfer to a new robot zero-shot -> motivates the little adapter. NEW-EMBODIMENT ADAPTATION (the
vision test, oxe_new_embodiment.py): little adapter [P_ref-guess, VLM-ctx]->c, data-efficiency no-adapt/50/
100/200/UB: toto -0.19/+0.38/+0.46/+0.58/0.97; viola -0.20/-0.13/+0.17/+0.31/0.68; ur5 -0.30/-0.15/+0.02/
+0.10/0.48; bridge -1.24/-0.02/../../0.06(barely predictable, thin). => LITTLE TRANSPORT WORKS: 50-200
samples flips a NEW embodiment from neg to positive, climbing toward its own ceiling; big VLA/subspace frozen.
Supports "learn new embodiment = fit little transport" (cheap few-shot, not zero-shot). CAVEAT: adapter is
paired few-shot MLP (have new-embodiment c labels); unpaired OT version still to test. CLOSED-LOOP hhv:
adapt trainings succeeded (prior R2=0.65) but serve hit /2000-vs-/1999 path bug (SAME bug as before) ->
SERVE_FAIL; ckpts exist at /1999; run_hhv_eval.sh (sed of run_hh_eval.sh, /1999 + VLM U) re-serving.
bridge RE-EXTRACTED big: data_oxe_full/bridge.npz now 2848 frames / 687 uniq langs (overwrote 361 ver);
refreshing vlm_feat_oxe_bridge.npz (OXE_DSS env added to oxe_vlm_feat.py). NEXT: closed-loop number; re-run
ladder with diverse bridge (cross-TASK stress); unpaired OT-flow adapter.

## Closed-loop + diverse-bridge + OT-flow + soft-pin (2026-08-02)
CLOSED-LOOP VLM-RRR pin (run_hhv_eval.sh, /1999 fix, few-shot adapted, held-out tasks): task11(goal) pin
0.40 vs scratch 0.47; task21(object) pin 0.00 vs scratch 0.80. => HARD PIN NEGATIVE closed-loop (<=scratch,
0.0 on object) -- hostage-to-prior: hard-clamp of low-K subspace to imperfect prior (R2 0.65) locks in error,
closed-loop compounds it. Offline c-R2 gains (0.82, unseen-task generalization) DO NOT translate to closed-
loop under a hard clamp. DIVERSE-BRIDGE ladder (bridge 2848 frames/687 tasks) CONFIRMS overnight: adapter
toto -0.15/+0.34/+0.46/+0.56/UB0.96, viola/ur5 similar; bridge still hard outlier (own UB 0.06 even at 2848
frames -> WidowX 6-D chunk ~unpredictable from VLM, NOT a data-quantity issue). OT-FLOW ADAPTER
(oxe_ot_adapter.py, held toto/viola/ur5): paired FLOW works but WORSE than MLP (toto 0.48 vs 0.58, viola 0.17
vs 0.33, ur5 neg); FLOW-UNPAIRED (shuffled coupling) CATASTROPHIC (-4 to -8). => vla^2 unpaired transport
does NOT pan out on real cross-embodiment; regressor beats flow at point-estimating c from paired data;
unpaired distribution-alignment underdetermined w/ ~50-200 samples + non-bridging cross-embodiment ctx.
(caveat: used random-shuffle coupling worst-case; minibatch-OT coupling untested but paired flow already no
gain.) DISTILLATION: VLM-grounded shared c + CHEAP PAIRED MLP ADAPTER = the working new-embodiment recipe;
flow/OT elegant but empirically unjustified here. SOFT PIN: patched pi0.py (env SNMVP_PIN_NOISE, backward-
compat, pi0.py.presoft_bak) -- inject c+sigma*std*eps during training so flow learns to CORRECT prior error
from obs (not pass-through). Retraining snmvp_src_pin_vlm_soft (sigma=0.7, GPU0, src_pin_vlm_soft.log) ->
closed-loop pending. Targets the hard-pin closed-loop failure.

## Pairing at scale: SEMANTIC coupling recovers the correspondence (2026-08-02)
Denis: unpaired OT fails because MLP has paired supervision; how to find pairing at scale? ANSWER (gate_
lang_ot.py, drone sim->real, sim&real unpaired diff-episodes but share instr left/right): train conditional
flow transporting sim-guess x0=P_sim(VLM(real_obs)) -> real-c x1 under couplings. R2: MLP-paired 0.61 (upper);
flow FREE(true VLM pairing) 0.405; flow LANG+CTX(same instr, nearest ctx) 0.400 ~= FREE; flow LANG-only 0.294;
flow RANDOM 0.310. => SEMANTIC PAIRING RECOVERS TRUE CORRESPONDENCE: couple on shared semantic keys (language
=task, VLM-context=phase) and it stands in for per-sample pairing. The VLM's language+context grounding IS
the pairing function. Instruction ALONE too coarse (0.29, phase dominates c variance -> need ctx/phase
refinement); at scale w/ richer language, language level carries more. Random not catastrophic on drone (0.31)
because ctx alone predicts c ~0.6 so flow ignores x0 -- coupling matters MOST when ctx weak (=cross-embodiment,
why OXE random collapsed -4..-8). Flow still < MLP (point-estimation) but coupling comparison is the point.
SCALING RECIPE for the pairing: OT-couple on shared semantic rep (language for task + context for phase) ->
extends the free VLM pairing to label-scarce/cross-embodiment where only distributions+language are available.
DECISION (Denis, 2026-08-02): for the DRONE task use the PAIRED MLP adapter (free VLM pairing, works, cheap);
keep the "context connector" (semantic language+context OT-coupling to find correspondence) for FUTURE plans
at scale / label-scarce / cross-embodiment. I.e. MLP now, semantic-coupling transport later.

## Soft pin closed-loop + falsify SIM on ec2 + RENDER-GAP proven (2026-08-03)
SOFT PIN closed-loop (run_hh_soft.sh, sigma=0.7 train-time noise on c so flow CORRECTS prior error): held-out
LIBERO success task11(goal) pin 0.67 vs scratch 0.87; task21(object) pin 0.87 == scratch 0.87. vs HARD pin
0.40/0.00. => soft pin FIXES the hostage-to-prior catastrophe (0.00->0.87 object), closed-loop-safe (matches
scratch object, small goal gap). Soft pin is the deployable pin.
HF DEPLOY done earlier: denis-liu-tri/gate-drone-pi0 (gate_both_scratch/both_pin/synth_scratch + assets +
gate_inference.py). Plug-and-play gaps: (a) register pi0_gate (5 lines, not in base openpi); (b) PIN mode
needs policy.infer(noise=) patch (policies/policy.py + models/pi0.py) — not base openpi.
FALSIFY SIM ON EC2 (denisfliu/falsify-pi, branch gate-pi0-sim-integration): full env UNPORTABLE to Blackwell
(nerfstudio1.1.0/tinycudann/gsplat0.1.13/acados/sagesplat = torch2.1/CUDA12.0-era, sm_120-hostile). BUT
reproduced the RENDER standalone: torch2.11+cu128 (sm_120 works; old torch failed "no kernel image"), gsplat
1.5.3 JIT-builds+rasterizes on sm_120 (CUDA-toolkit-12-8 via apt, sudo OK), sagesplat ckpt (6.1M gauss) loads.
Standalone renderer validated VISUALLY (coherent gate-lab render from start pose). Frame spec from sim-box
agent: gaussians RAW in NS frame (means/quats raw; scales=exp; opac=sigmoid; colors=cat(fdc,frest)->SH deg3;
gsplat SH->RGB); camera Tw2g(NED->NS) 4x4 dataparser_scale 0.1261; viewmat=world(NS)->cam OpenCV via
get_viewmat(R*[1,-1,-1]); K carl_dual fwd; bg composite [0.149,0.165,0.216]; render 1024x768 -> resize 224
BILINEAR RGB. Repro script /tmp/gate_render.py. RENDER-GAP TEST (controlled: same state+prompt, only image
varies, gate_both_scratch): REAL photo -> x net +4.6cm span 5.1cm; SIM render -> x net +0.1cm span 0.3cm
(TIMID/hover); action corr(real,sim)=0.21. => sim hovering is a RENDER-FIDELITY domain gap (gsplat floaters/
lighting OOD for the VLM), NOT integration/model (model flies on real images). Levers: improve splat fidelity/
deflicker/floater-mask (sim side), OR fine-tune policy on sim renders (doable on ec2: render poses + train
actions). Scripts: /tmp/render_gap_test.py. NEXT: close render gap (render-domain fine-tune or fidelity).

## CORRECTION + ec2 closed-loop rollout: creep was INTEGRATION, not render (2026-08-03)
Denis pushed back: synth data came FROM the sim, so model already trained on sim renders -> fine-tuning on
renders is circular. CORRECT. Re-tested: model is NOT render-OOD. With a synth-consistent state, policy
commands forward on the gsplat render (x net +0.036/50-step chunk ~ synth GT +0.022; corr(synth-img,render)
action = 0.73). Earlier "timid render" was a state<->image MISMATCH (real state + render). State convention
also fine: [px,py,pz,yaw,0,0,0] mocap, sim sends same, synth starts ~[0,0,1.5,0]. Built the FULL rollout on
ec2 (no sim box): serve_gate.py (openpi/JAX venv) websocket server + standalone render-client (tv venv:
torch2.11+gsplat1.5.3). pose_to_viewmat(pos,yaw) validated vs reference (diff 3e-9): pos_ned=(x,-y,-z),
Tnb=[Rz(yaw)|pos_ned], Tbody_cam_fwd=[[0,0,-1,.10],[1,0,0,-.03],[0,-1,0,-.01]], c2w_ns=Tw2g@Tnb@Tbc, viewmat=
get_viewmat. Integrator = vla.py absolute+reanchor. ROLLOUT SWEEP (left_gate, gate@x=0.86): apc=8 reanchor
(SIM DEFAULT) creeps to x=0.06 (reproduces "barely moves"); apc=25 ->0.21; apc=50 ->0.24 (40 chunks); apc=50
EXTENDED 150 chunks -> x 0->0.64 (74% to gate) steady flight then plateau ~0.6; raw-absolute (no reanchor)
STUCK ~0 (model abs targets hover, reanchor REQUIRED). => THE BOTTLENECK IS actions-per-chunk=8: it realizes
~1/8 of each chunk so the drone creeps & never leaves slow-start. FIX: apc~50 -> drone FLIES toward gate
(0.06->0.64). Model+render+state all fine. Remaining: plateau at ~0.6 short of gate 0.86 (model caution near
gate plane / reanchor damping) = secondary. Scripts: /tmp/gate_render.py, /tmp/gate_rollout.py; serve ~/serve
_gate.log :8777. ec2 sim stack: torch2.11+cu128 (sm_120), gsplat1.5.3, CUDA-toolkit-12-8, /tmp/tv venv.

## RESOLVED: actions are per-step DELTAS; wrong integration caused the creep (2026-08-03)
Denis: model has tons of examples of this trajectory, shouldn't fail. RIGHT. (1) TEACHER-FORCED replay (feed
model the synth TRAINING images+states along a full gate flight): model output tracks GT per-step actions
almost exactly the whole way -> model CAN do the task. (2) DECODED the action rep (/tmp/decode_action.py):
actions are PER-STEP DELTAS in ALL dims; CUMSUM(action[:, :3]) ~= actual Δpose to ~1cm at every t (t=0 cumsum
[0.624,0.324,0.126] vs actual [0.646,0.331,0.124]; t=50 [1.24,0.244,-0.20] vs [1.242,0.239,-0.202]). z is a
DELTA (~0.003), NOT the "1.53 absolute" the handoff/README claimed. (3) => the sim's action_space="absolute"
+ re-anchor was WRONG: it realizes last_delta-first_delta (~0.02) instead of cumsum (~0.6-1.2/chunk) = ~50x
too small = THE CREEP. (4) FIXED integration to delta/cumsum (pos+=act[:n,:3].sum(0)) and re-ran ec2 rollout:
apc=8 delta -> x 0->0.28->0.62->0.92->1.23 THROUGH THE GATE (0.86) toward goal(1.525); apc=50 -> 2.6 (overshoot
like training fly-out to 2.79). DRONE FLIES. Model/render/state all fine. SIM FIX: use --action-space delta
(cumsum), NOT absolute; apc=8 good (smooth), apc=50 overshoots. README's "delta blew up z 1.5->78->3983" was a
misdiagnosis (raw served deltas clean; blow-up was a separate serve/norm bug, likely the 7->32 norm-stats
padding corrupting z at inference). NET: gate task SOLVED in sim on ec2; the whole "hover" saga was action-
representation misread. Scripts /tmp/teacher_forced.py, /tmp/decode_action.py, /tmp/gate_rollout_delta.py.

## PIN STEERING in flying sim rollout — WORKS (2026-08-03)
With correct delta integration, ran pin(serve_gate_pin :8778, gate_both_pin + prior + U) vs scratch(:8777,
gate_both_scratch) x left/right prompt in the left_gate scene (gate y=+0.69), tracking lateral y. End y:
scratch LEFT +0.63 / RIGHT +0.27 (Δy=0.36, both +y toward the visible left gate); pin LEFT +0.01 (peaks +0.77
mid-flight) / pin RIGHT -1.97 (veers hard -y, away from the only gate in view). INSTRUCTION-STEERING |y_L-y_R|:
PIN 1.98 vs SCRATCH 0.36 => the source-noise PIN steers ~5.5x harder on the left/right instruction and even
OVERRIDES the visual scene when told "right" (counterfactual in the left-only scene), while scratch just
follows the visible gate regardless of prompt. Both pin conditions still fly forward through the gate plane
(x~1.5-1.9). => CORE SOURCE-NOISE-PIN THESIS DEMONSTRATED END-TO-END IN A FLYING SIM: pinned instruction
coordinate binds far harder than language conditioning. Caveats: left-only scene (right=counterfactual, makes
it stronger); pin_LEFT wobbles (0.77->0.01) = some trajectory instability to tighten. Script /tmp/gate_steer.py.

## Videos + CLE scoring (2026-08-03)
Rendered 4 forward-cam flythrough videos (pin/scratch x left/right) + top-down trajectory plot -> Artifact
"pin-steering-sim" (claude.ai/code/artifact/059c14e9...). Scripts /tmp/gate_video.py, build_art.py, pin_steer
_sim.html. CLE: copied falsify cem/scorer.py logic (gate-plane aperture transit + goal distance) into /tmp/
gate_cle.py (gate center [0.86,0.69,1.5] normal [0.749,0.663,0] aperture 0.45m; goal [1.525,-0.615,1.0]).
RESULTS (28-chunk / 50-chunk): scratch_left PASS gate (aperture miss 0.04); pin_left PASS (miss 0.34->0.15
longer); *_right MISS (pin veers off correctly - no right gate in left scene). BUT goal NOT reached by any
(min dist 1.2-1.7, thr 0.4). => gate TRANSIT works (left); FULL task (through-gate-then-hover-over-goal at
-y) incomplete: closed-loop does the +y through-gate leg but not the return-to-goal (-y) leg. Not a rollout-
length issue (50 chunks still 1.22). Likely CLOSED-LOOP DRIFT past the gate (high-x renders/states OOD ->
return command degrades). NEXT: teacher-force at LATE trajectory points (does model command the return-to-goal
on training states/images? isolates drift vs model); tighten pin; then re-score. Servers ~/serve_gate.log:8777
(scratch), ~/serve_pin.log:8778 (pin), both openpi venv, gate_both_scratch / gate_both_pin.

## CLE multi-seed + long-horizon instability (2026-08-03) — CORRECTS earlier single-run claims
Rollouts are STOCHASTIC (pi0 samples fresh source noise per infer; server not seeded) so single-run CLE
is unreliable. Ran 10 rollouts/cond at nch=200 (=1600 integ steps, gate_cle_dump.py SNCH/SNRUNS env,
/tmp/cle_multi_n200.json). RESULT [gate transit% / goal<0.4%]: pin_left 100/10, scratch_left 100/0,
pin_right 0/0, scratch_right 40/0. Aperture misses tight both (0.03-0.30). KEY: endpoint x-spread reveals
the PIN IS CLOSED-LOOP UNSTABLE over long horizons: pin_left end-x std 2959 (range -10174..+1.0), pin_right
-36219..-1861 (ALL diverge). scratch bounded (end-x 2..13, std 1.6-4.4). So gate transit (happens early
~step 640) is robust + correctly instruction-conditioned (pin 100%L/0%R vs scratch 100%L/40%R = pin OBEYS
left/right, scratch largely ignores it), but the pin blows up to |x|~1e4 in most long rollouts -> full task
(through-gate-then-return-and-hover) essentially UNSOLVED (pin_left 10% = 1 lucky stable draw; the 0.17
"success" I reported first was n=1 noise). At nch=50 the pin hadn't diverged yet -> shorter runs looked fine;
longer runs EXPOSED the instability (matches earlier "pin_LEFT wobbles" note). NEXT (push perf) = stabilize
long-horizon pin: action-sample AVERAGING at inference (draw K noise, mean the chunk; cf. earlier closed-loop
MEAN=0.933), lower flow temperature / more integ steps, or the soft-pin (may be more stable than hard pin).
Videos gate_video_long.py (nch=200, 4-stride) ALSO diverge (pin_left end -3230) confirming stochastic
instability, not a script bug (cle_dump + video drive integration identically).

## Teacher-forced pin eval — pin REPRODUCES the demo; divergence is COMPOUNDING FEEDBACK, not the pin (2026-08-03)
User reframe: "the pin makes it do the same thing it does from the start; care about PIN perf not scratch."
Correct. Demo data = data_gate_synth/ep_*.npz (image/wrist 256^2, state 7, action 7 per-step deltas, 301
steps; train eps = indices 100-199 per gate_synth_eps.json, held-out 0-99). gate_teacher_force.py feeds the
pin server (8778) the GROUND-TRUTH demo image[t]+state[t] each chunk and re-anchors to GT state (removes
render + compounding confound). RESULT: pin reproduces the trajectory to a few cm, LOW variance across 3
stochastic draws (agree to the mm): ep_0150(train) per-chunk ADE 0.069 m, TF min-goal 0.04 (GT 0.03);
ep_0042(held-out) ADE 0.084 m, TF min-goal 0.02 (GT 0.01). So on-distribution the pin flies the WHOLE task
(through gate + return + hover over goal to 2-4 cm) on train AND held-out. => the closed-loop blow-up to
|x|~1e4 is COMPOUNDING/OOD FEEDBACK (integrated action drifts ~0.07 m/chunk off-manifold -> state prior
evaluated OOD -> extrapolates large wrong c -> runaway), NOT a pin capability or stochastic-variance failure.
Pin performance is essentially perfect; the unstable thing is the closed-loop feedback path. NEXT (push perf)
= stabilize feedback to stay near the demo manifold: sample-averaging per step, shorter replan horizon, clamp
c / gate the prior off-manifold, or accurate state feedback (real drone has odometry, so real closed loop may
not drift like the sim integration). tf trajectories saved /tmp/tf_gate.json.

## CLE stabilization: averaging fails, WORKSPACE CLAMP fixes divergence (2026-08-03)
Two CLE experiments on the gate pin (10/8 seeds, nch=200, pin_left):
(1) SAMPLE-AVERAGING (client averages K flow samples/step, same c): K=1/4/8/16 -> gate 100%, goal 0%,
    endx_std 800-1500 ALL. Averaging does NOT help -> the divergence is NOT stochastic noise.
(2) DIVERGENCE DIAGNOSTIC (gate_diag.py, /tmp/gate_diag.json) 3 modes for the state driving the loop:
    base (dead-reckoned [pos,-yaw]): gate100 goal25 endx_std 1872 (diverges to |x|~1e4).
    clamp (clip pos to left-demo bbox lo=[-0.25,-1.06,0.74] hi=[3.08,0.88,1.98] before render+integrate):
      gate100 goal25 endx_std 0.6 -> DIVERGENCE ELIMINATED, task perf unchanged.
    anchor (feed prior nearest-demo state): gate0 goal0 -> BACKFIRES: replacing the state desyncs it from
      the rendered image; the flow needs state+image CONSISTENT (so can't fix by swapping in a clean state).
  => the blow-up is a SPATIAL RUNAWAY (drone leaves demo region -> OOD image+state -> pushed further out).
  A workspace clamp (general, legitimate deployment constraint) breaks the loop. STABILITY SOLVED.
  REMAINING GAP = completion: even clamped, goal-reach is only 25% (threads gate 100% but completes the
  return-to-goal-and-hover 1/4 of stochastic rollouts). NEXT: clamp+averaging sweep RUNNING (CLAMP=1
  SKS=1,8,16, gate_cle_avg.py -> /tmp/cle_clampavg.json) to see if variance reduction lifts completion now
  that divergence no longer masks it.
HISTORY PRIOR still blocked: action norm stats are 7-D (state 7-D too); prior uses onehot(L/R). My manual
c-target (pad->32, normalize w/ padded stats, @U(1600,5)) gives Hc=1 held c-R2 NEGATIVE — needs rework via
the data-loader c (make_prior-style, guaranteed correct) with per-frame onehot. Deferred pending clamp+avg.

## Pin c is 99% STATE-driven; divergence = brittle state->c map (2026-08-03)
Denis asked: is CLE failing because the pin only depends on the language instruction, not the state?
MEASURED (probe_pin_state.py, run JAX_PLATFORMS=cpu): NO — the opposite. Along a left demo, prior c=MLP([ms,onehot]):
c(left) std along traj per-dim [6.2,6.1,2.8,7.4,2.8], range ~[20,20,11,24,10]; flipping lang shifts c only
[1.0,0.8,2.9,0.9,1.7]. RMS state-driven 11.4 vs lang-driven 4.3 => STATE/LANG ratio 2.64; variance decomposition
STATE-fraction = 0.99 (lang ~1%). So the pin IS strong state feedback (why teacher-forcing tracks the whole
out-and-back to cm). The CLE failure is that this state->c point-estimate MLP is BRITTLE off-distribution: a
small drift -> OOD state -> MLP extrapolates -> large wrong c (c swings ~20/dim) -> drives further off ->
runaway. The workspace clamp works only because it keeps the state where the map is valid (symptom fix).
REAL FIX (general, no bounds): retrain the prior with INPUT-NOISE AUGMENTATION so the state->c map is robust
(drifted state -> sensible c that pulls back). PREREQ: fix c-target via data-loader path (manual pad+normalize
gave negative held R2; note action norm stats are 7-D not 32-D). History-conditioning optional; augmentation is
the active ingredient. Hypothesis (language-domination) REFUTED with data.

## ROBUST PRIOR FIXES CLE DIVERGENCE (2026-08-03)
c-target bug was the SPLIT not the targets: index-split (train 100-199/held 0-99) = OOD -> neg R2. RANDOM
split + std norm (confirmed via data loader: normalized-action std~1 => mean/std not quantile) ->
make_hist_prior.py gives c-R2 0.96-0.97, history helps modestly (Hc1 0.961 -> Hc8 0.974), input-noise
aug (AUG=0.15) doesn't hurt. Trained /tmp/hist_prior_gate.pt (Hc=8). Served serve_gate_pin_hist.py on 8781
(GPU1, launch_hist_server.sh auto-reads ckpt/norm/U from the 8778 pin server). CLE (10 seeds nch=200, NO
clamp, gate_cle_hist.py -> /tmp/cle_hist_robust.json): left gate100 goal20 goalmed 0.52 endx_std 1.5;
right gate0 endx_std 23. => the input-noise-robust state->c map ELIMINATES the runaway generally (endx_std
1872 -> 1.5, no workspace clamp) and halves goal-median (1.17 -> 0.52). Confirms divergence = brittle
state->c map. REMAINING GAP = final hover precision: reliably within ~0.5m but <0.4m only 20%. Averaging
still dead (clamp+avg K8 goal0 vs K1 goal10). NEXT: push precision on the now-stable pin (robust+avg via
server --samples, robust+clamp, Hc/AUG tune). serve_gate_pin_hist supports --samples (server-side avg,
buffer appended once/step so history stays correct).

## AUG sweep PROVES augmentation is the causal stability lever (2026-08-03)
Retrained prior at AUG=0.05 (sharper: c-R2 0.982 vs 0.974 at 0.15). CLE (10 seeds, no clamp, 8783):
left gate50 goal0 goalmed1.65 endx_std 1.67e7 (!!); right endx_std 9.3e6. => LESS augmentation = BRITTLE =
CATASTROPHIC DIVERGENCE (~1e7, ~10000x baseline), despite better on-distribution fit. Confirms causally:
input-noise augmentation on the state->c prior is THE stability mechanism (robustness > on-dist accuracy).
Robust+avg (AUG=0.15, server samples=8): left gate100 goal0 goalmed0.56 endx0.5 => averaging HURTS completion
(0 vs 20%), dead across all configs. FINAL CLE table (pin_left 10 seeds nch200): baseline endx1872 goal~10-25;
clamp endx0.6 goal25; ROBUST AUG0.15 endx1.5 goal20 goalmed0.52 (WINNER, general, no clamp); robust+avg8
endx0.5 goal0; AUG0.05 endx1.7e7 goal0. VERDICT: pin-on-CLE divergence SOLVED generally by robust prior
(no hack); both user goals met (history prior Hc=8 works, c-R2 0.974). REMAINING (minor): final-hover
precision goal<0.4m = 20% (median 0.52m); divergence itself is fixed. Servers on ec2: 8781 robust(AUG.15,Hc8),
8782 robust+avg8, 8783 AUG.05. Priors /tmp/hist_prior_gate.pt (AUG.15) /tmp/hist_prior_a05.pt (AUG.05).

## CORRECTION: videos show ALL closed-loop fail the real task; "robust fix" was a misleading metric (2026-08-03)
Denis watched the flythrough videos (pin_cle_videos artifact): ALL versions fly through walls/ceiling/ground
and do crazy things — none do the task. My endpoint-std / goal-distance metrics were UNFAITHFUL proxies.
REAL TASK (from demos, probe_demo_task.py): start (0,0,1.5) -> out through gate (cross ~step66) -> far side
(x~2.75) -> TURN around (yaw swings to -2.1rad ~-120deg @ step225) -> back through gate (cross ~step245) ->
SETTLE+STAY in final zone. FINAL-ZONE bbox x[1.43,2.37] y[-0.85,-0.05] z[0.94,1.35] (~1.66,-0.6,1.05).
WORKSPACE bbox (physically valid, else through-geometry) x[-0.09,2.88] y[-0.86,1.35] z[0.93,2.05].
FAITHFUL METRIC = in-workspace-bounds fraction (plausibility) + stays-in-final-zone (last20%) + ends-in-zone.
RE-SCORE: teacher-forced PASSES (ep0042 100% in-bounds, 100% stays, ends-in-zone True; ep0150 59/86/True) =>
the pin CAN do the task on-distribution. ALL closed-loop FAIL: robust prior 12% in-bounds (z -1.2..5.4 through
floor+ceiling) 0% stays; baseline-lucky 99% in-bounds but 0% stays; baseline-typical 19%; aug05 4%. So the
robust prior did NOT fix the task — it improved endpoint-clustering (a bad metric) while the PATH stayed
invalid. ROOT: closed-loop covariate shift over the long out-turn-back-settle maneuver; once off-path the
rendered image is OOD and the FLOW (not just the prior) emits bad actions -> prior-robustness can't fix it.
NEXT: (a) faithful eval harness (in-bounds+stays-in-zone) as the metric going forward; (b) reduce covariate
shift on the FLOW: shorter replan horizon (gate_cle_hist APC env, wired) tighter re-grounding; DAgger-lite /
noise-augment the flow on perturbed states+renders. FIX pin_cle_fix.html artifact (overstates "the fix").

## RRR near-tie for gate; uncertainty = SAFETY signal not success; center scene missing (2026-08-03)
Denis: use scalable methods (RRR), work task success in parallel via uncertainty-based stopping, test
center+right gates. RESULTS:
- Gate RRR U built (make_u_rrr_gate.py, pi0 VLM prefix-pool features -> OLS -> top-5 eigvecs of Cov(Yhat);
  pin_U_gate_rrr_k5.npy). Held c-R2: RRR 0.966 vs PCA 0.959 (NEAR-TIE); RRR+VLM-feat prior 0.966 ~= RRR+
  state+lang prior 0.965 (VLM features add nothing over state+lang here). Why: gate is single-scene+L/R so
  PCA's max-variance dirs ARE the predictable ones; RRR only wins when high-variance action content is NOT
  obs-predictable (multi-task LIBERO). Retraining gate_both_pin_rrr anyway (scaling foundation), GPU1.
- UNCERTAINTY probe (unc_probe.py, ensemble of 6 bootstrapped priors on RRR c): in-zone unc 0.274 vs not-in-
  zone 0.270 (ratio 0.99) => uncertainty does NOT drop at success (flat in-distribution) => stop-on-low-unc
  DOESN'T isolate task success. BUT far-OOD unc 6.44 vs in-dist 0.26 (ratio 24x) => strong OOD/SAFETY signal.
  => use uncertainty as a SAFETY GATE (hold/abort when unc spikes -> prevents the wall-clip runaway), and use
  PREDICTED ACTION MAGNITUDE (demos: |a| drops ~3x near zone, 0.006 vs 0.01-0.02) as the STOP-in-zone signal.
- SCENES: left_scene + right_scene splats exist (own ckpts), NO center_scene. Right-gate testing needs right_
  scene render transforms (its dataparser_transforms) + right gate/goal geometry. Asked Denis what "center" means.

## LEADING-FUNNEL: pin stalls just past the gate; fails the turn-around (2026-08-03)
Denis refocus: PRIMARY problem is the pin not leading the robot to the end location (stopping/hold is
secondary, deferred; uncertainty stop + manual stop later). Built gate_lead_diag.py (bounded ~40-chunk
rollouts, milestone funnel over 10 seeds). BASELINE pin (8778) funnel: gate-out 100%, far-side(x>2.3) 0%,
turn(yaw<-1) 0%, gate-back 0%, in-zone 0%, in-bounds 0.87. => the pin ALWAYS threads the gate (leg 1) then
STALLS at x~1.8-2.3 just past it (demo far side x=2.75), never turns (yaw stays ~0 vs demo -2.1 swing), never
returns, never reaches zone. It mills in-bounds (doesn't fly off within 40 chunks). So "not leading to the
end" = fails the TURN-AROUND-AND-RETURN half. LEADING HYPOTHESIS = phase ambiguity: x~2 is visited twice
(outbound +x vs inbound -x/-y, opposite actions); state-conditioned pin averages -> ~zero net -> stall. The
image should disambiguate out/back but closed-loop at a drifted pose doesn't. TESTS/FIXES: (a) RRR funnel
comparison (retrain ~2.4hr out); (b) teacher-force the far-side/turn phase (teacher-forced already reaches
zone incl. the turn -> pin CAN do it given right states -> confirms closed-loop stall not pin incapacity);
(c) history-conditioning or a progress/phase signal to break the out-vs-back ambiguity (history helps c-R2
only marginally in aggregate 0.961->0.974 but may matter exactly at the twice-visited region). RRR retrain
gate_both_pin_rrr ~623/5000 @ 2s/it GPU1.

## HISTORY BREAKS THE STALL (confirms phase ambiguity); new bottleneck = overshoot+no turn (2026-08-03)
Ran gate_lead_diag funnel on the Hc=8 HISTORY pin (served 8781 GPU0, /tmp/hist_prior_gate.pt on PCA flow,
auto-reset on start pose). FUNNEL history vs single-state: gate-out 90 vs 100; far-side(x>2.3) 100 vs 0 (!!);
turn 10 vs 0; gate-back 0 vs 0; in-zone 0 vs 0; in-bounds 0.50 vs 0.87. => HISTORY unlocks forward progress
past the gate (0->100% far side), CONFIRMING phase-ambiguity was the stall (recent-motion context disambiguates
out vs in at the x~2 overlap so the pin no longer averages-to-stall). NEW bottleneck: history OVERSHOOTS the
far side (maxx 6-7 past workspace 2.88 -> through far wall, in-bounds 0.50) and still doesn't TURN (10%) or
return. Recent-forward history says "keep going" but doesn't signal "arrived at far side, turn now." NEXT for
leading-to-end: (a) RRR funnel comparison (retrain pending); (b) shorter replan horizon (apc 8->2) to catch
the turn point before overshooting; (c) a PROGRESS/PHASE or goal signal so the pin knows where in the maneuver
it is (the turn point), not just recent motion; (d) note yaw(turn) is a big high-variance action -> in top-5
U -> pinned, so the prior SHOULD be able to command it with the right context. Uncertainty safety-gate would
catch the overshoot (hold) but that's not turning.
apc=2 (tighter replan) history funnel: gate70 far100 turn0 back0 zone0 inb0.49 -> tighter feedback does NOT
help the turn (still overshoots x4.8-7.7, 0% turn). So not a replan-frequency issue; the policy genuinely
doesn't COMMAND the turn at the far side. Synthesis: history over-commits to FORWARD (fixes stall, unlocks
far-side 100%) but overshoots into OOD and never reverses. Missing ingredient = maneuver PROGRESS, not more
feedback. Demos turn at a consistent TIME (~step150-225); turn=big yaw swing=high-variance=IN top-5 U=pinned,
so the prior's c CAN command it. NEXT FIX TO TRY: condition the pin prior on normalized PROGRESS (timestep,
known at inference from step count) so it learns "at this point in the maneuver, turn" independent of the
(overshot) position. Cheap prior retrain. If turn is pinned as expected -> should trigger it.

## PROGRESS-CONDITIONING UNLOCKS LEADING-TO-END: in-zone 0%->60% (2026-08-03)
Built progress-conditioned prior c=MLP([model_state, onehot, progress]) progress=t/T (make_progress_prior.py,
PCA U to match current flow, held c-R2 0.966, exp_len 271). Served serve_gate_pin_prog.py on 8784 (progress
supplied client-side as executed_steps/271). FUNNEL (10 seeds): gate-out 100, far 70, turn 10, gate-back 0,
IN-ZONE 60 (!!), in-bounds 0.71. vs single-state in-zone 0 / history in-zone 0. => PROGRESS is the key phase
cue: breaks the stall AND tells the pin when to head for the zone instead of overshooting. Reaches the zone
via a MORE DIRECT path (not the demo's full out-and-back-turn; far 70/turn 10 low) -> ends x~2.0 partial yaw
-0.6..-0.9 settling at the animal, which IS the task ("through gate + hover over animal"). So the pin now
LEADS the robot to the end location a majority of the time. REMAINING: 40% miss; in-bounds 0.71 (not always
plausible); "reaches" not yet "stays" (staying = the deferred stopping problem). NEXT: push 60%->higher
(progress+history combo, RRR flow when trained, tune), then the stop/hold. Servers: 8784 progress-pin,
8781 history-pin, 8778 baseline. Prior /tmp/prog_prior_gate.pt. RRR flow retrain still on GPU1.

## CORRECTED MECHANISM (data, not story): drift off-manifold at the gate->far-side transition (2026-08-03)
Denis pushed back on my "phase ambiguity" story (rightly: out/back at x~2 have different y [+0.4 vs -0.7] and
different images -> state NOT degenerate). Ran gate_stall_diag.py (single-state pin, per-chunk trace: pos,
cmdspd, nn_dist to demo manifold, nn_prog=phase of nearest demo state, nn_spd). FINDINGS: (1) NOT a stall --
cmdspd ~0.006-0.03 throughout = demo-scale (nn_spd~0.026), never zero; it's moving. (2) NOT ambiguity. (3)
REAL mechanism = COVARIATE-SHIFT DRIFT: tracks demo cleanly ch0-13 (nn_dist<0.23, nn_prog 0->0.39), then at
the gate->far-side transition (~ch14, x~1.5, ~40% through) DEVIATES off-manifold (this run: z climbs
1.66->2.4 through ceiling; nn_dist balloons 0.29->0.72), and once off-manifold nn_prog FREEZES ~0.30-0.33 for
25+ chunks -> never advances to far-side/turn/return. "Stall at x~2" = x stops advancing bc it drifts UP not
forward. => PROGRESS helps by supplying phase directly (drive toward goal despite drift), NOT by resolving
ambiguity (my earlier claim was wrong). Fix implications: reduce/recover drift at the transition (robustness,
obs-grounded phase), or supply phase (progress works, 60%). My phase-ambiguity explanation RETRACTED.

## Progress variants: plain CLOCK beats "smarter" progress (2026-08-03)
Leading funnel (10 seeds, reaches-zone metric), PCA flow: plain-progress-clock(AUG0.1) inzone 60% (best);
progress+robustness(AUG0.25) inzone 10% (higher aug BLURS the settle precision); history-derived-progress
(estimator->progress->pin, serve_gate_pin_progest.py) inzone 0% -- estimator is R2 0.996 on clean demos but
closed-loop on DRIFTED states it honestly under-reports progress (never says "90% done") so it never triggers
the settle -> overshoots (maxx to 4.8). => the CLOCK's blind monotonic march to 1.0 is the FEATURE that forces
the settle regardless of drift; making progress "smarter" (robust or history-derived) HURT for this drift
failure. (10-seed sampling noise caveat, but 60 vs 10 vs 0 is a real gap.) Winner stays plain-clock 60%.
NEXT: RRR flow + plain clock (scalable-basis comparison). RRR ckpt gate_both_pin_rrr/4999 servable (has
params/). make_progress_prior parametrized UPATH for RRR c.

## RRR FLOW WINS BIG closed-loop (turn 100%, zone 80%) -- c-R2 near-tie did NOT predict it (2026-08-03)
RRR flow (gate_both_pin_rrr/4999, retrained with RRR-U coupling) + progress-clock prior (prog_prior_rrr.pt,
RRR c) leading funnel (10 seeds): gate 90, far 80, TURN 100 (!!), gate_back 0, INZONE 80 (!!), in-bounds 0.78.
vs PCA flow + progress-clock: turn 10, inzone 60, inb 0.71. The RRR flow EXECUTES THE DEMO TURN (yaw swings to
-2.1..-2.7, matching demo ~-2.1) 100% of runs; PCA flow never turned. => moving to the scalable RRR basis
(Denis's push) made the FLOW do the proper out-and-back-turn maneuver + reach the zone 80%. My earlier
prediction that RRR~PCA (from c-R2 near-tie 0.966 vs 0.959) was WRONG: c-predictability didn't capture the
closed-loop behavior; the RRR flow (pinning the OBSERVATION-predictable subspace, incl. the obs-triggered
turn) generalizes much better closed-loop. RRR flow + plain progress-clock is now the best leading config
(80% to zone, does the real maneuver). Rendering flythrough videos of all variants next.

## Gate-passing proximity + right-scene setup (2026-08-03)
Denis observed RRR clips the gate edge (worse gate-centering than single-state) + RRR descends through the
table at the end (deferred: no hold + no z-floor). PROXIMITY diagnostic (gate_proximity.py, 8 seeds, nn-dist
to demo states + aperture-miss at crossing): RRR aperture-miss 0.29 (near edge) gate-phase nn-dist 0.038;
single-state aperture-miss 0.12 (more centered) nn-dist 0.069; LEFT DEMOS cross at aperture-miss 0.19. =>
REFUTES "RRR is off-distribution at the gate": RRR is CLOSER to demos (0.038<0.069) but crosses more off-
center because the DEMOS cross off-center (0.19, angled toward the far-side turn) and RRR follows faithfully
(0.29, slightly amplified); single-state deviates MORE from demos and cuts a more-central line (0.12) but
that deviation is why it stalls later. So RRR's gate-clip = faithfulness to an off-center demo, not OOD drift.
RIGHT-SCENE render VALIDATED: derived Tw2g recipe from left (Tw2g[:3,:3]=scale*dp[:3,:3]*diag(1,-1,-1),
Tw2g[:3,3]=scale*dp[:3,3]; reproduces left exactly). right_scene scale 0.1368, ckpt .../right_scene/.../
2026-05-11_144353/...29999.ckpt; test frames render coherent room (mannequin, table, shelving). Right demos
(100) share world frame; goal ~(1.57,-0.58,1.01) same animal, far-side (1.68,-1.12) = -y side. gate_video_
scene.py (SCENE=left/right, SIDE prompt). Rendering RIGHT-gate video batch: scratch(no-pin, gate_synth_
scratch/4999, serve_gate.py) + single + progress + RRR (Denis: ADD scratch, DROP history). Scratch ckpt
servable (4999/params). BOX-CRUFT LESSON: orphaned worker python procs from rapid launch/kill cycles block
fresh pi0 servers (not VPN, not disk-499GB, not RAM-9GB used); pkill -9 -f openpi/.venv clears; readiness
strings differ: serve_gate_pin_prog="ready on ws", falsify serve_gate/serve_gate_pin="serving on ws".

**RIGHT-GATE RENDER RESOLVED (2026-08-03, big time-sink resolved).** The "right-gate videos
don't go through the gate" had TWO causes, neither the transform math being fundamentally wrong:
(1) WRONG EPISODE — data_gate_synth task map (meta.json): task0=left gate ep0100-0149, task1=RIGHT
gate ep0150-0199, task2=center-from-left ep0000-0049, task3=center-from-right ep0050-0099. I'd been
rendering ep_0050 (a CENTER-gate demo) against the right_scene splat. Use ep0150-0199 for right gate.
(2) MISSING ICP — my hand-ported right Tw2g used the raw right dataparser (dp) and skipped ICP align.
CORRECT: Tw2g = M_mocap_to_ns @ diag(1,-1,-1), where M_mocap_to_ns is joint_mocap_to_nerf.json
[scenes][{left_gate_new|right_gate_new}][joint_mocap_to_nerf_4x4] (ICP-composed, per scene). Verified:
my hardcoded LEFT Tw2g == M_left @ diag(1,-1,-1) exactly. Demo `state` is in MOCAP frame (z-UP, z>0),
NOT ned; render pos flip pn=[x,-y,-z] reproduces M_mocap_to_ns@pos correctly. Tbc (body_from_camera)
= inverse of scene cfg cam_body->cam_forward edge (R=[[0,1,0],[0,0,-1],[-1,0,0]], t=[.03,-.01,.10]) =
[[0,0,-1,.10],[1,0,0,-.03],[0,-1,0,-.01]] — my Tbc was already correct. Viewmat: R[:,1:3]*=-1 then
invert = nerfstudio get_viewmat (my *[1,-1,-1] col-flip matches). ep_0150 render vs its STORED image
matches (same room/pose); residual is only FOV/crop (stored 256^2 square; exact training crop unknown).
WHY I COULDN'T JUST RUN falsify's renderer: /home/ubuntu/code/falsify-pi/.venv is a DEAD symlink ->
/home/dfliu/code/SousVide/.venv (nonexistent on this box); no python here imports figs; no uv/lockfile.
So the render chain had to be re-ported into the /tmp/tv gsplat env. Faithful render chain source of
truth: falsify GSplatRenderer._set_tw2g_from_graph + FiGS render_rgb (Tc2g=Tw2g@T_c2w, camera_to_worlds
set directly, nerfstudio flips). Gate scoring is a STATE-SPACE question (right_gate.yaml gate_region,
mocap AABB x[-.06,1.15] y[-1.55,-.75] z[.05,2.05], plane anchor[.544,-1.147,.074] nrm[.385,-.923,0]);
ep_0150 ground truth crosses (18 frames in AABB, plane closest-approach 0.005) — judge rollouts here,
not by eyeballing renders.

**CORRECTION (2026-08-03): RRR pin does NOT pass the RIGHT gate; earlier PASS was a metric bug.**
My first right-gate score used "enters gate AABB + crosses infinite plane" = PASS. That is too loose:
a trajectory grazing the +x corner of the AABB and crossing the plane at the box edge counts as PASS
without going through the opening. Correct test = AT the plane crossing, is (x,z) inside the opening
bounds (x[-0.06,1.15], z[0.05,2.05])? Aperture-aware result: DEMO ep_0150 crosses at x=0.48 z=1.51
(THROUGH), then back at f206. RRR crosses the plane at x=1.29/2.42/1.50 EVERY time = all outside the
opening (x-max 1.15); min dist to opening-center 0.735; deepest y only -1.00 (demo -1.52). RRR
overshoots to the +x side (out to x~2.0,y=-1.0), loops the far +x region, returns low (end 1.28,-0.89,
0.65). SCRATCH: 0 in-AABB, veers +y, leaves room. So on the RIGHT gate NEITHER passes; RRR closer but
misses the aperture and turns back early. (The earlier "RRR reaches zone 80%/turn 100%" was the LEFT
gate, not right.) Render fix does NOT change trajectory (policy ~99% state-driven) — it only changed
what we see; RRR was always missing. Lesson: score gate tasks by THROUGH-APERTURE (in-opening at plane
crossing), not AABB+plane. gate_video_scene.py PASS field still uses the loose test — treat as
in-AABB indicator only; use the aperture crossing check for real verdicts.

**BREAKTHROUGH (2026-08-03): true VLM-RRR c (VLM predicts the pin coord) goes THROUGH the right
gate — routing solved.** Denis's point: since the pi0 VLM prefix is computed for denoising anyway,
predict c from it (not a state+onehot MLP). Built c = ridge-on-standardized(prefix-pool feature phi)
-> U coord. Pipeline: phi = mean-pooled pi0 embed_prefix(obs) (2048-d), c = ((phi-mu)/sg)@W + c0,
clamp to training range, noise=(I-UU^T)g+Uc, RRR flow (gate_both_pin_rrr/4999). Serve:
/tmp/serve_gate_pin_vlmc.py; map /tmp/vlmc_ridge.npz (mu,sg,W,c0,clo,chi); builder recomputes W from
vlm_feat_gate_prefix.npz cache. KEY BUG FOUND + FIXED: raw lstsq feat->c map is pathologically
ill-conditioned — bias ~±1000-1700 nearly cancels the mean feature contribution (~∓1000-1700) to
leave c~±7; feature-std/|bias|~0.01. In-dist R2=0.94 but ANY off-manifold phi breaks the cancellation
-> c explodes -> closed-loop divergence to y=-92 (and clamping/wrist-fix didn't help because direction
already wrong). FIX: center+standardize phi + ridge (lambda=100): held c-R2=0.949, maxabsW=0.30 (vs
implicit ~1000). RESULT with ridge map: crosses the gate plane at f24 (x=0.53,z=1.34) INSIDE the
opening = THROUGH APERTURE (17 frames in AABB, matches demo's x=0.48 crossing). Scratch, state+onehot
RRR (+x overshoot), and ill-conditioned VLM-c all MISSED; ridge VLM-c is the first to route through.
CAVEAT: post-gate it DIVERGES (ends x=7.84 y=-16.92 z=-1.69, descends through floor) — outbound pass
solved, far-side turn/settle not (novel far-side poses degrade c; no progress cue to force settle).
Also confirmed the DUAL-CAM obs fix: rollout must render the DOWNWARD wrist cam (Tbc_d=[[0,1,0,0],
[1,0,0,0],[0,0,-1,0.05]], downward K fx478 fy477 cx512 cy383.5) + composite carl_wrist_overlay_pinhole
_rgb.png (256^2 RGBA) after the native->256 PIL-bilinear squash; forward FOV still ~wider than the 256^2
stored (unresolved, minor). gate_video_scene.py now renders both cams. Wrist-fix alone did NOT change
the divergence (map conditioning was the real issue). Next: far-side stability (add progress to VLM-c,
or on-policy/DAgger far-side data).

**ONE-HOT WORKAROUND = VLM-RRR grounding, but our gate impl uses the WASHED-OUT pre-fusion feature
(2026-08-03).** Denis: never use one-hot for task instructions (studied already). The documented way
around it (experiments/rung3/vlm_rrr_libero.py): DEFINE U as the VLM-predictable action subspace and
predict c from the VLM's grounding — replaces hand-crafted one-hot/slot encodings; ONE U + ONE prior
carries the instruction across suites. Sub-findings: keyword one-hot is robust in-domain but doesn't
generalize to paraphrases (gate_inference.py/export_pin_prior.py comments); frozen sentence-embedding
(gate_text_embed.py, all-MiniLM) "washes out the left/right MINIMAL PAIR". CRITICAL: vlm_rrr_libero.py
has a pluggable feature backend --feat prefix|context: **prefix = PRE-FUSION embed_prefix mean-pool
(cheap baseline), context = FUSED contextualized prefix "set by the extractor once wired"** (i.e.
likely NEVER wired). embed_prefix (openpi/models/pi0.py:116) embeds SigLIP image tokens + language
tokens and CONCATS them BEFORE the LLM fuses — mean-pooling it averages ~1-2 language tokens against
hundreds of image tokens => the left/right instruction is washed out. Our gate pin_U_gate_rrr_k5 AND
serve_gate_pin_vlmc BOTH use this pre-fusion mean-pool. => explains the LEFT-gate failure: VLM-c ignores
"left", both left+right drift -y; right "works" only because the right SCENE (image) + a -y bias suffice
without needing the language. FIX: use the FUSED/contextualized (post-LLM hidden-state) features for
both U and the c-prior, so the instruction actually modulates c. FINDINGS NOT CONSOLIDATED: FACTORING_
ARC.md is the ledger for the pin construction + toy/Panda/cross-embodiment (2.1-2.9) but the GATE/drone
+ VLM-RRR + language findings live scattered in script headers, lang_rrr.json/log, and this memory —
worth consolidating into FACTORING_ARC.md or a gate findings doc.

**CTX RE-GROUNDING CLOSED-LOOP: BOTH SIDES FAIL; ROOT CAUSE IS PROMPT-SCENE CONFOUNDING, NOT THE
FEATURE (2026-08-04).** Ran the fused-feature fix end-to-end: sharded contextualized (post-LLM)
prefix extraction over data_gate_synth (6900 recs, 2 GPUs; single-GPU build OOM-killed earlier —
shard it), ridge sweep lam={10,100,1000} held c-R2 {0.948,0.920,0.892}, served lam=100
(maxW 0.70) and lam=1000 (maxW 0.31, = pre-fusion conditioning) through the RRR flow
(gate_both_pin_rrr/4999). RESULTS: lam=100 — first-ever instruction-consistent LATERAL split
(left rollout +y, right −y; pre-fusion both drifted −y) but both sides z-collapse through the
floor pre-gate (end z −3.5/−2.6), 0 in-AABB. lam=1000 — milder divergence but the language signal
shrinks with the noise (right no longer steers −y); still 0 in-AABB, THROUGH=False everywhere.
BALANCED STEER DIAGNOSTIC (rung3/ctx_steer_diag.py, 24 L + 24 R held frames) is decisive:
prompt-swap moves c strongly (||dc|| 4.4 vs behavioral-axis norm |b|=0.97) but with
direction-consistency only 0.20 and cos(prompt-swap axis, behavioral L/R axis) = +0.014 (lam=1000:
−0.77 at consistency 0.09) — i.e. the language content of the fused feature maps to c-NOISE, not
to left-vs-right. WHY (structural): in data_gate_synth prompt and scene are PERFECTLY CONFOUNDED
(each splat room shows one gate; label check: 200 eps split exactly 100L/100R along task
boundaries, center-from-X transits the X gate) — language is never the sole disambiguator, so no
fit on (obs, true-prompt) pairs can identify the language direction; the ridge assigns it an
arbitrary image in c-space. This RESOLVES the arc: pre-fusion failed because the feature LACKED
language; fused features CARRY it (prompt-swap moves c ~4x demo-side separation) but the MAP
cannot ground it without identifying data. Also: true-prompt d'=0.46 along b, and the demo c's
themselves separate sides weakly on balanced frames (left 1.81+-6.02 vs right −0.49+-2.02) —
c is state/progress-dominated, consistent with the 99%-state finding. NEXT (both arms fed by one
swapped-prompt extraction pass, running): (1) ABLATE control — PCA the paired prompt-delta
subspace out of the features, refit; language-insensitive by construction; isolates language-noise
vs render-gap as the divergence cause. (2) CFGROUND — augment with swapped-prompt features +
phase-matched counterfactual targets (opposite side's mean c at same episode progress; the
semantic-OT-coupling idea specialized to phase), forcing the language direction onto the
behavioral axis. Both fold into the same ridge npz schema (ablation projection composes into W) so
serving is unchanged. CONSOLIDATION: maintained pipeline now in-repo — rung3/gate_ctx_common.py
(single source of truth for split/segY/feature/ridge; serve imports it so serving phi == extraction
phi by construction), extract_ctx_features.py (PROMPTS=true|swap), ctx_steer_diag.py,
serve_gate_pin_vlmc.py, build_ctx_ridge_variants.py, gate_video_overlay.py (+scene). All /tmp
one-offs + ridge maps rescued verbatim to rung3/tmp_scripts_rescue/ (61 files; /tmp gets cleaned).
NOTE the /tmp/tv gsplat env is itself in /tmp — re-port per 2026-08-03 entry if cleaned. Artifacts:
~/ctxrun (Xshard/Xswapshard caches, scores, overlay mp4s, steer logs); maps
rung3/vlmc_ridge_{prefusion,ctx_lam100,ctx_lam1000}.npz.

**CFGROUND: COUNTERFACTUAL PHASE-MATCHED TARGETS GROUND THE LANGUAGE DIRECTION — OFFLINE PERFECTLY,
CLOSED-LOOP LATERALLY; THE RESIDUAL FAILURE IS A CTX-FEATURE VERTICAL BIAS, NOT LANGUAGE
(2026-08-04).** Built two map variants from one swapped-prompt extraction pass (Xswapshard caches in
~/ctxrun; extract_ctx_features.py PROMPTS=swap; builds in build_ctx_ridge_variants.py, CPU-only).
(1) CFGROUND: train rows = (X_true -> demo c) PLUS (X_swap -> phase-matched counterfactual c) where
the counterfactual target for a frame at episode-progress p under the swapped prompt is the OPPOSITE
side's train-episode mean c in the same progress bin (NPHASE=20) — supplies exactly the
(same scene, other instruction) pairs the demos never provide, making the language direction
identifiable. OFFLINE: cos(mean prompt-swap c-shift, behavioral axis b) = +0.99 with magnitude
0.94 vs |b|=0.97 (baseline map: +0.36 misaligned noise), held R2 0.872 (vs 0.920) — grounding
essentially exact in expectation, per-frame std 5.1 still large. (2) ABLATE control FALSIFIED
ITSELF, informatively: prompt-delta PCA is HIGH-RANK (top-8 = 73% var); K=8 removes the aligned
component but leaves |swap-shift| 4.3; K=32 destroys the c-signal (heldR2 0.92 -> 0.22) without
removing prompt sensitivity; K=128 catastrophic (-17). Language cannot be linearly excised from
fused features — it can only be grounded. CLOSED-LOOP (RRR flow 4999, aperture scoring): LEFT —
first gate-region entry of ANY ctx variant (56 frames in-AABB), strong monotone +y drive to and
past the left gate (y 0 -> 2.77), but crosses the region at z~0.4-0.6 (gate center z~1.5) and
descends throughout -> THROUGH=False, then flies past (no settle). RIGHT — correct -y direction but
weak (-0.5 of -1.15 needed), same descent. VERDICT: the language/grounding problem is SOLVED as far
as offline metrics + lateral closed-loop behavior can show; the remaining blocker is a systematic
DOWNWARD bias in c common to ALL ctx maps (lam100/lam1000/cfground, both sides) that pre-fusion
maps did NOT have (pre-fusion flew through right gate at correct height). Lead hypothesis: fused
features shift under closed-loop obs (gsplat renders + off-demo states) in a direction whose image
under the ridge is "descend"; the pin executes it faithfully. NEXT: render-gap probe — extract ctx
features for stored demo frames vs gsplat re-renders at the SAME poses (two-stage: tv env renders,
openpi env features), compare induced c per dim; if confirmed, fit the c-map on RENDERED training
obs (match the prior's inputs to the serving client — the established pitfall, now at feature
level). OPS NOTE: first cfground launch reused the lam1000 chain's output filenames and truncated
its two score .txt (restored from session record with provenance note; mp4/traj mtimes verified
original). All lam1000 artifacts now *_lam1000, cfground *_cfground in ~/ctxrun. /tmp/tv gsplat env
backed up to ~/tv_env_backup/ (restore to SAME path: rsync -a ~/tv_env_backup/ /tmp/tv/).

**RENDER-GAP PROBE KILLS THE SIMPLE RENDER-GAP HYPOTHESIS (2026-08-04).** Two-stage probe
(rung3/render_gap_stage{1,2}.py): stage 1 re-renders 64 demo-pose obs pairs (both scenes) through
the EXACT serving chain (1024x768 -> 256 bilinear -> wrist strut overlay -> 224 bicubic,
render_yaw=-state[3]); stage 2 compares features + clamped c on stored vs rendered frames.
RESULTS: the render gap is LARGE FOR BOTH feature types — ctx per-dim shift rms 0.63 ~= 1.0x its
in-dist per-dim std; prefusion 0.14 ~= 1.4x its (smaller) std; and the induced |dc| ~= 9 for ALL
THREE maps (ctx_lam100 / cfground / prefusion) — enormous vs the behavioral axis (0.97). Implied
per-chunk net-displacement bias: ctx z -0.09 m, cfground z -0.06 m, PREFUSION z -0.17 m (demo net
|z| scale at these frames 0.12 m). THE SURPRISE: prefusion carries the LARGEST descend bias at demo
poses yet FLEW THROUGH the right gate at correct height with the same RRR flow — so render-induced
c corruption at demo poses does NOT discriminate ctx z-collapse from prefusion success, and
"fit the c-map on rendered obs" is NOT obviously the fix. REFRAME: the kill mechanism must be in
the closed-loop feedback — c behavior at DRIFTED (off-demo) poses where rollouts actually operate
(compounding-OOD, consistent with the established teacher-forced-vs-closed-loop gap). NEXT
INSTRUMENT: c-along-rollout — re-render the poses of the saved rollout trajectories (stage-1
machinery takes arbitrary pose lists), compute c per map along each rollout, and find where the
commanded z diverges for ctx vs prefusion. Also implies: per-chunk c corruption of |dc|~9 at
VISITED demo poses is apparently survivable closed-loop (prefusion flew with it) — the pin+flow
tolerates sizable command noise when the trajectory stays near-distribution.

**B1 TOY MULTI-CONTINUATION GATE: PASS — a pinned flow cleanly selects among same-state
continuations by command; PLUS a metric confound that affects ALL pinned follow-rate claims
(2026-08-04).** experiments/toy_multicont/ (3 seeds, CPU, reuses toy_frame dataset/pin +
toy_embodiment flow_embod unchanged; linear pins only). Multi-trained executor B on
forward+reversed+hover continuations from SHARED states: err-to-command fwd 0.0111 / rev 0.0098 /
hover 0.0106 (rev and hover AT OR BELOW its forward err — bar 1 PASS at 2x margin), forward
non-regression 1.09x vs fwd-only arm A (bar 2 PASS), endpoint spread 0.004-0.007 all types = no
mode collapse. Fwd-only A commanded reverse: pinned-coordinate err stays LOW (1.49x fwd — bar 3
"A fails reverse" FAILED as pre-registered) BUT full-chunk RMSE exposes it: A's reverse chunks are
FORWARD-SHAPED Frankensteins (chunk RMSE 0.194 = 5.2x B's 0.036, ~75% of typical action magnitude;
commanded hover it keeps moving at 30% speed vs B's 7%). METHODOLOGICAL FINDING (carry-forward):
the pin's passthrough makes pinned-coordinate err-to-command a CONFOUNDED follow metric — any
pinned-trained executor approximately passes commanded coordinates through even for continuations
it cannot execute; follow claims (incl. at pi0 scale) need a paired whole-trajectory (teacher-
forced ADE / chunk RMSE) or unpinned-coordinate consistency check. CONSEQUENCE: retrain A3
launched (gate_aug_pin_rrr, pi0_gate_aug on local/gate_nav_aug, 1503 eps = 300 orig + reverse/
crops/hover per gate_traj_algebra; SNMVP_PIN_U RRR coupling, 5k steps, GPU1; norm stats reused
from gate_nav so U/clamps/c-caches stay comparable — G0 all PASS incl. reversal adds no
action/state inconsistency beyond the data's own teleop jitter, identical 6.75e-2 max).

**C-ALONG-ROLLOUT: THE SINK IS A PERSISTENT NEGATIVE cmd-z WITH NO RESTORING FORCE — PRESENT FROM
CHUNK 0 AT THE DEMO START POSE (2026-08-04).** Instrumented rerun of all map variants with per-call
c logging (serve_gate_pin_vlmc SNMVP_C_LOG + analyze_c_rollout.py; commands decoded to net meters
per chunk). CTX RUNS (valid): commanded dz at chunks 0-2 = -0.26 (cfground L) / -0.25 (cfground R)
/ -0.27 / -0.35 (lam100 L/R) vs healthy demo command reference mean -0.035 (p10 -0.24) — i.e. the
descend command exists ON-DISTRIBUTION at the exact demo start pose, ~5-10x the healthy mean, and
cmd-z NEVER GOES POSITIVE anywhere along any flight: the map has no climb-back response because
demos contain no vertical-error-recovery data (z errors are ABSORBING; lateral has scene-driven
restoring structure, z does not). Executing ~16% of a persistent -0.2..-0.5 net-z command per chunk
compounds to exactly the observed sink. Drift signatures differ by map: lam100_left DEGRADES with
drift (corr(drift,cmd-z) -0.50, commands grow — runaway); cfground_left commands SHRINK with drift
(corr(drift,|cmd|) -0.60 — regression-to-mean mush); cfground start-bias ~= lam100 start-bias, so
cfground fixed grounding but not the z bias. CAVEAT: prefusion clog reruns are INVALID as A/B —
commanded dy sits at the clamp rails (-2m+/chunk, flies to y=-14): my reconstructed prefusion
pooling (gate_ctx_common.prefusion_pool) does not match whatever produced vlm_feat_gate_prefix.npz
(builder lost to /tmp cleanup pre-rescue); historical prefusion success stands as recorded but is
not reproducible with current artifacts. IMPLICATIONS: (1) the render-gap probe's small mean z-bias
(-0.09/chunk) UNDERSTATED the serving-time bias at rollout poses (-0.25..-0.35 at chunk 0) — obs
along rollouts differ from probe frames (start pose is shared across all 4 tasks = maximally
ambiguous phase, where c averages tasks and z-averaging skews negative since ALL demo continuations
from start descend on net). (2) The augmentation attacks part of this (hover/crops teach commanded
z~0 at varied heights) but NOT error recovery — no data commands climb-from-below anywhere. The
GENERAL fix: render-DAgger — stage-1 gsplat machinery renders obs at arbitrary OFF-PATH poses;
label perturbed poses with corrective return-to-path chunks; extends the same dataset/pipeline.
Queued as the follow-up experiment after G2/G4 of the current retrain.

**G2 PASSES — THE TRAJECTORY-ALGEBRA RETRAIN GIVES THE REAL pi0 FLOW REVERSE AND HOVER AS
COMMANDABLE MOVEMENTS AT ZERO FORWARD COST (2026-08-04 08:5x UTC).** gate_aug_pin_rrr/4999
(pi0_gate_aug, local/gate_nav_aug 1503 eps, SNMVP_PIN_U RRR coupling, 5k steps, loss 0.022).
Teacher-forced whole-trajectory eval (g2_teacher_forced.py, held-out frozen-split eps, 25-step ADE,
n=72/72/24): AUG forward 0.035 m / reverse 0.035 m (IDENTICAL — the movement vocabulary transfers
perfectly) / hover ADE 0.003 with net drift 0.006 m. BASELINE (gate_both_pin_rrr): forward 0.038 /
reverse 0.107 (3x worse) / hover drift 0.081 m — the pi0-scale replication of B1's toy result,
including the Frankenstein failure (baseline moves 8 cm when commanded to hover). Gates: G2.1
reverse<=2x fwd PASS (1.0x), G2.2 hover drift<0.05 PASS (0.006), G2.3 fwd non-regression PASS
(aug 0.035 vs base 0.038 — aug slightly BETTER on forward). The toy->real ladder (B1 -> G2)
predicted this exactly. REMAINING for closed-loop (G4): the aug c-map (extraction ~70%, builds with
fwd/back grounding check from identical-frame pairs) + the known z-sink pathology — note the aug
map trains on hover/crop rows (commanded z~=0 at varied heights), which may partially mitigate the
never-positive-cmd-z pathology; render-DAgger remains the principled fix if not.

**G4 PRECURSOR + PROFILE MICRO-PROBE: THE AUG FLOW IS TEACHER-FORCED-PERFECT INCLUDING HOVER;
CLOSED-LOOP SINK PERSISTS WITH THE OLD MAP AND IS FULLY MAP-ATTRIBUTABLE (2026-08-04 ~09:40 UTC).**
(1) aug flow + cfground map closed-loop (canonical L/R + hover probe from the penguin pose
[1.52,-0.61,1.0]): all sink as before (map unchanged -> commands unchanged, cmd-z -0.18..-0.27 near
demo). (2) Hover probe detail: at the penguin the map's cmd-z was ~HEALTHY (-0.03) yet realized
descent was -0.08/chunk -> suspected receding-horizon truncation bias (pin constrains whole-chunk
NET; dip-early/rise-late profiles would realize the dip at apc=8/50). (3) MICRO-PROBE REFUTES
truncation bias on-distribution: teacher-forced hover at the penguin gives a FLAT profile (net
0.003 m, no dip, 3 noise draws); teacher-forced midflight profile tracks the true cum-z step-for-
step ([-0.041 vs -0.056]@8, [-0.265 vs -0.262]@49). The aug flow's within-chunk profiles are
demo-faithful; G2 + this = executor exonerated end-to-end. The closed-loop hover sink is the MAP's
full c-vector (not just its z-net projection): rendered obs + out-of-vocabulary prompt ("hold
position" absent from cfground training) -> commanded ~1 m lateral motion; executing that
off-manifold command descends early. EVERYTHING now converges on the aug c-map (extraction ~done):
G4 = aug flow + aug map on canonical L/R + reverse-from-penguin + hover-settle. Watch item for G4
scoring: gate_video_overlay PROMPT/START env overrides added (repo copy); hover-settle scored from
traj (stay-within-radius), not the gate THROUGH field.

**AUG C-MAP: REAL WITHIN-SCENE LANGUAGE DIVERSITY GROUNDS THE INSTRUCTION AXIS WITH A PLAIN RIDGE —
NO COUNTERFACTUAL AUGMENTATION NEEDED (2026-08-04 ~10:30 UTC).** Extraction over the augmented
synth set (15,362 recs, stride 12, grouped split by SOURCE episode to prevent stored-frame leakage;
extract_aug_features.py) + ridge build: held c-R2 0.861 overall (orig 0.884 / reverse 0.805 /
crop_from 0.901 / crop_to 0.523; hover R2 is a degenerate metric — targets exactly constant, MAE
0.75 on c-scale 7-20 = healthy). HEADLINE: FWD/BACK grounding from 870 IDENTICAL-FRAME held pairs
(same stored image, forward vs backward prompt, real executed targets): cos(predicted axis, target
axis) = +0.999 at 0.97x magnitude — a plain ridge on naturally-diverse data grounds language
essentially perfectly, confirming Denis's hypothesis that cfground is a small-data patch and the
scalable answer is data whose instructions vary within-scene. Also: the fwd/back behavioral axis is
|b|=7.7 — 8x the left/right axis (0.97): direction-of-travel dominates c-space, which is why it
grounds so cleanly. Caveats: crop_to R2 0.523 (stop-at-gate is the hardest sub-task to predict —
arrest timing); maxW 1.52 at lam=100 (spicier than prior maps; lam=300 fallback saved,
vlmc_ridge_aug_lam300.npz). Build-mode bugfix: generators need images even in build (load_eps
with_images=True). G4 (aug flow + aug map: canonical L/R, reverse-from-penguin, hover-settle, all
c-logged) LAUNCHED.

**G4 ROUND 1 (aug flow + aug map, stored-frame fit): THE SINK IS CURED AT THE COMMAND LEVEL; the
render gap now fires the DOMINANT (fwd/back) axis instead -> lateral runaway (2026-08-04 ~11:00
UTC).** Scores: all four tasks fail (canonical L/R, reverse-from-penguin, hover-settle), but the
failure DIRECTION moved: cmd-z near demo is now +0.06..+0.25 (healthy; hover run even commands
CLIMB, ends z=1.65 ABOVE start) vs -0.25..-0.41 for all pre-aug maps — the trajectory-algebra data
(hover/crops/reverse) fixed the vertical channel exactly as hoped. NEW failure: constant large
lateral commands from chunk 0 (dx -0.5..-1.1, dy -0.4..-1.2 at the demo start pose) = the
"fly backward" direction; every run flies -x/-y to y=-4..-9. MECHANISM (unifying, now 3-for-3):
the stored->rendered feature shift projects onto whatever axis DOMINATES the map — previously the
z-descend structure, now the 8x-larger fwd/back axis (|b|=7.7, maxW 1.52 amplifies). Teacher-forced
(stored frames) everything is perfect; served (rendered frames) commands are systematically wrong.
CONCLUSION: fit the c-map on RENDERED frames — the "match the prior's inputs to the serving client"
pitfall closes the loop at feature level. RUNNING: full domain-matching chain (render_aug_frames.py
renders all 7,890 unique aug frames through the exact serving chain in ~5 min; extract 15,362 recs
on rendered obs 2-GPU; build vlmc_ridge_rend with grounding checks; G4 retest). Infra: generators
now carry per-frame SOURCE indices (fidx) and tolerate image-free episodes; extractor has
OBS=stored|rendered.

**OPS INCIDENT: NpzFile memory bomb livelocked the box for ~8h (2026-08-04 10:41-18:32 UTC;
diagnosed + fixed by Denis).** extract_aug_features.py OBS=rendered subscripted the lazy NpzFile
per record (rf["fwd224"][i]) — every access re-decompresses the full ~1.2GB array and the row view
pins the fresh parent -> ~2.3GB leaked per record x2 shards; 500GB gone in ~a minute. The box has
NO swap and NO OOM killer (no earlyoom/systemd-oomd), so the kernel livelocked instead of killing:
thrashing, lost DHCP lease at 13:20, unreachable until reboot. FIX (Denis): materialize the arrays
once before the loop (verified 4,000 records = 2.3GB total under a 20GB ulimit). Consequences +
mitigations: /tmp wiped on reboot -> /tmp/tv restored from ~/tv_env_backup (the backup made hours
earlier paid for itself); relaunched chain (run_rend_chain2.sh) now carries a MemAvailable<20GB
watchdog that kills the workers (can't install earlyoom: no sudo); memory note saved (never
subscript NpzFile in a loop). Rendered frames npz survived (home dir); extraction rerunning from
step 2, memory flat at 27GB.

**G4 ROUND 2 (rendered-domain ridge): FIRST REPRODUCIBLE CLOSED-LOOP THROUGH-APERTURE PASS —
canonical RIGHT flies THROUGH the gate; z-sink eliminated on every task; residual failures fully
characterized (2026-08-04 ~21:30 UTC).** Domain-matched map (vlmc_ridge_rend, ridge on features
from serving-chain re-renders; held R2 0.784, fwd/back grounding on RENDERED pairs cos +1.000 at
0.97x): canonical RIGHT crosses the aperture at (1.10,-0.91,z=1.19), 15 in-AABB frames, ends at
healthy altitude — command curves show what health looks like: |cmd|~0.0-0.4, drift NEVER exceeds
0.5 m (stays on the demo manifold), cmd-z ~0. NO run ends below z=0.14 (floor-crash era over).
REMAINING failures (canon LEFT, reverse-from-penguin, hold-position) share one signature: L/R and
task-selection content still scene-defaulted on rendered features — left gets -y (the historical
right-bias), and at the penguin "fly back"/"hold" both get +x/-y continue commands. DIAGNOSIS
(offline, held rendered frames): lambda sweep 100->3000 leaves endpoint disambiguation FLAT
(reverse-start cmd err 0.68-0.70 m) -> conditioning exonerated; the limit is LINEAR-READOUT
CAPACITY for pointwise prompt-conditional switching. MLP TEST (Denis's hypothesis, now
domain-matched): 2048->256->256->5 GELU on the same rendered features: heldR2 0.932 (vs 0.784),
endpoint cmd err hover 0.03 m (vs 0.18), orig-end 0.07 (vs 0.20), reverse-start 0.43 (vs 0.68) —
capacity is the right axis ONCE the domain is matched (deferring it until then was correct: on
stored features it would have won R2 and lost closed-loop). RUNNING: L/R-swapped-prompt extraction
on rendered frames (orig rows only, 4,500 recs — the L/R contrast lives only in orig prompts) for
cfground-on-rendered. NEXT: combined map = MLP on rendered features + cfground L/R counterfactual
rows -> G4 round 3 (canonical L/R + reverse + hover). OPS: pkill self-match trap struck again
(compound command contained its own kill pattern; exit 144) — one action per call, bracket trick.

**G4 ROUND 3 (combined MLP + cfground-L/R on rendered) + THE STATISTICS RECKONING (2026-08-04
~23:00 UTC).** Combined map offline panel: heldR2 0.932, endpoint hover 0.05 / orig-end 0.07 /
reverse-start 0.51, L/R grounding cos +0.917 at 1.29x (overstrong), fwd/back +0.998 preserved.
CLOSED-LOOP single flights: canon_right 44 in-AABB frames but THROUGH=False (vs round 2 ridge:
THROUGH=True); left improved directionally (+y commands, too weak, barely realized); reverse
commands wrong sign (-y instead of +y); hover holds altitude but slides. Curves show the language
axes now MOVE commands but mis-calibrated — consistent with the 1.29x cf overshoot contaminating
true-prompt predictions via MLP smoothing. CRITICAL METHOD POINT before ranking maps: every G4
comparison so far is ONE stochastic flight per condition (serving g is random per call); the Phase-1
protocol-noise finding (+-5-6 pts at 10 trials) applies. LAUNCHED: overnight battery — {ridge-rend,
mlp-rend} x {canon L/R, reverse, hover} x 5 trials = 40 flights (run_battery.sh) for real rates.
OPS: first G4m launch flew against DEAD servers (server startup print KeyError'd on the MLP schema
— a patch had silently no-op'd; symptom: score files full of connection errors yet chain wrote
DONE). Lesson: chain steps must check their own outputs (grep frames) not just completion markers;
verify patches by grepping the TARGET text, not by the patch script's own success message.

**HUMAN VIDEO REVIEW VETOES THE METRICS — ALL FOUR SHOWCASED FLIGHTS ARE FAILURES, INCLUDING THE
"THROUGH" ONE (Denis, 2026-08-04 ~22:20 UTC).** Frame extraction confirms his read: the g4r
canonical-right "pass" shows a drone drifting past walls/windows with no gate-directed behavior;
the trajectory merely wandered across the gate plane inside the region box. THE METRIC WAS WRONG
TWICE: (1) the "aperture-aware" fix (2026-08-03) uses the gate-region AABB FACE (x[-0.06,1.15],
z[0.05,2.05], ~1.2x2.0 m) as "the opening" — the real hoop is ~0.45 m radius; (2) a single plane-
crossing instant says nothing about flight quality (approach, alignment, transit). STANDING RULE:
no closed-loop success claim without human (or at minimum frame-level) video review; scalar gates
are for filtering, never for declaring victory. CONSEQUENCE: the entire VLM-c line (prefusion ->
ctx -> cfground -> rendered -> MLP) has likely NEVER produced a true gate transit; the recorded
2026-08-03 successes came from the CLASSIC stack — serve_gate_pin :8778 = gate_both_pin flow +
PCA pin_U_gate_k5 + state+task-onehot MLP prior (all packaged in ~/hf_bundle/gate-drone-pi0:
gate_inference.GatePolicy(mode='pin') + assets) — which the funnel measured at gate-out 100%/10
seeds (its failure was AFTER the gate: stall/no-turn). REPRODUCTION RUNNING (run_classic_repro.sh):
classic stack re-served via adapter (serve_gate_pin_classic.py), 10-seed gate_lead_diag funnel +
2 overlay videos for human review. Battery of VLM-c maps killed mid-run (comparing two failed
methods at fine grain has no value under the corrected success criterion). NEXT after reproduction
verifies: fix the scorer to the TRUE hoop aperture; re-evaluate all past claims under it; then
improve FROM the classic stack (state-driven prior + VLM only for task selection) rather than from
the VLM-c line.

**CLASSIC STACK REPRODUCED (2026-08-04 ~23:15 UTC): funnel matches the Aug 3 record exactly —
gate-out 100%/10 seeds, far/turn/gate-back 0%, in-bounds 0.92 (record: 0.87), one seed reached
in-zone.** Two review flights cross the gate plane at (0.71,0.87,1.61) and (0.80,0.76,1.53) —
within ~0.3 m of the left gate center (0.86,0.69,1.5), i.e. inside the TRUE hoop radius, which no
VLM-map flight ever achieved (g4r's "pass" crossed ~1.6 m from the right gate center). Videos
published for Denis's judgment (rule: human review decides, metrics only filter); my own frame
review could not confidently identify the gate structure in the forward view, so no claim is made
here. Stack: hf_bundle GatePolicy(mode='pin') = gate_both_pin + PCA pin_U_gate_k5 + state/task
MLP prior, served untouched via rung3/serve_gate_pin_classic.py adapter. IF Denis confirms the
classic flights are real transits: the working recipe is state-driven commands + pin execution,
and the VLM line's role narrows to task selection (its offline grounding results remain valid);
next steps would be (1) scorer rebuilt on true hoop geometry, (2) re-judge all historical claims,
(3) attack the classic stack's known post-gate stall (history/progress signal per the Aug 3
findings) rather than continue the VLM-c estimator line.

**DENIS'S CLAIM VALIDATED + AUTHORITATIVE SCORER ADOPTED (2026-08-05 ~00:45 UTC).** New single
scoring path: rung3/gate_success.py -> falsify.safety.posthoc.check_directional_transit with the
TRUE aperture corners + strict goal box from configs/safety/<scene>.yaml (miss_gate.corners,
goal_position, goal_tolerance_half_extents; the 1.5m sphere is documented-deprecated). All ad-hoc
scorers in rollout clients are superseded. CALIBRATION on known trajs matches ground truth: record
config (RRR flow + progress-clock) left flights = FULL SUCCESS (transit through physical aperture,
correct direction, ends in goal box) 2/2; classic (no progress) = transit yes / goal no (the
documented stall); VLM-map left flight = nothing. RIGHT-SCENE VALIDATION (10 trials, record
config): **0/10 — zero aperture transits.** Signature (all 10 consistent): heads to right-gate
territory (NOT the left gate — task selection via onehot is working), crosses the infinite gate
plane every time, closest approach to aperture center 0.65-0.85 m, but ALWAYS passes outside the
opening, overshooting +x (ends x 1.6-2.2 vs aperture x-span 0.20-0.92; matches the Aug 3 recorded
crossings x=1.29/2.42/1.50). So the right-gate failure is a consistent ~1 m +x AIMING BIAS, not
confusion and not drift. NEXT DIAGNOSTIC (attribution): teacher-forced G2-style check on right
demos comparing (a) flow given TRUE c vs (b) flow given prog-prior c — separates
prior-misprediction from flow-execution bias on the right side. North-star record board updated
with validated numbers (left 2/2 strict success pending video confirmation; right 0/10).

**RIGHT-GATE ROOT CAUSE: TASK-LABEL CONTAMINATION IN THE PRIOR — the flights fly the
CENTER-FROM-RIGHT route (2026-08-05 ~01:30 UTC).** Attribution chain: (1) teacher-forced on
held-out right demos, BOTH true-c and prior-c aim cleanly (aim-err x +0.007/+0.016 m — nothing like
the 1 m overshoot): flow and prior are fine ON DEMO STATES. (2) Prior replay along flown right
trajectories: commands plausible, drift-to-"right-labeled"-demos tiny — suspicious. (3) The strict
checker splits the "right-labeled" demos 50/50 pass/fail -> the binary is_left labeler (used by
make_progress_prior.py AND gate_ctx_common) folds the 50 CENTER-FROM-RIGHT episodes into the
"right" task. (4) Decisive: flown "right gate" flights track the center-from-right demo cluster at
0.24-0.32 m mean vs 0.80-0.97 m to true right-gate demos — the prior, trained on a 50/50 route
mixture under one label, resolves "right" to the center route (explains the consistent +x
crossings at 1.2-3.2 and the historical Aug-3 misses). The FLOW is not contaminated (LeRobot
prompts from authoritative 4-task meta). LEFT presumably survived because its mixture resolves
toward the left-gate route. DOWNSTREAM CONTAMINATION WARNING: gate_ctx_common labels feed the
VLM-line behavioral axes, cfground phase-matching, and the trajectory-algebra prompts — all mix
center episodes into the gate tasks; re-examine after the prior fix lands. FIX RUNNING
(run_prior4.sh): make_progress_prior4.py (authoritative 4-task labels by episode index, 4-way
onehot, aug mask corrected to spare the onehot block) + serve_gate_pin_prog4.py (exact-string task
match — keyword matching is ambiguous with 4 tasks; fails loudly) -> 10 right + 10 left trials ->
gate_success verdicts.

**LABEL FIX VINDICATED: RIGHT-GATE TRANSIT 0/10 -> 10/10 (2026-08-05 ~02:15 UTC).** Decontaminated
prior (make_progress_prior4.py, authoritative 4-task labels; serve_gate_pin_prog4.py exact-string
task match) on the unchanged record backbone: RIGHT transit 10/10 correct-direction (was 0/10),
full success 2/10 — the residual gap is the settle/goal leg (right transits at step ~88-112 vs
left ~75-85, less budget for the return; plus the known no-hold flaw). LEFT: 10/10 transit, 9/10
full success. The right gate never needed a new command architecture — it needed correct labels.
NEW CONTAMINATION FOUND while answering Denis's data-composition question: local/gate_nav_aug
(which trained gate_aug_pin_rrr) carries the SAME binary-label bug in its 300 original episodes —
100 synth center episodes are prompted as gate tasks (aug tasks are only the 2 gate prompts + 4
side-agnostic aug prompts). The aug flow therefore learned contaminated gate semantics; rebuild
gate_nav_aug with 4-task original prompts before any further use. Record board updated.

**PHASE A (label decontamination) UNDERWAY (2026-08-05 ~02:45 UTC).** A1 DONE: gate_ctx_common.
load_eps now takes labels from the authoritative data_gate_synth/meta.json (fails loudly on
unknown prompts); is_left carries a deprecation warning (geometry helper only, never for labels);
gate_traj_algebra crops skip center-route episodes (no gate plane wired for them). Verified:
50/50/50/50 task counts, ep_0150=right gate, ep_0050=center-from-right; augmented composition now
4 correct original prompts + crops only for gate episodes. A2 RUNNING: gate_nav_aug rebuilt with
clean prompts -> aug flow retrain (gate_aug_pin_rrr2, GPU1). A3 RUNNING in parallel (GPU0):
clean-label re-extraction of stored + rendered feature caches (rendered_frames.npz poses are
label-independent and reusable; the new rec set's (si,fidx) pairs are a subset of the rendered
set). Phase B (re-derive all offline grounding numbers, incl. the gate-vs-center within-scene
contrast that invalidates the "prompt/scene perfectly confounded" claim) queues on these.

**TRIVIAL-ERROR AUDIT (Denis's request, 2026-08-05 ~01:45 UTC): two review passes over the active
pipeline; 12 findings, all fixed or guarded; 3 provenance flags for Phase B/C.** Highlights:
- **HIGH, live: rendered-frame store had wrong-scene renders** — render_aug_frames assigned scenes
  by the binary-label test; pixel-verified that ALL center episodes live in the LEFT splat (CFR
  frames matched left-scene renders at |img diff| ~7 vs ~39 cross-scene). 50-100 episodes' renders
  wrong depending on label era -> EVERY rendered-domain map to date (vlmc_ridge_rend, vlmc_mlp_rend)
  carried wrong-scene features for part of the data. FIXED (scene := right iff PROMPT_R) + corrected
  re-render armed to land before the running chain's rendered extraction consumes the store.
- **HIGH: swaplr/cfground folded center into "right"** — extract's swap map now mirrors within
  task family (L<->R, CFL<->CFR) and asserts OBS=rendered; build_mlp_map counterfactual rows and
  L/R axes now restricted to gate episodes.
- **MED-HIGH: make_progress_prior4 split violation** — its held set was 20 frozen-TRAIN episodes;
  all 40 frozen-TEST episodes were in its training data. Fixed to the frozen 160/40 split;
  prog_prior_rrr4 to be retrained before any offline comparison (closed-loop 10/10 transit stands —
  rollouts are not test episodes). state_dim metadata fixed (-5 not -3).
- **Chain robustness (recurring dead-server/DONE failure mode):** run_phaseA.sh now verifies its
  own outputs (checkpoint dir, extract sentinels, watchdog marker) before DONE; watchdog pkill
  narrowed (bare "train.py" matched ANY process — shared-box hazard); readiness loops now fail
  loudly; run_g4.sh stale-clog rm fixed; legacy chains still print the deprecated overlay score —
  gate_success is the only admissible verdict (north-star rule 2).
- **Guards:** legacy serve_gate_pin_prog refuses >2-task priors (its keyword onehot is two-hot for
  4 tasks — silently OOD inputs otherwise).
- **Also known:** gate_lead_diag sends fwd image duplicated as wrist (dual-cam fix never reached
  the funnel client) — funnel numbers vs video-client runs see different obs; fix before next
  funnel use.
- **PROVENANCE FLAGS (not bugs, decisions):** (1) pin_U_gate_rrr_k5 was fit on VLM features
  extracted under contaminated prompts — rebuilding U implies retraining the flow; defer to Phase C
  with eyes open. (2) progress calibration: training prog=t/(Tn-1) capped ~0.83, serving
  steps/271 can exceed 1.0 — prior queried outside training support exactly at the endgame (~±11%
  task-length miscalibration); candidate explanation for the settle-phase weakness. (3) prior
  checkpoints don't record their U; server takes --pin-u independently — mismatch is silent.

**PHASE B RESULTS — the clean-data re-derivation (2026-08-05 ~09:30 UTC).** (1) CLOSED-LOOP,
split-clean prior (prog_prior_rrr4b) on record flow, strict scorer: LEFT 10/10 FULL SUCCESS
(overnight pooled left: 28/30); right 1/10 full with transit intact (pooled right full: 4/30,
transit ~100%) — the settle gap is the sole open deficiency, Phase C attacks it (NCH budget +
hover-capable flow). (2) OFFLINE GROUNDING SURVIVES DECONTAMINATION: fwd/back identical-frame
pairs give cos +1.000 at 0.98x on BOTH clean caches (stored heldR2 0.863, rendered 0.809).
(3) G2 ON CLEAN AUG FLOW (gate_aug_pin_rrr2): reverse PASS (0.036 ~= fwd), hover PASS (9mm);
G2.3 forward non-regression FAILS BY REGISTRATION at 1.34x vs the 1.3x bar — NOTE the bar
tightened because the BASELINE improved with correct prompts (0.038 -> ~0.025 fwd ADE: the
un-retrained record flow performs better when evaluated with the labels it was trained on —
itself evidence of how much the contaminated prompts were costing). Margin 1mm; flagged not
blessed. (4) WITHIN-SCENE TASK GROUNDING WITHOUT COUNTERFACTUALS: at held START frames (t<24,
near-identical obs within a room), the plain clean ridge on RENDERED features predicts the
gate-vs-center task axis at cos 0.917 (left room) / 0.944 (right-labeled pair) with 0.79-1.11x
magnitude — language is grounded by natural 4-task diversity alone. THE "PROMPT/SCENE PERFECTLY
CONFOUNDED" CLAIM IS RETRACTED as a label artifact; cfground is retired to the toolbox (its
counterfactual construction remains valid for datasets that truly lack within-scene contrast).
Phase C batteries running (combo flow-swap, first center-task closed-loop, right NCH=60 probe).

**PHASE C RESULTS (2026-08-05 ~10:00 UTC): three of four tasks at/near ceiling closed-loop; the
right settle gap survives both attacks and points at progress calibration.** Strict scorer, 10
seeds/arm: C1 COMBO (split-clean prior + CLEAN AUG FLOW): left 10/10, right 2/10 — a hover-capable
flow does NOT fix the right settle, exonerating the flow; the command at the settle phase is the
blocker. C2 CENTER TASKS (first closed-loop ever): center-from-left 10/10, center-from-right 7/10
— the clean 4-task prior flies center tasks essentially out of the box. C3 RIGHT NCH=60 BUDGET:
3/10 (vs 1-2/10 at NCH=40) — more budget helps marginally; not the root cause. STANDING SUSPECT
for the right settle: the progress clock's calibration (single exp_len=271 vs per-task lengths;
training support capped ~0.83 while serving exceeds 1.0 — and right transits later (~step 90 vs
78), pushing the settle phase further off-support). NEXT KNOB (Denis's call in the morning):
per-TASK clock calibration — still a blind monotone clock (the variant class that WINS), just
calibrated to each task's demo length; plus the audit's flagged fix of training-time prog cap.
Scoreboard after the clean sweep: LEFT 10/10 (pooled 28-38/40), CFL 10/10, CFR 7/10, RIGHT
transit ~100% / full 1-3 per 10. Phase D (VLM-line clean retest, 2 maps x 4 tasks x 5 trials +
center top-up seeds) running until the 14:40 UTC guard.

**PHASE D RESULTS (2026-08-05 ~14:45 UTC): the VLM-feature command path is 0% closed-loop even
FULLY DECONTAMINATED — the offline/closed-loop chasm is now a clean finding.** Clean labels, clean
rendered-domain features, clean aug flow, both map classes: ridge 0/5 + 0/5, MLP 0/5 + 0/5 on
canonical left/right (strict scorer). With offline numbers this good (heldR2 0.93, fwd/back
grounding cos 1.00, within-scene task axes 0.92-0.94), the conclusion is no longer attributable
to any data bug: READOUTS OF FUSED VLM FEATURES DO NOT SURVIVE CLOSED-LOOP FEEDBACK in this
regime, while the state+clock prior does. This cleanly supports the split command architecture
(state carries geometry closed-loop; VLM features carry task selection — which they demonstrably
ground). CENTER TOP-UP pooled: CFL 20/20, CFR 16/20. FINAL OVERNIGHT SCOREBOARD (strict
gate_success, clean stack): LEFT 10/10 · CENTER-FROM-LEFT 20/20 · CENTER-FROM-RIGHT 16/20 ·
RIGHT transit ~100%, full 1-3/10 (settle gap; progress-calibration knob queued for Denis).

**THE CLOCK FALLS: no-progress ablation scores LEFT 9/10, RIGHT 10/10 FULL — removing the
progress input FIXES the right settle gap (2026-08-05 ~17:30 UTC).** c=MLP([state, task-onehot]),
no clock, same pin flow, strict scorer: right 10/10 (clock version: 1-3/10), left 9/10. Combined
with C1/C3 (hover-capable flow didn't fix right: 2/10; +50% budget didn't: 3/10), the right settle
gap was CAUSED by the progress clock — its single exp_len=271 calibration + off-support endgame
queries actively misled the settle phase on the longer-transiting right routes. REINTERPRETATION
OF THE HISTORICAL RECORD: the clock's famous unlock (in-zone 0%->60-80%, Aug 3) happened on the
CONTAMINATED-LABEL prior — the clock was compensating for a prior that couldn't advance phase
because its "task" was a two-route mixture. With clean labels, state+task alone is a sufficient
phase signal for these non-self-intersecting routes. Also: CENTER UNDER CORRECTED RENDER
(gsplat_scene_edit.py move_gate replication, eye-validated against stored frames): clock prior
CFL 10/10, CFR 10/10 (up from 7/10 under the wrong render). SCRATCH CONTROL: transit 20/20 (clean
4-task conditioning routes fine) but FULL 0/20 (never settles) — completion, not routing, is what
the pin+prior system adds on this benchmark; steerability remains the pin's distinctive property.
Denis's design position vindicated twice over: the progress bar was a hack (a priori duration
knowledge), and it is now REMOVED from the best config. Pending: no-prog on center tasks (running)
to complete the 4-task table; then the record board flips to the simplest system yet.

**NEW RECORD — THE SIMPLEST SYSTEM WINS: 39/40 ACROSS ALL FOUR TASKS, NO CLOCK (2026-08-05 ~18:15
UTC).** No-progress prior c=MLP([state, task-onehot]) on the RRR pin flow, strict scorer, corrected
center render: LEFT 9/10 · RIGHT 10/10 · CENTER-FROM-LEFT 10/10 · CENTER-FROM-RIGHT 10/10. No
clock, no VLM features in the command loop, a two-layer MLP prior. Controls: scratch (no pin/prior)
transit 20/20 / full 0/20; clock prior right 1-3/10 (the clock CAUSED the gap); VLM maps 0/5
everywhere. The week's journey in one line: every architectural elaboration (fusion features,
counterfactual grounding, domain matching, MLP capacity, the clock itself) was compensating for
data bugs — fix the labels, the scorer, and the render, and the minimal factorization (pin channel
+ tiny state/task prior) solves the benchmark. Pending video confirmation by Denis (grid updated).
Known scaffold: task-onehot (north-star non-negotiable) — replacing it with VLM task SELECTION
(not command regression) is the language path forward; the pin remains the movement channel.

**VLM TASK SELECTION, GATES a+b (2026-08-05 ~19:30 UTC): in-distribution perfect, zero-shot
paraphrase FAILS — semantic variation must be trained in.** Selector = 128-unit GELU head on clean
rendered fused features, trained on true+swapped canonical prompts (same-frame pairs dissociate
language from scene). GATE a: held accuracy 1.000 on BOTH true and swapped rows. GATE b (32
hand-authored unseen paraphrases x 24 held frames, serving domain): 0.61/0.56/0.31/0.22 per task
vs the 0.90 bar — FAIL. Chance is 0.25; the head learned string idiosyncrasies of 4 canonical
prompts, not task semantics. Fused features do NOT confer free zero-shot paraphrase robustness on
a small head (refines the vlm_rrr_libero "context generalizes" finding: it generalizes when
training exposes variation). NEXT: paraphrase-augmented selector training (12 fresh training
paraphrases/task, DISJOINT from the 32-eval set which stays untouched as the held-out bar),
feature extraction pass, retrain, re-run gate b.

**SCRATCH ON CENTER/RIGHT + THE DWELL CRITERION (2026-08-05 ~21:30 UTC).** Scratch (clean data,
no pin/prior): center-from-left 0/10, RIGHT 0/10, but CENTER-FROM-RIGHT 8/10 touch — and a dwell
analysis (>=16 consecutive post-transit frames in the goal box, ~2 chunks) confirms 7/10 are
GENUINE settles (median best-dwell 24 frames): scratch's language conditioning fully completes the
one task whose route it happens to master, and no others. The no-clock pin system holds under the
stricter dwell criterion: CFL 10/10, CFR 10/10, left 8/10, right 9/10 -> **37/40 dwell vs
scratch 7/40** (median dwells 33-69 frames = real station-keeping for 4-8 chunks). METHODOLOGY
ADOPTED: dwell>=16 joins the success criterion going forward (touch-only inflated scratch CFR and
would eventually inflate anything); this operationalizes the endgame-methodology discussion
(Denis) — with the computed-c_hold endgame still to come for unbounded-horizon station-keeping
(the 2mm/chunk deterministic hover bias integrates linearly; per-chunk dwell is fine, minutes-long
hover needs the position-referenced command).

**HUMAN VERDICT NOTE (Denis, 2026-08-05 ~22:15 UTC): scratch center-from-right is a FAIL on video
despite scoring 8/10 touch / 7/10 dwell — the automated criterion (even with dwell>=16) is still
not human-equivalent.** Per his instruction: noted without over-analysis; starred in the flight
matrix; treat scratch CFR cells skeptically. The pin-vs-scratch qualitative gap on the same task is
obvious on video (his words). Standing implication: automated scores rank and filter; human review
remains the decision authority (north-star rule 2 unchanged and reaffirmed).

**TASK-SELECTION PARAPHRASE LADDER + SIM->REAL PRIOR TRANSFER (2026-08-05 evening).**
(1) SELECTOR: canonical/swap training = 100% held (gate a); zero-shot paraphrases FAIL (0.22-0.61);
paraphrase-augmented (12/task) FAIL (0.41-0.63); LANGUAGE-POOLED features + augmentation FAIL
(0.51-0.73, center improved to ~0.73). Conclusion so far: trained heads on this VLM's pooled
features don't reach the 0.90 paraphrase bar at hand-authorable data scale. Zero-training
RETRIEVAL selection (nearest canonical prompt in lang-pool space, same frame) running as the last
representation-direct probe.
(2) SIM->REAL PRIOR TRANSFER (Denis's inferred-real-center hypothesis): held-real per task —
sim-only NEGATIVE (-0.32/0.00: route-style mismatch, reproduced clean); real-only 0.56/0.73;
naive sim->real FT ties real-only but FORGETS center (-59.6); FT-on-left-only EXPLODES elsewhere
(R2 -1e8: small-net off-support pathology; old cross-instruction negative confirmed on clean
labels). **MIXED (sim+real co-trained): real 0.66/0.75 (BEST, beats real-only -> sim data helps
real) with synth-center fully preserved (0.98)** — the deployment candidate. Explicit inferred-
real-center test queued: shared task-independent residual Δ(state), leave-one-task-out.

**PARAPHRASE LADDER COMPLETE — FOUR METHODS, FOUR FAILURES: VLA FINE-TUNING APPEARS TO COLLAPSE
PARAPHRASE SEMANTICS (2026-08-06 ~00:30 UTC).** Zero-training retrieval (nearest canonical prompt
in lang-pool space, same frame): 0.18-0.61 — FAIL. Full ladder vs the 0.90 bar: zero-shot head
0.22-0.61 / augmented head 0.41-0.63 / language-pooled head 0.51-0.73 / retrieval 0.18-0.61.
With retrieval failing too, the deficit is REPRESENTATIONAL, not a head/training issue: the
gate-fine-tuned pi0's PaliGemma pathway (trained against 4 fixed strings) no longer encodes
paraphrase-invariant task semantics. COROLLARY TO TEST: base (pre-VLA-fine-tune) PaliGemma
features should retain them — selection may need the base language tower (extractable from the
base pi0 checkpoint). DECISION: gates c/d deferred until a paraphrase-robust representation
exists; the canonical-prompt selector (100% on canonical+swap) is functionally a language-derived
one-hot — insufficient north-star progress to justify rollout spend. The COMPOSITION experiment
(left gate -> center gate, duplicated-gate scene, milestone-switched modules) is tonight's
closed-loop test instead — flying now.

**COMPOSITION EXPERIMENT COMPLETE, WITH SCRATCH CONTROL (2026-08-06 ~01:45 UTC).** Compound task
(left gate -> duplicated center gate, falsify left_and_center scene + ordered_miss_gate criterion;
milestone-switched task modules, zero new training). PIN SYSTEM (no-clock prior): gate-1 10/10,
CFL route topology from the never-trained entry state 10/10 (out along +y corridor, cross, return
to within ~0.3 m of the goal), aperture at the duplicated gate 0/10 (consistent +x overshoot,
crossings x~3.7-3.9 vs opening max 3.16). SCRATCH CONTROL (same scene + switch protocol): gate-1
9/10, then STALLS near the second gate region (ends ~(2.7,-0.1)), never crosses, never returns —
no route completion. VERDICT: composition of movement modules through the shared state-referenced
command space is REAL (route knowledge transfers to off-nominal entry states; scratch cannot do
this); the residual gap is METRIC precision at the composed boundary, shared by both systems
(novel-scene gate), with the geometry-referenced computed command (c toward the published gate_2
anchor — the c_hold construction at a waypoint) as the queued fix. Grid updated with the compound
row + videos; Denis's review pending.

**CORRECTION IN PROGRESS (Denis's catch, 2026-08-06 ~02:15 UTC): scratch compound "stall" may be
slow transit — endpoints sit AT the gate threshold (y -0.06..-0.27 vs plane -0.33, several inside
the aperture x-span, crossing height), creeping 0.1-0.4 m/40 steps at the NCH=60 cutoff. My
"stalls, no return" read was premature — budget-truncation confound. NCH=140 rerun (10 seeds)
flying; grid caption to be corrected with the outcome either way.

**CORRECTION LANDED (2026-08-06 ~03:45 UTC): scratch compound at NCH=140 — 8/10 genuinely loiter
(never cross gate-2 with 2.3x budget), 2/10 drift through very late (step ~819 vs pin's full route
~350), 0/10 reach/hold the goal (dwell 0).** Refined characterization: scratch is not strictly
incapable at the composed gate — it is behaviorally STUCK: parked at the threshold, occasional
drift-through, no completion. Contrast with pin composition: decisive full-route execution 10/10
in ~350 steps with a lateral aperture miss at the novel gate. Opposite failure modes: pin =
geometric offset at an unseen boundary; scratch = no route program beyond its trained ending.
Grid cell corrected. (Credit: Denis's video-based challenge of my "stall" read — second time
human review corrected an automated characterization in two days.)

**FIRST COMPLETE COMPOUND SUCCESSES — ORACLE SCREENS (2026-08-06 ~04:45 UTC, fast tier 5 seeds).**
Post-switch command replaced by oracles (Denis's proposal): DEMO-NN oracle (nearest CFL demo
state's true chunk-c): 1/5 FULL success (both gates + 127-frame dwell — the first complete
compound task in project history). WAYPOINT oracle (computed c toward the published gate-2
aperture center, then goal — zero learning, the c_hold construction at a waypoint): 1/5 full
(dwell 49) + 1/5 both-gates-no-dwell. Diagnostic that motivated it: the learned prior is
demo-consistent AT the switch but lags the corner turn off-manifold (prior +0.08,-0.37 vs demo
-0.20,-0.55 at x~2.9) — a tracking lag, not a phase reset (Denis's "origin" guess refined).
CONCLUSIONS: (1) the pipeline CAN complete the compound task end-to-end — command quality was the
binding constraint; (2) residual fragility is execution variance at the never-trained gate (even
oracle commands: 1/5) — flow-side, likely the novel two-gate scene visuals; (3) the VLM-
generalization target is now concretely bounded: sequencing + waypoint selection, both scene-
readable quantities. NEXT (fast tier): variance study on the oracle arms (10 more seeds when
promoted), and the flow's novel-scene sensitivity (fly CFL solo in the two-gate scene vs the
one-gate center scene — isolates the visual novelty factor). Videos on the composition page;
metadata fix noted: make_progress_prior4 state_dim wrong for NOPROG variants (in_dim-5 vs -4).

**DISPLACED-START VERIFICATION PARTIALLY REFUTES "CORNER LAG -> WIDE CROSSING" (2026-08-06 ~06:00
UTC, fast tier).** CFL solo, single-gate center scene, learned no-clock prior, no oracle: nominal
start 5/5 (crossings 2.60-2.92); mid-displaced 5/5 (2.86-3.04 — mild outward drift); FAR start
(the compound handoff point) 0/5 — but NOT wide: 3/5 fly to the corner (max-x 2.9-3.0) then
RETREAT/loop back without crossing; 2/5 cross INSIDE the aperture (2.70, 2.81) yet fail
goal/dwell. REFINED PICTURE: (a) the prior's off-manifold corner failure = unreliable turn
COMMITMENT (retreat), not wide crossing; (b) the compound run's 3.7-3.9 wide crossings therefore
require the compound context — the two-gate scene visuals (flow-side) and/or arrival dynamics —
isolation cell running (CFL solo, far start, TWO-GATE scene); (c) the command-replay corner lag
(prior +0.08,-0.37 vs demo -0.20,-0.55) stands as measured but its closed-loop consequence was
misattributed. Second self-correction via direct verification tonight (Denis's methodology
pressure paying off).

**ATTRIBUTION COMPLETE + TWO REPLICATION RESULTS (2026-08-06 ~07:15 UTC).** (1) ISOLATION CELL
(CFL solo, far start, TWO-GATE scene): 5/5 cross, scattered 0.62-3.02, NO wide crossings, 1/5 full
— combined with the single-gate far arm (retreat/inside-cross), the compound's consistent 3.7-3.9
overshoot is attributed to ARRIVAL DYNAMICS at the module seam: the drone reaches the handoff
carrying outbound momentum the position+yaw-only prior cannot see, and the receding-horizon loop
rides it past the corner. Fix menu: (a) HOLD-SPLICE — 2 chunks of computed hover c
(segY(zeros)@U, pure geometry) between modules as a momentum damper, using the aug flow's hover
vocabulary — non-oracle, SCREEN FLYING NOW; (b) velocity in the prior state; (c) locator-driven
waypoint through corners. (2) GATE-LOCATOR TRAINED: fused features -> 3D anchor of the
prompt-named gate at held error 0.17-0.21 m (aperture 0.83 m wide) — precise enough to replace the
waypoint oracle; compound-scene generalization test pending. (3) BASE-TOWER chain relaunched
after a config fix (base ckpt is non-LoRA; extract under pi0_libero config).

**NIGHT-CLOSE NEGATIVES + AN HONEST CONFOUND (2026-08-06 ~08:15 UTC).** (1) HOLD-SPLICE: 0/5, and
crossings STILL WIDE (3.28-3.85) — but the splice run used the AUG2 flow (for hover) while the
isolation cell used the RECORD flow: FLOW VARIANT is now confounded with the splice. The momentum
attribution is therefore INCOMPLETE — wide crossing correlates with something not yet isolated.
TOMORROW'S FACTORIAL (flow held fixed): {record, aug2} x {rest-start, momentum-arrival} x
{splice, no-splice} in the two-gate scene, plus verify the splice actually zeroes velocity at the
seam (check traj speed through the hold chunks). (2) BASE-TOWER SELECTOR: FAIL (0.55-0.67) —
barely different from the fine-tuned tower; the paraphrase deficit is NOT a VLA-fine-tuning
artifact. Fifth ladder rung: pooled PaliGemma features + small heads do not yield paraphrase-
robust task selection at hand-authorable data scale, base or fine-tuned. The corollary is
REFUTED; instruction robustness needs a different mechanism (generative LLM readout, larger text
encoders, or instruction canonicalization) — future work, honestly scoped. (3) Banked positives
today: gate-locator 0.17-0.21 m (oracle-replacement candidate); first complete compound successes
under both oracles; the displaced-start and isolation cells (mechanism narrowed to seam
conditions); mixed sim+real prior (0.66/0.75 real, center preserved).

## Gate-locator generalizes to the UNSEEN two-gate scene; language switches the target (2026-08-05)

Test: 24 rendered corridor poses in the left_and_center compound scene (duplicate_aabb edit —
never in locator training, which saw only single-gate scenes), fused ctx features extracted at
each pose under two prompts, locator head (`gate_locator.npz`) predicts the 3D anchor of the
gate the prompt names. Script `experiments/rung3/locator_compound_test.py` (render stage in
`run_locator_compound.sh`).

- LEFT prompt: 0.273±0.163 m error vs the left gate anchor [0.861,0.694,1.075] — matches the
  in-distribution held error band (0.17-0.21 m). The added second gate does not distract it.
- CENTER prompt: mean prediction [2.15,-0.04,0.89] vs true duplicated-gate anchor
  [2.756,-0.3275,1.0] — 0.717±0.563 m. Direction is unambiguous (pred x=2.15 is the far gate's
  side, not the left gate's x=0.86), so the head DOES switch target gates on language alone in a
  single scene; the high std says per-frame predictions are noisy/possibly bimodal on the unseen
  duplicate, not uniformly shifted.

Reading: language-conditional gate localization survives scene composition qualitatively
(prompt selects which gate) but the duplicated gate's position estimate is ~3x looser than
in-distribution. Good enough to disambiguate/waypoint-seed coarsely; not yet oracle-replacement
precision in compound scenes. Next lever if needed: add compound-scene frames to locator training
(renders are cheap; labels are the YAML edit params).

## Shared task-independent residual Δ(state): NEGATIVE — the sim->real correction is task-specific (2026-08-05)

The queued leave-one-task-out test of the inferred-real-center hypothesis, surgical form
(`experiments/rung3/shared_delta_residual.py`): sim prior FROZEN (noprog_prior_rrr4.pt), Δ =
ridge on transformed state (no task one-hot, λ=1 by 5-fold CV on left-train only), fit on real
LEFT train rows (998), target c_true − prior(x). Held-row R² (same frozen split as the
2026-08-05 transfer table):

  model              real-L   real-R   synth-CENTER
  sim prior          -0.317    0.003      0.982
  prior + Δ_left      0.404   -1.876     -2.701
  (mixed co-trained   0.66     0.75       0.98  — reference)

Δ fixes the task it was fit on (left −0.32→0.40) but makes RIGHT much worse than doing nothing
(0.00→−1.88): the sim->real correction does NOT live in a shared state-conditioned map — it is
route/task-specific. Synth-center also poisoned (−2.70), as an always-on additive Δ must when
the correction isn't universal. With FT-on-left's −1e8 blowup, that's two independent forms of
one-task real adaptation failing to transfer; the old cross-instruction negative is now
confirmed on clean labels in both forms. Consequence: no free inferred-real-center c from
left-only real data; MIXED sim+real co-training stands as the deployment path, and real center
competence needs real center demos (or a mechanism beyond state-conditioned residuals).

## Wide-crossing factorial: NOT momentum, NOT splice, NOT flow — near-gate execution drifts +x; seam states are off the prior's manifold (2026-08-05)

Factorial in left_and_center ({record, aug2 flow} x {momentum, momentum+2-chunk hold splice,
rest-start at seam [1.522,-0.614,0.997]} x 5), one splice server per flow (`run_factorial.sh`,
scores `fac_scores.txt`, kinematics `fac_seam.txt`, geometry pass logged below). 0/30 — but the
cells attribute cleanly:

- SPLICE VERIFIED KINEMATICALLY, NO EFFECT: post-switch speed min 0.0002 m/step (aug2; rec
  0.0013-0.0024) — the hold really parks the drone; crossings stay wide (rec spl cross-x
  3.69-3.81 vs mom 3.62-3.75; aperture edge 3.16). Arrival momentum is REFUTED as the mechanism.
- FLOW VARIANT: no effect (record and aug2 same +x-wide signature; aug2 rest cells instead
  descend/retreat — consistent with aug flows' hold/descend vocabulary off-manifold).
- REST-START CELLS WERE OFF-MANIFOLD COMMAND TESTS, not momentum controls: decoding the no-clock
  prior at the seam state gives cmd [-0.305,0.096,0.203] m vs to-gate [1.23,0.29,0] — cos -0.86,
  pointing AWAY. CFL demos never visit the seam region; the prior extrapolates badly there
  (same class of failure as displaced-start "far").
- ON-ROUTE COMMANDS ARE GATE-DIRECTED: along a failed momentum run (rec_mom_t1) cos(cmd, to-gate)
  = 0.79/0.62/0.96/1.00 at steps 85/120/160/200 — the drone rejoins the CFL route (swings to
  [1.0,0.8] = demo-start region, then +x), commands point at the gate, yet the realized path
  gains +x the command doesn't ask for and crosses at x 3.5-3.8 (0.4-0.65 m outside). Same ~+x
  aim-bias signature as the right-scene failure (closest approach here 0.78-1.16 m vs 0.65-0.85).

Open discriminator (running): waypoint oracle x5 in the same scene/route — oracle exactness at
the near-gate window vs the prior's clamped/weaker commands. If oracle ~5/5 the miss is a
COMMAND deficiency near the gate; if oracle ~0-1/5 the earlier single success was luck and the
compound RENDER corrupts execution (flow-level visual deviation).

## Waypoint oracle 5-trial: 0/5 — but NOT wide: THRESHOLD STALL at 0.11-0.19 m from gate-2 center; oracle switching bug found and fixed (2026-08-05)

The discriminator rerun (`run_oracle5.sh`, record flow, canonical compound route x5): 0/5, yet
the geometry is the OPPOSITE of the prior-driven failures — every run parks at the aperture
mouth: closest approach to gate-2 center 0.11-0.19 m, ends [2.73-2.98, -0.27..-0.32, 0.87-0.90]
(plane y=-0.3275). The earlier single-trial "success" (dwell 49) does not reproduce as-is.

Root cause is the ORACLE'S OWN switching rule, not the flow and not the render:
`target = GATE2_CENTER if y > PLANE_Y else GOAL` — the commanded displacement decays to zero
exactly at the aperture center and the target flip-flops across the plane, so the drone
oscillates at the threshold. Fixed stateless in `serve_gate_pin_oracle.py`: aim at a point
0.40 m BEYOND the plane until y < PLANE_Y-0.20, then the goal (goal is further -y, no
reversal). Rerun x5 in flight.

Standing implications regardless of rerun outcome: (a) under exact commands the flow positions
to ~0.15 m precision at a gate pasted into a scene it never saw during training — compound
RENDER does not corrupt execution, killing the render-OOD hypothesis for the wide-crossing;
(b) combined with the factorial, the prior-driven +x-wide crossing is now isolated to the
COMMAND SIDE near the gate (the no-clock prior's late/weak turn — "faithful to off-center
demos" amplified on the compound route), the last surviving explanation.

## FIRST REPRODUCIBLE COMPOUND COMPLETIONS: waypoint oracle 4/5 strict (gates 2/2 all trials) after two oracle fixes (2026-08-05)

Iterating the waypoint oracle in left_and_center (record flow gate_both_pin_rrr/4999, canonical
route, COMPOSE switch, 5 trials per rung, `serve_gate_pin_oracle.py`):

- or6 (carry-through fix): gate-2 transits go DEAD-CENTER all 5 (cross x 2.61-2.74 vs center
  2.756, z 0.94-0.98 vs 1.0) — but gate-1 latch LOST: the straight-line pull toward gate 2
  begins while the drone is inside gate-1's aperture slab and drags it back through
  wrong-direction (posthoc wrong=1). The demo route loops east around the post; a straight-line
  oracle lacks that route knowledge.
- or7 (+ east-clearance waypoint [2.05,0.85,1.15] until x>1.9, stateless 3-phase): BOTH gates
  latch 5/5 (steps ~76, ~277) — endgame short: post-transit approach to goal crawls ~0.05
  m/chunk, ends 0.9-1.25 m +x of the goal box at NCH=45, dwell 0.
- or8 (NCH 45->70): **4/5 STRICT SUCCESS (judge_compound: ordered transit + dwell>=16;
  dwells 121/115/63/21; the miss latched 2/2 gates but drifted in the endgame)**.

n=10 extension running (or9). PENDING HUMAN VIDEO REVIEW (rule 2) — overlay_or8_t*.mp4.
Reading: the flow EXECUTES the full compound task whenever commands are right — compound is
now a COMMAND-QUALITY problem end to end (selection: locator/VLM; near-gate turn: prior too
faithful to off-center demos; endgame: slow -x follow post-transit worth its own look). The
oracle encodes exactly three things a grounded command source must supply: which gate (locator
does 0.27/0.72 m in this scene), a through-the-aperture waypoint (carry-through), and route
topology around obstacles (east loop). Oracle route knowledge is the generalization target for
the VLM line, per Denis's directive to get oracle successes a VLM can learn to reproduce.

## COMPOUND 8/10 STRICT (n=10, claim-tier count) + RIGHT SCENE 5/5 STRICT under the waypoint oracle — the +x aim bias is COMMAND-SIDE everywhere (2026-08-05)

or9 (second 5 of the n=10): 4/5 — pooled with or8 the fixed waypoint oracle scores
**8/10 strict compound** (every trial latches both gates in order; both misses fail only the
goal dwell — the endgame, not the route). PENDING HUMAN VIDEO REVIEW.

RIGHT SCENE (`run_oracle_right.sh`, --geom right: carry-through 0.4 m along the diagonal
aperture normal, east clearance waypoint on the return, then goal; record flow, 5 trials):
**5/5 strict (transit + goal, wrong_dir=0, transit steps 103-140)** — vs the standing 0/10
with the one-hot no-clock prior (~1 m +x aiming bias, closest approach 0.65-0.85 m).
PENDING HUMAN VIDEO REVIEW (overlay_orR_t*.mp4).

Unification: with exact near-gate commands the SAME flow that missed by ~1 m in the right
scene threads it 5/5, and the compound duplicate gate goes dead-center. Combined with today's
factorial (momentum/splice/flow/render all eliminated), every remaining flight failure —
right-scene bias AND compound wide-crossing — is the COMMAND SIDE near the gate: the
demo-fit prior turns late/wide (faithful-to-demos + extrapolation), while flow execution is
verified everywhere tested. The oracle is a hand-coded scaffold (three ingredients: gate
selection, carry-through aim, route clearance); the open problem is LEARNING those.
First step running: oracle-distilled prior (state+onehot -> oracle c, train-fit R2 0.955
oracle rows / 0.915 demo rows, `oracle_distill_prior.pt`) closed-loop x5 on the compound
route. Grounded (VLM-feature) input swap is the step after.

## Oracle-distilled prior closed-loop: ROUTE LEARNED (4/5 both gates), endgame blurred — v2 with endgame oversampling training (2026-08-05)

`oracle_distill_prior.pt` (state+onehot MLP, 6k closed-form oracle rows for CFL + demo rows
for the other tasks, train-fit R2 0.955/0.915) served drop-in via serve_gate_pin_prog4 on the
record flow, compound route x5: **4/5 latch BOTH gates in order** (transits 279-325, ~30-50
steps later than the oracle's ~277-294) — the oracle's route knowledge (east clearance +
carry-through) IS learnable from (state, task). All 5 fail dwell: tails park at ~[2.0,-0.4,
0.82], ~0.2-0.4 m outside the goal box +x edge, with 220+ steps of horizon left — an
IMITATION-BLUR endgame: near the goal the oracle command magnitude -> 0 (relative fit error
explodes) and across the through-latch boundary the target jumps (MLP smooths the
discontinuity). v2 running: +3k endgame-ball rows and +2k boundary-band rows, 90 epochs.
Structured alternative if v2 stalls: predict the TARGET POINT and compute c analytically
(the discontinuity moves out of the regression).

**SCOPE RULE (Denis, 2026-08-05): oracle distillation is DIAGNOSTIC ONLY — the final product
(VLM or prior) must never train on oracle/waypoint labels.** Hand-set geometry imitated by a
learner is enumeration, not understanding — one-hot's failure class. The v1/v2 distillates
stand as capacity probes (route representable by the deployed MLP class: yes). The deployable
line remains: demo-fit priors + soft-pin correction + VLM for grounded selection.

**SCOPE RULE EXTENDED (Denis, 2026-08-05): no SIM GROUND TRUTH as product supervision — the
gate-locator included.** The locator regresses scene-YAML anchors, which exist only because we
built the environment ("used to create the sim environment, not for models to cheat off of").
Reclassified: the locator is a REPRESENTATION PROBE (it shows the fused VLM features carry
language-selectable gate location — 0.27/0.72 m in the unseen compound scene), not a
deployable component. Deployable gate localization must come from data a real robot has:
demonstrations (gate position is inferable from where demo trajectories converge/transit),
own observations, generic pretrained perception. Test of legitimacy for any supervision
signal: would it exist outside our sim?

## VIDEO VETO -> CLEARANCE AUDIT: oracle "successes" all clip the gate (aim-height bug); record board survives except CFR caveat; scoring rule upgraded (2026-08-05)

Denis's video review of the 15 oracle trials: compound successes "sus... most of them are
colliding with the gate." Built `gate_clearance.py` — min trajectory distance to the gate
GAUSSIAN CLOUD (what falsify COLLISION_GATE fires on), per scene with edits applied; body
threshold 0.18 m (half-extents 0.175/0.175/0.075).

- ORACLE VETOED: compound or8+or9 min clearance 0.001-0.005 m (all 10 contact); right scene
  0.004-0.085 m (0/5 clean). CAUSE: hand-set aim height — I used the safety-AABB mid-height
  z=1.0, but the AABB spans the POSTS (z 0.125-1.875); demos transit the physical opening at
  z~1.5. The oracle drove ~0.5 m below the hoop, through the lower frame. (Second instance of
  the region-box class bug: scorer/aim bounds must match the physical hoop — and the transit
  judge shares it, which is why it passed the clippers.)
- CALIBRATION: 15 demo flights = 0.28-0.38 m min clearance, all clean (threshold sound).
- RECORD BOARD AUDIT (traj_nop_left/right, traj_nopc_cfl/cfr): LEFT 10/10, RIGHT 10/10,
  CFL 10/10 clearance-clean (0.20-0.40 m, demo band). **CFR: 4/10 clean — 6 trials graze
  0.11-0.16 m** near [2.2,-0.1,1.2] (left post region on the from-right approach). Record
  board CFR 10/10 downgraded to "10/10 transit, 4/10 clearance-clean, grazes marginal".
- SCORING RULE (operating rule 2 extended): strict success = transit judge + CLEARANCE AUDIT
  (`gate_clearance.py`) + human video. Clearance added to all future screens incl. the soft
  battery post-hoc.
- Standing: executability-under-exact-commands is UNPROVEN again at the clean tier (transit
  yes, collision-free no). Diagnostic rerun with demo-derived aim (z~1.5 from demo transit
  altitude — data-derived, not YAML) queued post-battery to close the bound honestly.

## SOFT-PIN BATTERY (gate_aug_pin_rrr_soft/4999, sigma=0.7): first clean VLM-commanded transit — but the soft flow loses closed-loop competence under good commands (2026-08-06 ~00:30 UTC)

Teacher-forced: soft flow fwd ADE 0.043 (vs hard 0.025 — expected adherence cost), reverse
0.045 (vs 0.107), hover 0.004 (vs 0.059) — G2.1 PASS. Closed-loop, 3 command sources x 2
scenes x 5 (strict transit + clearance; videos
https://claude.ai/code/artifact/b3d941f2-e7de-4677-bad2-72e9e504bec5, pending Denis review):

- VLM MLP map, right scene: **2/5 transit, t1 CLEAN (clearance 0.27 m, transit @120) — the
  FIRST clean closed-loop gate transit from a pure VLM-feature command path** (hard-pin
  baseline 0/5 with zero transits anywhere, ever). t3 transited with contact (0.06 m).
- VLM ridge map: right 0/5 (but 3/5 reach the gate region, contacts 0.002-0.15 — off-center
  toward the low-x post); left 0/5 (wander).
- VLM MLP map, left scene: 0/5 with a consistent VERTICAL FLYAWAY (immediate climb to z 2-2.4,
  never approaches; clearances 0.8+). Echoes the fused-feature vertical-bias family.
- **CONTROL (demo-fit no-clock prior on the SAME soft flow): left 5/5 clean transit but 0/5
  goal (post-gate phase lost); right 0/5 transit (fly left of the gate and past). The hard-pin
  record config was left 9/10, right 10/10 FULL success.** Confound note: record flow is
  pi0_gate/gate_both; the matched hard-pin AUG flow (gate_aug_pin_rrr2) + prior control was
  not run — queued to separate soft-pin cost from aug-dataset/config cost.

Reading: sigma=0.7 buys exactly what LIBERO promised on the command-error side (VLM commands
now produce transits) but at a closed-loop competence price LIBERO did not show — the drone
task's precision demands are higher. Next levers: sigma sweep (0.3/0.5), longer training, and
the matched hard-aug control. The G2.1-passing teacher-forced numbers did NOT predict the
closed-loop degradation (fwd ADE 0.043 looked benign) — closed-loop screens remain mandatory.

## DEMO-AIM ORACLE v2: right scene 5/5 STRICT FULL SUCCESS; compound 5/5 both-gates with transit-clean crossings — executability bound re-established at the demo altitude (2026-08-06 ~01:00 UTC)

Aim heights corrected to DEMO transit altitude (z 1.51/1.49 vs the vetoed v1's YAML-mid 1.0).
Record flow, 5 trials each, transit + clearance in-chain (videos pending Denis review):

- RIGHT: **5/5 strict success (transit @66-79 + goal, wrong_dir=0)**. Clearance 1/5 clean
  overall, BUT all contact windows are LATE (steps 200-512, near the east post at
  [~1.1,-1.0,1.2]) — i.e. on the post-transit RETURN leg, an oracle route artifact; the
  transits themselves cross at the demo altitude without contact.
- COMPOUND: 5/5 latch both gates; 1/5 full success (dwell 32; others 0-10 — the endgame
  slow-approach again); 3/5 fully clearance-clean, and the two contacts are a 0.158 m gate-1
  graze (t2 @71) and an endgame linger near gate-2's post (t1 @552) — no transit collisions.

Conclusion the v1 veto suspended, now honestly held: WITH demo-derived aim, the flow executes
gate transits cleanly under exact commands in both scenes; residual contacts are oracle ROUTE
artifacts (return-leg waypoint too close to the east post; endgame loiter), not execution
noise at the aperture. The 0.45 m opening / ±0.05 m centering demand is meetable closed-loop.
Diagnostic line closes here per the sim-ground-truth rule; the learned-command line (soft
sigma sweep + matched hard-aug control, both running) inherits the demo-altitude lesson:
command targets must come from DEMO statistics, which naturally encode the true opening.

## DECONFOUND (hard-aug control) + FLOW-MATCHING PRIOR offline (2026-08-06 ~02:00 UTC)

HARD-pin AUG flow (gate_aug_pin_rrr2/4999) + demo-fit prior, 5x per scene:
- LEFT: **5/5 strict FULL success, 5/5 clearance-clean** (transits @58-72, clr 0.29-0.38).
- RIGHT: 2/5 transit, 0/5 goal, 0/5 clearance-clean (all graze/hit near the low-x post,
  0.004-0.143 m; drift toward x~0-0.3 vs gate center 0.56).
Deconfound of the soft battery (same data/config, only sigma differs):
- The LEFT goal-phase loss (soft: 5/5 transit but 0/5 goal) IS the soft pin's cost.
- The RIGHT-scene loss is NOT the soft pin: the AUG flows (hard AND soft) both fail right,
  while the record pi0_gate/gate_both flow does 10/10 — the aug dataset/config itself costs
  right-scene competence (candidates: trajectory-algebra aug distribution, pi0_gate_aug
  config diffs). Soft-pin sigma sweep remains justified for the left-goal cost; the right
  cost needs an aug-recipe look instead.

FLOW-MATCHING PRIOR p(c|state,onehot) (`flow_prior.py`, rectified flow 3x256, Euler-10;
same rows/split as the MLP prior; Denis's suggestion):
- Held R^2: 1-sample 0.906, 8-sample-mean 0.947 vs MLP 0.967 — near-parity; the 1-sample
  gap includes legitimate sample diversity, not just error.
- Spread probe REFUTES the OOD-detector hope: sample std at the off-manifold compound seam
  (0.06,0.06,0.03) is SMALLER than on-manifold (0.11,0.07,0.07) — small CFM nets are
  confidently wrong off-manifold too (seam cmd-mean [0.32,0.86,0.26], cos~0.55 to gate —
  better direction than the MLP's -0.86, but no calibrated uncertainty).
- Verdict so far: offline parity, no free uncertainty signal; the interesting question —
  whether SAMPLING (mode commitment) beats MSE mode-averaging closed-loop on endgame
  parking / turn timing — is still open; closed-loop screen queued when a GPU frees.
Saved: flow_prior_rrr4.pt.

## Flow-matching prior CLOSED-LOOP: parity with the MLP prior at screen tier (2026-08-06 ~02:40 UTC)

Record flow + flow prior (sample-1, Euler-10), 5x per scene, strict judge + clearance:
LEFT **5/5 full success** (3/5 clean; t3/t4 graze 0.12-0.17 m crossing high z~1.8);
RIGHT **4/5 full success** (4/5 clean; t1 missed with a post contact). Reference: the MLP
prior record config is 9/10 / 10/10. A sampled generative command head is closed-loop VIABLE
as a drop-in — first flow-matching stage-1 result. Discriminating tests running: (a) compound
route x5 (where MSE mode-averaging is the suspected wide-turn/endgame mechanism); (b) the real
target per Denis — TWO-STAGE FLOW IN THE VLA: CFM head p(c|phi) on fused VLM features (no
one-hot, no state input; `train_vlmflow_head.py`, cache-comparable to vlmc_ridge/mlp_rend),
then closed-loop on the soft flow (battery-comparable rows).

## Two-stage flow updates: VLM-feature CFM head offline parity; mode-averaging REFUTED as the compound mechanism (2026-08-06 ~03:20 UTC)

- `train_vlmflow_head.py` (stage 1 = CFM p(c|phi), fused VLM features ONLY — no one-hot, no
  state): held R^2 1-sample 0.847 / 8-mean 0.911 vs ridge_rend 0.809 / mlp_rend 0.926 on the
  SAME held rows. Offline parity for the VLA-native stage 1. Closed-loop screen next (soft
  flow, battery-comparable).
- Flow prior COMPOUND x5 (record flow): 0/5, same +x-wide gate-2 signature as every MSE prior
  (crossings x 3.25-3.42 past the 3.16 edge; 4/5 latch gate 1 only; t1 0/2). MODE-AVERAGING
  IS REFUTED as the wide-crossing mechanism — a sampling head draws the same late/wide turn,
  so the deficiency is command CONTENT on the compound approach (prior trained on original-
  scene demo routes), consistent with the oracle-v2 lesson (demo-altitude carry-through aim
  fixed it). Compound needs better command targets (demo-statistics-derived aim), not a
  different head.

## Two-stage head LANGUAGE DIFFERENTIATION: same frame, different prompt -> command flips on the true axis (2026-08-06 ~03:50 UTC)

`test_vlmflow_language.py` (held identical-frame pairs from the aug cache): fwd-vs-back
(870 pairs) cos(Δcmd_sampled, Δtrue) = **0.983±0.052** at magnitude ratio **0.95±0.13** —
the CFM(phi) head's sampled c responds to TEXT, correctly, matching the deterministic ridge's
0.999 grounding but generatively. fwd-vs-hold (18 pairs): |cmd| 0.61 -> 0.32 m (target ~0) —
partial stop-word suppression, consistent with hover-row scarcity (oversample if needed).
Scope: within-scene contrasts only (direction/stop words); task-word selection is
scene-confounded in single-gate scenes; paraphrase remains open.

## COMMAND-FIELD COMPARISON (5 heads, fixed states+frames): the closed-loop chasm is a MISSING RESTORING FORCE off-manifold, not a static bias (2026-08-06 ~05:00 UTC)

`cmd_compare.py`: same chunk-start states, freshly rendered frames, LEFT prompt; heads =
{MLP prior, flow prior} (state+onehot) and {vlmc_ridge, vlmc_mlp, vlmflow} (fused features).

- ON THE GOOD ROUTE (hard-aug prior 5/5 trajectory): ALL FIVE heads are near-equally good —
  mean |err| 0.47-0.74 m, no static z-bias (first-6 z-cmds ~0 for state AND feature heads,
  matching ref). The VLM heads' commands are FINE on-manifold with matched rendering — the
  static-feature-bias hypothesis for the flyaway is dead.
- ON THE FAILED vlmflow ROUTE: the drone climbs (+0.76 m over the first horizon) although
  the 8-mean commands at those states say z~0 — drift starts from 1-sample noise + the soft
  flow's weak pin authority. THEN the heads diverge: once high (z>2), the STATE heads command
  corrective descent (mlp_prior z-cmd -0.41/-0.45) — a global restoring field from state
  extrapolation — while ALL THREE feature heads go flat (z-cmd ~0 at z 2.2+): the rendered
  cache never saw those viewpoints, features go OOD, and the command field has NO restoring
  force. Runaway is then self-sustaining.

Mechanism named: state-based heads fail gracefully (restoring), feature-based heads fail
neutrally (no correction) — and closed loop punishes neutral failure absolutely. Fix
candidates, deployable-line-legit: (1) HYBRID head input [state, phi] (state supplies the
restoring field, phi supplies language/task; still no one-hot) — training now; (2) on-policy
feature coverage (extract features along own-rollout states incl. off-manifold, demo-derived
targets — DAgger-flavored); (3) server-side k-sample median + temporal EMA on c (cut the
1-sample noise that seeds the drift).

## HYBRID head closed-loop (soft-0.7 flow): FIRST FULL STRICT SUCCESS from a language-grounded command source; flyaway tamed (2026-08-06 ~02:40 UTC)

CFM([model_state, fused phi]) -> c, k=4 median, on gate_aug_pin_rrr_soft/4999, 5x/scene
(videos in https://claude.ai/code/artifact/d497da6a-a829-4541-bb2a-d40b6c69fc39, pending
Denis review):
- RIGHT t1: **transit @91 + goal + clearance-clean 0.26 m — the first full strict task
  success EVER from a VLM-feature command path (no one-hot anywhere)**. Right 1/5 overall;
  the other 4 drift left of the gate (x -0.3..-0.55).
- LEFT: 1/5 transit (t2 @96, 0.051 graze, no goal); NO vertical flyaway anywhere — failures
  cap at z~2.1 and drift laterally. The state input's restoring field works closed-loop as
  the offline probe predicted (z-cmd -0.39 at z=2.5).
- Same-flow baselines: feature-only heads 0/10 transits (runaway); state-prior control 5/5
  left transit-only + 0/5 right. The hybrid is the only arm to complete a full task on this
  flow. Still a screen-tier lead (1/10 full), not a claim.
Next levers queued: sigma=0.35 flow battery (prior/vlm-map/hybrid arms armed), sigma-rand and
learned-U trains behind it; on-policy feature coverage remains the untried fix for the
lateral drift (features OOD off-route in x/y even when z is held).

## COMMAND JITTER quantified (Denis's video observation) + free smoothing transform (2026-08-06 ~03:10 UTC)

Along the fixed good-route points: MLP prior jitter 0.099 m/chunk (smoother than the true
motion's 0.115); feature heads 0.14-0.24 — 1.5-2.5x jerkier, matching the jerky flight Denis
saw. Decomposition: CFM same-frame RESAMPLE moves c by 0.230 m — 1-sample CFM jitter is
almost entirely SAMPLING noise. Transforms (jitter/err): k=8 mean 0.223->0.130 / err slightly
BETTER; +EMA(0.5) -> ~0.08-0.10 / err unchanged; shrinkage-to-demo-mean HURTS err (0.68->0.76,
rejected). Deployed: serve_gate_pin_hybrid now k=8 MEAN + EMA(0.5) with position-jump episode
reset — feature heads now command at MLP-prior smoothness for free. Residual accuracy gap
(err 0.72 vs 0.47) is separate and remains the on-policy-coverage target.

## SIGMA=0.35 BATTERY: the trust dial works — prior recovers AND beats hard pin on right; smoothed hybrid transits right 5/5 clean (2026-08-06 ~04:40 UTC)

gate_aug_pin_rrr_soft035/4999, three arms x 2 scenes x 5 (strict + clearance; videos
https://claude.ai/code/artifact/394cf6fd-520e-4bab-96b9-d6e827789c0c, pending Denis review):
- STATE PRIOR: left 5/5 transit (2 full, 4/5 clean); right 5/5 transit (3 full, 5/5 clean).
  Sigma story on the AUG config, right-scene transit: hard 2/5-with-contacts -> 0.35 5/5
  CLEAN -> 0.7 0/5. Moderate soft pin doesn't just recover competence, it FIXES the aug
  flow's right-scene problem (best aug-config result to date; record non-aug flow still the
  overall best at 9-10/10 full).
- SMOOTHED HYBRID (k=8 mean + EMA0.5): right 5/5 TRANSIT, all clearance-clean (@160-266) —
  a language-grounded, one-hot-free command source now transits reliably; 0/5 goal (endgame
  gap remains; transits late). Left 0/5 (drifts, no flyaway). Confound noted: smoothing AND
  sigma changed together vs the 1/5 on soft-0.7 — attribution needs the smoothed head on 0.7
  or unsmoothed on 0.35 if it matters.
- VLM MLP MAP: 0/10 (left far; right reaches the gate area, 3 contacts, no thread) — the
  deterministic map is now clearly the limiting component, not the flow.
Goal-phase completion on aug flows is the common residual (prior full-success 5/10 here).

## BASIN PROBE: the accuracy-vs-behavior paradox resolved — VLM map's stability basin is the cache tube (~0.3 m); the prior's grows with deviation (2026-08-06 ~05:30 UTC)

Denis's challenge (raw 5-dim c near-identical on held rows, VLM map even better on c2/c4 —
yet prior behaves far better closed-loop) answered with a perturbation-response probe
(`probe_gain2_render.py` + gain eval): render frames at good-route points +- {0.25,0.5,1.0} m
in y/z, measure feedback gain = -Δcmd/Δpos.
  prior     y .48/.44/.35   z .39/.67/.73   (restoring force GROWS with deviation)
  vlm map   y .64/.37/.09   z .23/.21/.11   (stabilizing only inside ~0.3 m; blind at 1 m)
Closed-loop competence = BASIN WIDTH, not on-route accuracy; offline R^2 cannot see it.
Also: near-route local gains are PARITY — the map is not statically worse anywhere the
data covers. AUG-vs-NONAUG answer (same session): non-aug record flow still dominates core
tasks (9-10/10 full vs aug-best 2-3/5); aug buys hover/reverse vocabulary, not performance.
FIX (data, not architecture): FAT-TUBE coverage — render frames at +-0.5-1.0 m offsets
around demos, label with nearest-demo continuation c (demo statistics — legitimate), retrain
feature heads, and gate on this probe (gain at 1.0 m) before any closed-loop spend.
The basin probe joins the offline instrument set alongside the steer diagnostic.

**SCOPE NOTE on fat-tube (Denis, 2026-08-06): sanctioned as an experiment, but it's a
small-data PATCH, not the destination** — return-to-route labels imply movement is only valid
on the training trajectory (route imitation, not understanding); cfground's lesson repeats:
the scalable answer is DATA whose natural diversity covers the state space (many routes/tasks/
viewpoints), after which basin width comes for free and no tube construction is needed. Treat
tube-trained heads accordingly: useful to prove basin width is the binding constraint and to
unblock screens; not a component we should still need at scale.

## FAT-TUBE PASSES THE BASIN PROBE: pure-feature gain at 1.0 m goes 0.05-0.15 -> 0.62-0.70 — the chasm was a DATA COVERAGE problem (2026-08-06 ~07:10 UTC)

Retrained on 2400 tube rows (LEFT+RIGHT scenes, offsets to 1.0 m, return-to-route labels):
  vlmflow rend->fat: y/z gain @1.0 m 0.05/0.15 -> 0.62/0.70 (state-free wide basin!)
  hybrid  rend->fat: 0.12/0.45 -> 0.71/0.77 (near-full correction at all radii)
Causal confirmation of Denis's data-problem hypothesis: nothing about fused VLM features
prevents a wide stability basin — the cache simply never contained off-route views. Held R^2
on tube rows 0.788/0.829 (harder row distribution, not comparable to the 0.91-0.93 thin-cache
numbers). Fat-tube remains patch-tier per the scope note (route-imitation labels); at data
scale the basin should come from natural diversity. Closed-loop screens of both fat heads on
the sigma=0.35 flow next (chasm's final test: does basin width convert to flight?).
Chain note: the first extract run tripped the NpzFile-subscript memory bomb AGAIN
(fat_tube_gen loops indexed the npz directly; watchdog caught it; arrays now materialized) —
the 2026-08-04 rule holds: never index an NpzFile in a loop.

## Fat-only heads FAIL closed-loop (0/20, wander) — basin gained, accuracy lost; UNION heads restore both offline (2026-08-06 ~08:40 UTC)

Closed-loop on sigma=0.35: fat-only vff 0/10, fat-only hybrid 0/10 (the thin hybrid had 5/5
right transits on this flow — fat-only training DESTROYED on-route competence: tube-only 2.4k
rows, held R^2 0.79 vs 0.91). Wander signature: +x/+y beyond the tube's thin x-coverage, then
blind. Lesson: coverage must ADD to accuracy, not replace it. UNION training (thin 12.4k
on-route rows, 4 tasks + 2.4k tube rows): held on-route R^2 back to 0.905 (feature-only) /
0.902 (hybrid) WITH tube data. Union basin probe + closed-loop screens running.

## UNION HEADS: first reliable transits from a STATE-FREE, ONE-HOT-FREE command source — right 4/5 (2026-08-06 ~09:30 UTC)

Union basin probe: both heads hold gain at all radii (vfu 0.55-0.77, hyu 0.63-0.84). Closed
loop on sigma=0.35: **pure-feature union head 4/5 RIGHT transits (@183-245)** — progression
on this flow: thin features 0/10 -> fat-only 0/10 -> union 4/5. Denis's data-coverage
hypothesis validated closed-loop: vision+language features alone, given coverage, command
reliable transits. Goal phase 0 (aug-flow endgame residual, shared by all arms incl. the
state prior). INVERSION: hybrid-union 1/5 right vs hybrid-thin's 5/5 — with state available,
tube labels shift the state-conditional mapping; un-chased for now (pure-feature is the
north-star architecture). LEFT 0/5 for all feature heads — per-scene coverage gap, separate
investigation. Videos: overlay_un_*.mp4.

## LEARNED-U ANATOMY: joint training does NOT rediscover the coarse code (2026-08-06 ~10:20 UTC)

gate_aug_pin_learnu/4999 (K=5, attention readout, lam=1, sigma=0.35): principal angles of the
learned U vs RRR U = 19/65/90/90/90 deg; vs net-displacement 4-space = 17/64/90/90; DCT
modes 0-2 carry only 16-32% of column energy (RRR: ~100%); 3/5 columns yaw-dominant.
INTERPRETATION: the flow loss co-opts the pinned channel for the FINE/high-frequency
component (pinning pays most where the flow predicts worst), and lam=1 predictability barely
resists — the coarse factorization is NOT the natural optimum of naive end-to-end training.
North-star relevance: hand-derived (RRR) or strongly-regularized factorization is doing real
work. Follow-ups queued behind bigger trains: lam=10, RRR warm start, low-frequency prior on
U. Closed-loop eval of this checkpoint deferred (its c is mostly fine-detail — the prior/head
machinery has nothing meaningful to command in that basis).
Launched on GPU1: sigma=0.35 on the RECORD recipe (gate_both_pin_rrr_soft035) — claim-tier
candidate pairing.

## SIGMA-RANDOM battery (CFG-dropout flow): universal clean transits, dead goal phase (2026-08-06 ~15:50 UTC)

gate_aug_pin_rrr_softrand (sigma~U(0,0.7) per sample), strict: prior 10/10 TRANSIT both
scenes, all clearance-clean (@69-176) — but 0/10 goal; vmlp map 3/5 right transits (its best
on any flow; 0/10 on 0.35); hybrid-union 1/5 right. Reading: the trust-spectrum flow
CORRECTS command error better than any fixed sigma (map transits!) but never commits to the
goal/stop phase — plausibly because high-sigma training samples teach it to distrust exactly
the small endgame commands. Sigma=0.35 remains the best aug flow overall (5/10 full).
NOTE: brand_scores.txt echo labels say "035/" — sed artifact; the flow IS softrand.
Record-recipe sigma=0.35 checkpoint landed; claim-tier battery launched (prior x10/scene +
union feature head x5/scene).

## CLAIM-TIER: record recipe + sigma=0.35 — LEFT 8/10 FULL (10/10 clean), RIGHT 10/10 clean transit but 0/10 goal (2026-08-06 ~17:00 UTC)

gate_both_pin_rrr_soft035/4999, prior x10/scene + union feature head x5/scene:
prior LEFT 8/10 strict full success (hard-pin record: 9/10) — soft pin at 0.35 preserves
record-level left competence; prior RIGHT 10/10 clearance-clean transits in a tight @113-123
band but 0/10 goal (hard-pin record right: 10/10 full) — the soft flow specifically loses the
right POST-GATE/goal phase. Union pure-feature head on the record flow: right 4/5 transits
(@128-141), left 0/5 — the state-free grounded head transfers across flows. One endgame fix
from a board entry; sigma-conditioned flow (training) with SIGMA_INFER->0 near goal is the
designed lever. Videos overlay_rb_*.mp4.

## POSE-BOTTLENECK DECOMPOSITION: why the MLP prior wins — the deep answer (2026-08-06 ~17:05 UTC)

Since true c is ~99% pose-driven here, any pixels->c head implicitly recovers pose. Measured:
- POSE PROBE (phi->xyz, union rows): held 0.141 m mean (p90 0.218), yaw 0.06 rad; on basin
  frames 0.203 m on-route, 0.451 m at 1.0 m offsets. Pixels determine position to ~15-20 cm.
- CASCADE (mlp prior fed the VLM pose estimate): |field Δ vs mlp(true)| = 0.212 m — command
  error ~= pose error (g is near-isometric); gains flatten (0.15-0.45 vs 0.39-0.73).
- DIRECT union head: |Δ| = 0.543 m = 0.21 pose ceiling + ~0.33 MAPPING SLOP (2.5x the bound).

DECOMPOSITION: the VLM command gap = (a) ~0.2 m information ceiling (pose-from-pixels at
current pooled-feature quality) + (b) ~0.33 m fixable mapping slop. (b) is addressable
(attention readout — running; more data). (a) is a floor NO pixels-only head can beat, and
0.2 m command error vs the +-5 cm gate-centering demand means pixels-only c cannot thread
gates at current feature precision — the deep reason the MLP prior (pose as input, error 0)
"points right" everywhere, including the start (on-route pose err 0.2 m — Denis's
observation). ARCHITECTURE CONSEQUENCE: the state-for-geometry / VLM-for-semantics division
on the record board is information-forced in this regime, not a preference; hybrid heads are
principled. In north-star regimes where c depends on semantics pixels alone know, the VLM
channel becomes necessary rather than merely imprecise. Finer tokens (full 16x16, higher res)
are the only lever against ceiling (a) — the 4x4-pooled attention head tests a step of this.

## CHANNEL-DISAGREEMENT PROBE: hybrids are FEATURE-dominated (lazy-modality-toward-state FALSIFIED) (2026-08-06 ~17:50 UTC)

Mismatched (state,phi) pairs on the gain2 frames: image-only change moves the thin hybrid's
command ~85% of the consistent response (0.53/0.63 at 1 m); state-only ~45% (0.28/0.63).
No joint-OOD blowup (mismatches sub-additive). READING: concat hybrids are feature heads
with a state trim — they inherit the feature channel's pose ceiling and jitter at ~full
weight while the perfectly-informed state channel gets a minority vote; that's why adding
state never rescued the feature head. Union training shifts weight toward state at large
offsets (tube labels are pure position functions) — matches its different failure profile.
Repair direction (evidence-based): give the STATE pathway architectural priority (features
as low-weight additive correction), or skip concat entirely — division of labor per the
pose-bottleneck finding. Modality dropout on state would be the WRONG direction here.
Probe lesson: consistency-only perturbation (basin probe) overrates multimodal heads — the
disagreement probe joins the instrument set.

## REAL vs SYNTH RRR BASES DIFFER (Denis's question, direct computation) + LADDER LAUNCHED (2026-08-06 ~19:00 UTC)

RRR fit per domain (same recipe, ridge lam=10, K=5): principal angles U_synth vs U_real =
[6.2, 14.7, 26.3, 47.0, 60.8] deg — a shared ~3-dim displacement-like core, two divergent
directions. U_real tilts away from the net-displacement 4-space ([9,18,39,49] vs synth's
[7,11,19,42]): real flight's predictable structure carries extra timing/curvature. Real
chunks' c coordinates across bases: two dims near-identical (|corr| .94/.95), rest .45-.64.
The deployed pin_U ~= synth basis (1.5-18 deg, last dim 64). Explains part of sim->real
negatives (commanding real flights in a basis under-expressing real dynamics). FIX for the
real line: pool sim+real chunks when fitting U (mixed-prior lesson at the basis level) or
gate_invariant_U constructions. Synth lines unaffected.
LOW-DATA LADDER launched (flagship): local/gate_nav_n{12,40,160} built (pyarrow schema-
preserving subsets, nested, per-task balanced; pandas rewrite broke HF image metadata — fixed),
configs pi0_gate_n*, shared norm stats + synth U; pin arms GPU1, scratch arms GPU0 after the
attention extraction. sigma-dial battery shelved per Denis (minimal-first); sigcond checkpoint
kept for later interpretability work.

## LADDER PASS 1: confounded — flagged, corrected pass launched (2026-08-07 ~05:00 UTC)

5 trials/cell, prior commands, strict+clearance (videos
https://claude.ai/code/artifact/c7fbc5f8-2452-4c3b-9b06-519e57153ee2):
transit/10 (L+R): pin 1->9->1 vs scratch 9->6->10 across {12,40,160}; full success ~0
everywhere except n160-pin right 1/5; clearance-clean pin 17/30 vs scratch 0/30.
CONFOUNDS: (1) steps scaled with data (2k/3k/5k) — pin non-monotonicity (n40 L 5/5 -> n160 L
0/5) indicates optimization, not data, drives part of the curve; full success far below the
300-demo record (9-10/10) => undertrained ladder. (2) scratch arms served through the pin
server — pinned noise is uninformative to a flow trained without the pin, so scratch =
vision route-following; its transits CONTACT the gate (0/30 clean). Interpretable signal so
far: pinned flows fly demo-band clearance at all data sizes; scratch clips at all sizes.
CORRECTED PASS: n12/n40 pairs retraining at 5k steps (equal optimization); n160 pair already
5k. ATTENTION HEAD verdict (same day): held R^2 0.912 vs pooled 0.905; basin y/z@1.0
0.93/0.72 vs 0.77/0.69, but @0.5 dips (0.52/0.57) — not a clean win; pooled head retained
per the pre-registered decision rule.

**LADDER EVAL FIX (Denis, 2026-08-07): scratch arms must not be served through the pin
server.** The pinned coordinates carry |c| ~ 4-6 per dim where a scratch flow's training
noise has unit std — 4-6 sigma off-distribution in 5 directions; the "scratch clips the gate"
observation from pass 1 may be partly an artifact of this. Added serve_gate_plain.py
(policy.infer with its own N(0,I) noise); the corrected-pass evaluation serves scratch arms
plainly and re-evaluates all cells at equal training steps (5k). Pass-1 scores preserved in
ladder_scores_pass1.txt for comparison.

## 2026-08-07 — Ladder eval pass 2 INVALID: stale-server bug (fixed, rerunning)
Pass-2 scores (archived `ctxrun/ladder_scores_pass2_invalid.txt`) are garbage: every cell
after the first was served by the FIRST cell's server (gate_n12_pin5k + pin prior). Two
stacked harness bugs: (1) the between-cell kill pattern `serve_gate_p[a-z]*[.]py` matches
serve_gate_plain.py but NOT serve_gate_pin_prog4.py (`_prog4` breaks the [a-z]* class), so
the first pin server survived on port 8821; (2) serve_gate_plain.py printed "ready on ws"
BEFORE binding, so the log-grep readiness check passed while the new server crashed with
EADDRINUSE and rollouts silently hit the stale server. Signature that exposed it: all 12
cells nearly identical (left arms all clipping the frame at the same point, right arms all
timid), including the unchanged n160 checkpoints that had scored differently in pass 1.
Fixes: kill pattern `serve_gate_.*[.]py`; port-free guard (`ss -ltn`) before launch;
readiness = port actually LISTENing, not a log line; serve_gate_plain pre-binds a probe
socket before the slow model load. Rule for future harnesses: readiness and exclusivity
must be checked at the PORT, not in logs; uniform scores across supposedly different arms
means suspect the harness first.

## 2026-08-07 — Ladder corrected pass: valid harness, two findings, stratified rerun launched
Rerun with the fixed harness completed; cells now differ and the unchanged n160 checkpoints
reproduce their pass-1 signatures (harness confirmed sound). Results (transit/full/clean per
10): n12 pin 2/0/5, n12 scratch 3/0/5, n40 pin 7/0/5, n40 scratch 0/0/10, n160 pin 0/0/8,
n160 scratch 2/0/10 — full success 0 everywhere. Page (curve + all 60 videos):
https://claude.ai/code/artifact/c7fbc5f8-2452-4c3b-9b06-519e57153ee2
(1) DATASET-COMPOSITION CONFOUND: first-n-per-task subset selection filled every rung's
left/right demos with REAL-domain episodes only (orig eps 0-99 sort before synth 200-299),
while the closed-loop eval runs in the gsplat synth renderer; the working record flow trained
on a 50/50 real/synth mix per gate task. Explains the across-the-board weakness and arms that
never leave the start. Fix: build_subsampled_datasets.py now interleaves real/synth per task
(n12 gate tasks 2+1 each, n40 5+5, n160 20+20; still nested); all six flows retraining at 5k
steps (exp names gate_n{N}_{pin,scratch}strat), eval chain armed. Confounded-pass scores
archived as ctxrun/ladder_scores_nostrat.txt.
(2) SERVING CONTROL FINDING (confirms Denis's 2026-08-07 concern quantitatively): plain-served
n12/n40 scratch arms on the right scene do not move from the start (min clearance 1.30 m at
step 0) where the pin-served pass-1 cells showed 5/5 dirty transits — those transits were an
artifact of pinned noise driving a pin-free flow, not competence. Scratch arms must always be
served plain (serve_gate_plain.py).

## 2026-08-07 — Per-domain U (queue #3): NEGATIVE — deployed basis retained
Stage A (experiments/rung3/per_domain_u.py + eval_perdomain_heldreal.py). Bases refit with
one recipe (ridge lam=10, K=5, stride 8) and authoritative labels (clean-label synth L/R eps
100-199; real all 100; caches vlm_feat_gate_prefix_{synthLR,real}.npz — the old cache's
geometric-label contamination NOT inherited). Principal angles: synthLR vs real
[5.9 39.8 56.4 76.4 83.2] — only ~1-2 shared directions (more divergent than the 2026-08-06
ridge measurement); pooled vs deployed [1.9 3.3 3.6 9.8 51.8] — pooled is nearly the deployed
basis. Decision metric, held-out REAL c-R2 of the state+onehot prior:
  real U + real data           0.504   (per-dim tail 0.18/0.16 — barely predictable)
  deployed U + real data       0.685
  pooled U + synth+real data   0.692
  deployed U + synth+real data 0.717   <- best (existing noprog_prior_mixed.pt)
Conclusions: (1) refitting U on real or pooled chunks does not improve command
predictability — the RRR objective (VLM-feature-predictable chunk variance) selects real-data
directions the state prior cannot predict; (2) real chunks are harder in ANY basis (best 0.72
vs 0.97 synth) — a data property, not a basis mismatch; (3) deployed basis retained, the
contingent closed-loop flow retrain is cancelled (contingency "clearly beats" failed).
Artifacts: pin_U_gate_rrr_{real,pooled,synthLR}_k5.npy, noprog_prior_{realU,
realdata_synthU,pooledU}.pt (diagnostics, not deployed).

## 2026-08-07 — VLM-error-matched soft pin v1 (covpin): NEGATIVE
Battery on gate_both_pin_rrr_covpin/4999 (record recipe, pin noise ~ N(mean, Sigma) from the
union head's measured errors; mean is negligible at 0.02-0.06 of a std — the change vs
isotropic is the correlation structure). Same command sources as the references.
  prior arm: LEFT 0/5 transit (4/5 reach the gate region but never cross clean; 1/5
  clearance-clean) — vs record hard pin L9/10 and record sigma0.35 L8/10 FULL with the SAME
  prior. RIGHT 5/5 transit (fast, @116-128), 3/5 clean, 0 full.
  union head arm: RIGHT 2/5 transit vs 4/5 for the same head on the sigma0.35 aug flow;
  LEFT 0/5.
Verdict: error-matched noise is substantially worse than isotropic at comparable magnitude
(per-dim ratios 0.25-0.39 vs 0.35). Mechanism hypothesis: the head's errors are correlated
with task-relevant command directions (errors correlate with signal), so training the flow to
tolerate THAT noise teaches it to ignore the specific subspace where commands differ —
selective degradation (left aim collapses, right transits survive) rather than uniform
softening. Isotropic noise spreads distrust evenly and mildly. Per the pre-registered rule
(v2 only if v1 improves closed-loop): v2 is NOT triggered; line closed. Scores:
ctxrun/covpinb_scores.txt; videos ctxrun/overlay_covpinb_*.mp4.

## 2026-08-07 — Component-wise decomposition battery designed + armed (Denis's direction)
Denis's question sharpened the mechanism analysis: in the pinned subspace the flow target is
v = noise - actions with U^T noise = U^T a, so the target velocity in U-coordinates is ZERO —
the pin channel is an identity map BY CONSTRUCTION (straight lines in the pinned subspace are
the design, not an artifact; explains 1:1 error passthrough / brittleness / covpin). The
method's entire value claim therefore lives in whether the complement (1595 dims) is
generated CONDITIONED on c. Existing evidence: contradictory-c commands produce full
contradictory ROUTES (nonlinear in c), G2 vocab composition, toy conditioning arm 26x worse.
Missing decisive control now built:
- serve_gate_pin_democ.py — demo-oracle command: c = inv-distance-weighted k=5-NN over demo
  (state, chunk-c) pairs, same task, demos only (no sim ground truth). Smoke test: exact c
  recovery at demo states. This is the perfect-pin execution-ceiling bound.
- serve_gate_additive.py — CONTROL: plain (scratch) flow + post-hoc algebraic overwrite
  a' = a + U(c - U^T a) in normalized chunk space, same demo-oracle c (edit verified exact to
  4e-6). Identical coarse content, no pin training — isolates whether pin TRAINING makes the
  complement cooperate.
- run_component_battery.sh (armed behind the stratified ladder eval, GPU0): demo-oracle on
  record flow + n160/n40/n12 pinstrat, additive on n160 scratchstrat; 5x{L,R} each, strict
  judge + clearance. Decomposition it yields: execution ceiling (oracle c per data size) vs
  channel value (pin vs additive at n160, matched data/steps) vs prediction gap (prior/VLM
  vs oracle, from existing batteries).

## 2026-08-07 late — Disk-full incident: strat trains crashed at step-2000 checkpoint write
All six stratified trains died with ENOSPC (1.9 TB disk 100% full). Cleanup approved by
Denis: orbax tmp dirs from the failed writes; ALL train_state (optimizer-state) subdirs
across pi0_gate*/pi0_libero* checkpoints (params retained everywhere — every result stays
servable/evaluable; training resumability lost); superseded pass-1 ladder exps
gate_n{12,40}_{pin,scratch}. Freed 240 GB. run_ladder_strat.sh now deletes each train's
train_state after verifying params — checkpoint growth on this box was optimizer state.
Trains relaunched both GPUs; eval + component battery chains remained armed throughout
(they wait on params dirs).

## 2026-08-08 — STRATIFIED LADDER RESULT (clean design): pin carries movement at 12 demos
Domain-stratified subsets, equal 5k steps, plain scratch serving, full-data state prior held
constant across rungs. Per 10 (L+R, 5 trials/cell — LEAD tier, not claim tier):
  transits:      pin 10/10 at ALL of n=12/40/160; scratch 6/7/5.
  full success:  pin 2 (n12), 3 (n40), 1 (n160); scratch 0 at every size.
  clean:         pin 7/9/10; scratch 10/5/8.
Reading: with the coarse action supplied through the source noise, TWELVE demos suffice for
10/10 gate transits and the first full completions; without it, no ladder size completes a
single task. This is north-star claim (b) in its first quantitative form. Caveats: 5
trials/cell; full-success rates low everywhere (goal phase remains the weak end); n160 pin
full success (1/10) below n40 (3/10) — within protocol noise at this n. Videos + curve:
https://claude.ai/code/artifact/c7fbc5f8-2452-4c3b-9b06-519e57153ee2 (pending Denis video
review). Component battery (demo-oracle ceiling + additive control) now running on GPU0.

## 2026-08-08 — COMPONENT DECOMPOSITION BATTERY (execution vs channel vs prediction)
Cells (10 rollouts each, strict judge + clearance): demo-oracle c (kNN over demo state->c,
demos only) on record/n160pin/n40pin/n12pin flows; additive control (n160 scratch flow,
post-hoc a' = a + U(c - U^T a), same oracle c). Results (transit/full/clean per 10):
  record+democ  10 / 7 / 10   <- execution ceiling; includes RIGHT full successes 3/5
  n160pin+democ 10 / 2 / 10
  n40pin+democ  10 / 5 / 6
  n12pin+democ   8 / 0 / 4
  n160scr+ADD   10 / 2 / 4    <- matches pin on transit/full; clean collapses (clips 0.002m)
Three conclusions:
(1) RIGHT GOAL PHASE IS COMMAND-SIDE: with oracle commands the record flow completes right
full tasks 3/5 (prior commands: 0). The prior never commands the hover/settle tail; the flow
can execute it. Queue item "record-soft goal phase" reframed: fix the PRIOR's goal-phase
commands (short), not the flow.
(2) CHANNEL VALUE, REFINED: the additive control matches the pin-trained flow on transits
and full successes at n160 — the coarse channel steers largely ALGEBRAICALLY, even patched
onto a pin-free flow post-hoc. What pin training buys at n160 is trajectory coherence:
clearance-clean 10/10 vs 4/10, additive runs clip the gate to 0.002 m. Denis's "straight
lines" intuition is half right: the channel is identity by construction; the trained
complement's cooperation shows up in HOW cleanly the commanded motion is realized, not in
whether it happens. Caveat: 5 trials/cell; the additive host flow saw the same demos.
(3) EXECUTION still improves with data under perfect commands (n12 8/0/4 -> record 10/7/10):
goal phase + cleanliness are the data-hungry parts; transit steering is nearly free.
Videos (all 50): https://claude.ai/code/artifact/1341df28-c8a4-4060-80bf-034e26546a40

## 2026-08-08 — TAIL-WEIGHTED PRIOR: 10/10 FULL SUCCESS on the record flow (lead tier)
Chain from the component battery's finding (goal phase is command-side): offline diagnosis
(goalphase_diag.py) -> prior tracks c well early/transit (err 5-15% of a std) but tail err
jumps 3-5x and |c|_prior > |c|_true (commands motion where demos settle; never says stop).
Fix: retrain the clockless prior with tail rows (frac>=0.75) upweighted 4x
(make_progress_prior4.py TAILW env; noprog_prior_rrr4_tailw4.pt; aggregate held R2 unchanged
0.9705; tail err -30-40%, overshoot gone). Closed-loop on gate_both_pin_rrr/4999:
  FULL SUCCESS 10/10 (L 5/5, R 5/5) — first right-scene full successes from a learned prior;
  previous best L 8/10 / R 0. Clearance 7/10 clean (two 0.17x grazes at the 0.18 threshold,
  one 0.096 dip). Transits @75-93. LEAD TIER (5/side): 10/side extension + ladder-pin re-eval
  with the fixed prior launched (tailw4x chain). Phase input remains observational (state
  only, no clock) — consistent with the no-wall-clock rule.

## 2026-08-08 — TAIL-WEIGHTED PRIOR AT CLAIM TIER: 19/20 full success (record flow)
10 trials/side on gate_both_pin_rrr/4999 + noprog_prior_rrr4_tailw4.pt:
  LEFT 10/10 full (9/10 clearance-clean); RIGHT 9/10 full (6/10 clean; grazes remain the
  right-side weak point). Pending human video review (record-board rule) — page:
  tailw4 20-rollout artifact. If confirmed on video this supersedes the 39/40-transit board
  line with the first full-success board entry: 19/20 FULL, simplest system yet (hard pin,
  clockless state+onehot prior, tail-weighted training).
LADDER PIN RE-EVAL with the fixed prior (5/side): n12 1/10 full, n40 0/10, n160 0/10 —
NO lift over the old prior (2/3/1, within noise). The command fix unlocks the FULL-DATA
flow only; low-data flows are execution-limited in the goal phase (consistent with the
oracle cells: n160+oracle 2/10, n12+oracle 0/10 full). Goal-phase EXECUTION is the
data-hungry component; transit steering is nearly free at any size.

## 2026-08-08 — Zero-pin control launched (Denis's mechanism question)
Question: is the pin's low-data advantage just "denoising from a consistent/familiar source"
(easier training) rather than the coarse answer being supplied? These separate: ZERO-PIN
trains with noise = (I-UU^T)g, c forced to 0 (SNMVP_PIN_ZERO=1 in pi0.py) — source exactly
as consistent as the real pin, but U^T v_target = -U^T a: the coarse action is back in the
regression target and there is no command channel. Matched serving (serve_gate_zeropin.py,
c=0 noise). Cell: gate_n12_zeropin (5k, n12 subset) vs n12 pinstrat (10/10 transit) vs n12
scratchstrat (6/10). If zeropin ~ pin: familiarity story dominates. If ~ scratch: the
supplied answer is what matters. Note the mechanism analysis already on record: coarse U
structure matters for the PREDICTOR (RRR = predictability), not the flow (learned-U chose
high-freq detail); command-following is structural (loss geometry), not learned.

## 2026-08-08 — Standing instrument: phase-resolved command error (Denis)
Tail-weighting is a tool, not a default: apply where it measurably helps, and every command
head's offline report must include PHASE-RESOLVED c-error (early / transit / tail splits, as
in goalphase_diag.py) so tail quality is checked rather than assumed. Note for VLM heads:
their error is large in ALL phases (pose floor ~0.2 m dominates), so phase resolution will
mostly matter once a head is good enough for the tail to be the residual problem — the
instrument is what tells us which regime we are in.

## 2026-08-08 — Grounded selector closed-loop (task #6): two serving bugs found and fixed
v1: server pooled PRE-fusion prefix features; language washes out pre-fusion (2026-08-03
finding, rediscovered live) -> selector output a constant task for all 361 calls. Right
scene incidentally ran grounded-correct: 4/5 FULL with the selector in the loop. Fix:
gate_ctx_common.ctx_pool (post-fusion).
v2: per-frame re-voting -> task flip-flops mid-flight (probs unstable across frames),
command thrash: right 3/5 full but 0/5 clean, left 0/5. Lesson: TASK SELECTION IS AN
EPISODE-LEVEL DECISION — per-frame classification must be latched. Fix: accumulate probs
over the first 12 calls of an episode (episode boundary = state at spawn), lock argmax.
v3 running. If latch is insufficient, next lever is retraining the selector on the SERVING
checkpoint's features (cache came from gate_both_pin; serving is gate_both_pin_rrr).

## 2026-08-08 — ZERO-PIN CONTROL: familiarity alone buys NOTHING (0/10 transits)
gate_n12_zeropin (5k, n12 subset, noise=(I-UU^T)g, c=0 in training AND serving): 0/10
transits — left runs fly low/short of the gate, right runs barely leave spawn. Comparison
at n12: true pin 10/10 transits, scratch 6/10, ZERO-PIN 0/10. Verdict on Denis's
hypothesis: a consistent/"familiar" source WITHOUT the answer in it does not reproduce any
of the pin's low-data advantage (if anything it underperforms plain N(0,I) — the coarse
target is still there to learn, and the zeroed source may remove useful sampling
diversity). The ladder advantage is the SUPPLIED COARSE ANSWER. Mechanism triad now
complete: (1) the coarse channel steers algebraically (additive control matches transits);
(2) pin TRAINING buys complement coherence (clearance 10/10 vs 4/10); (3) source
consistency alone buys nothing (this control). 5 trials/side.

## 2026-08-08 — Paper outline started (docs/PAPER_OUTLINE.md)
Living outline: thesis, contributions with the evidence each rests on, section plan, figure
list, and an EVIDENCE STATUS table marking tier (lead = 5 trials, claim = >=10 + video) and
what is missing before submission. Section 5 is the mechanism decomposition (additive-edit
control + zero-pin control + oracle ceilings) — the section that separates the factorization
claim from cheaper explanations. Keep numbers in sync with this log; the log is authoritative.

## 2026-08-08 — MECHANISM DECOMPOSITION AT CLAIM TIER (10 trials/side, per 20)
                          transit   full   clean
  n160 pin + demo-oracle    20/20   3/20   18/20
  n160 scratch + ADDITIVE   20/20   4/20    8/20   <- same steering, half the coherence
  n12  zero-pin              0/20   0/20   20/20   <- never reaches the gate (clean = never near it)
  n12  pin (state prior)    20/20   4/20   16/20
  n12  scratch (plain)      12/20   0/20   20/20
All three mechanism conclusions SURVIVE the 5 -> 10 trial upgrade, and two sharpen:
(1) Additive edit reproduces steering exactly (20/20 transits, full 4/20 vs 3/20 — if
anything nominally higher) but clearance halves (8/20 vs 18/20). The coarse channel is
algebraic; pin TRAINING buys trajectory coherence. Now claim tier.
(2) Zero-pin is 0/20 transits vs pin 20/20 and scratch 12/20 — a consistent source without
the answer is not merely unhelpful but worse than plain N(0,I). Its 20/20 "clean" is an
artifact of never approaching the gate: a reminder that clearance is conditional on transit.
(3) n12 pin 20/20 transits vs n12 scratch 12/20, and 4/20 vs 0/20 full — the ladder's
low-data claim holds at claim tier for the n12 rung.
NOTE the 5-trial cells over-read full-success in some cells (n160 oracle 2/10 -> 3/20;
n40 oracle 5/10 was 5 trials only) — report full success at 10+ trials only.

## 2026-08-08 — LADDER AT CLAIM TIER (10 trials/side, per 20) + threshold rung launched
                pin transit/full/clean     scratch transit/full/clean
  n12  (3/task)     20 / 4 / 16                12 / 0 / 20
  n40  (10/task)    20 / 4 / 16                14 / 0 / 10
  n160 (40/task)    20 / 1 / 20                10 / 0 / 16
Shape correction vs the 5-trial read: PIN TRANSITS ARE SATURATED (20/20) AT EVERY RUNG and
scratch transits are flat (10-14/20, no data trend); completions do not improve with data in
either arm across 12-160. So the ladder is a LEVEL separation, not a data-efficiency curve:
supplying c makes route-following essentially data-free (3 demos/task suffice), while
scratch never completes a task at any size in range. Terminal precision is the part that
stays hard.
Open question this creates: the record flow (same recipe, ~100 demos/gate task) reaches
19/20 completions, so completions have a THRESHOLD between 40 and 100 demos/gate task.
Launched the intermediate rung: gate_n240_pinstrat (220 eps; 60 per gate task — the two
center tasks cap at 50, hence 220 not 240), 5k steps, evaluated at 10 trials/side with the
TAIL-WEIGHTED prior, plus a re-evaluation of n160-pin with the same prior so the curve uses
one command source end to end. Correct x-axis for the figure is DEMOS PER GATE TASK
(3, 10, 40, 60, 100), not total episodes.

## 2026-08-09 — ENUMERATION-FREE LANGUAGE PRIOR WORKS: 6/10 full success, no task list
c = MLP([model_state, e64]), e64 = PCA-64 of the post-fusion LANGUAGE-token embedding
(gate_ctx_common.lang_pool) of the live prompt; episode-latched over the first 12 calls.
NO task list, NO classifier, NO string matching anywhere in the command path — the first
grounded command source in this project to complete tasks.
  offline: held c-R2 +0.9393 (train +0.9394 — no overfit; embedding-noise aug at 2x the
    measured within-task embedding std). Phase-resolved: early +0.881, transit +0.966,
    tail +0.924 — EARLY is the weak phase, matching Denis's observation that VLM-based
    predictors have been poor at trajectory starts (the state prior's profile is the
    opposite: good early, bad tail).
  closed loop (5 trials/side, record flow): FULL SUCCESS L 4/5, R 2/5 = 6/10.
    Clearance L 3/5, R 0/5 — it completes tasks but flies dirty (grazes), consistent with
    a command field that is accurate on average but noisier frame-to-frame than the
    one-hot prior's.
Reference: string-matched one-hot + tail-weighted prior = 19/20 on the same flow. So the
grounded source costs ~35 points of completion and most of the clearance margin — a real
gap, but the interface is now the north-star-correct one and improvable by data (paraphrase
rows, phase weighting) rather than by architecture change.
Follow-ups launched: claim-tier extension (10 trials/side) + HELD-OUT paraphrase probe
(langprior_paraphrase.py; gate_b_paraphrase eval set, never used in this line).

## 2026-08-09 — LADDER THRESHOLD LOCATED: completions jump between 40 and 60 demos/gate task
Single command source (tail-weighted prior), 10 trials/side, per 20, pin arms:
  demos/gate task   40 (n160)   60 (n240)   ~100 (record)
  transit             20/20       20/20        20/20
  FULL                 3/20       13/20        19/20
  clean               18/20       12/20        (record 15/20)
So terminal precision is a THRESHOLD phenomenon in this range: 40 -> 60 demos per gate task
takes completions from 15% to 65%, and ~100 saturates near 95%. Transits are 20/20 from the
smallest rung (3 demos/task) upward — the pin makes route-following data-free while the
settle phase needs roughly an order of magnitude more data than the route.
Note the clean/full trade at n240 (12/20 clean vs n160's 18/20): flows that actually attempt
the goal box fly closer to the gate; clearance conditioned on completion is the fair
comparison and should be reported that way in the paper.
n12/n40 cells re-running with the SAME prior so Fig. 4 is single-prior end to end.

## 2026-08-09 — Language prior at claim tier (13/20) + paraphrase probe (metric caveat)
CLOSED LOOP, 10 trials/side, record flow, enumeration-free prompt-embedding prior:
  FULL SUCCESS 13/20 (L 7/10, R 6/10); clearance L 4/10, R 1/10.
  Reference on the same flow with string-matched one-hot + tail-weighted prior: 19/20
  (clearance 15/20). So the grounded source costs ~30 points of completion and most of the
  clearance margin, but it completes the task without any task inventory.
PARAPHRASE PROBE (held-out gate_b_paraphrase set, never seen by this line): my drift metric
(||c_para - c_canon|| / ||c_canon - c_othertask||) reads ~0.6-1.6x and looks catastrophic,
but the NORMALIZER IS BAD: between-task command gaps are small (1.5-3.5) relative to |c|
(~10), so the ratio inflates. The informative columns:
  cosine(c_para, c_canon) 0.94-0.995 for most paraphrases (worst 0.77) — DIRECTION preserved
  c-R2 vs true demo c: canonical 0.94 -> paraphrase 0.74-0.80 (LEFT), 0.65-0.89 (RIGHT)
So paraphrases keep the command's direction and lose precision. Given the measured
precision sensitivity (0.2 m command error kills clearance), the expected closed-loop cost
is real but not a task-identity failure. Report absolute c-units + c-R2 in the paper, not
the ratio.
DECISIVE TEST LAUNCHED: fly the record flow with held-out paraphrase prompts (one distinct
paraphrase per trial, 5/side) and compare completions against the canonical-prompt 13/20.

## 2026-08-09 — Single-prior ladder curve complete (Fig. 4 ready)
Pin arms, tail-weighted prior throughout, 20 trials/rung, x = demos per gate task:
  3 -> transit 20/20, full 3/20      10 -> 20/20, 4/20      40 -> 20/20, 3/20
  60 -> 20/20, full 13/20            ~100 -> 20/20, full 19/20
No-pin arms (plain serving): transits 12/14/10 of 20 at 3/10/40, ZERO completions at every
size. Curve published: https://claude.ai/code/artifact/c7fbc5f8-2452-4c3b-9b06-519e57153ee2

## 2026-08-09 — PARAPHRASE FLIGHT: task-asymmetric generalization (held-out prompts)
Record flow + enumeration-free language prior, one distinct HELD-OUT paraphrase per trial:
  LEFT  5/5 transit, 3/5 FULL   (canonical reference 7/10 = 3.5/5 — no measurable loss)
  RIGHT 1/5 transit, 0/5 FULL   (canonical reference 6/10 = 3/5 — collapse; 4/5 never transit)
So paraphrase robustness is TASK-ASYMMETRIC, not uniformly absent: left-gate paraphrases fly
as well as the trained prompt; right-gate paraphrases miss the gate entirely. Consistent with
the offline probe's asymmetry (RIGHT paraphrase c-R2 fell to 0.65-0.89 with cos down to 0.77,
vs LEFT 0.74-0.80 with cos >0.95) and with the long-standing right-scene fragility (the
+x aim bias, the right-gate label bug, the right goal phase). Interpretation: the right task
sits closer to a command-space decision boundary, so embedding perturbation costs more there.
REMEDY LAUNCHED (GPU1): paraphrase-augmented prior — train rows include 12 TRAIN paraphrases
per task (disjoint from the eval set), one sampled per frame at 1/3 of frames; then fly BOTH
canonical and held-out paraphrase prompts. Also launched (GPU0): gate_n240_scratchstrat, the
no-pin arm at the threshold rung (60 demos/gate task), to complete Fig. 4's comparison where
the pin arm jumps to 13/20.

## 2026-08-09 — Latching is a single-stage scaffold (Denis) + no-latch ablation queued
Denis: "go through the left gate twice then the right gate then the left gate — the
viewpoint matters after 12 frames, so this doesn't generalize." Correct: latching the
language embedding assumes the instruction's MEANING is constant for the episode. For
multi-stage instructions the active sub-goal is a language x image reading that must keep
running; a latch freezes the interpretation at the establishing view and every later stage
is commanded stale. Latching is therefore a scaffold in the same category as the one-hot,
and it also breaks the existing prompt-switch composition experiments.
Second problem: the latch was inherited from the CLASSIFIER's failure mode (argmax is
discontinuous, so feature jitter flipped the whole one-hot mid-flight). The regression prior
has no such discontinuity and is already trained with embedding-noise augmentation, so the
latch may be doing nothing. Never tested — testing now.
LATCH_N env added to serve_gate_pin_langprior (0 = recompute every step). No-latch battery
queued on GPU1 behind the paraphrase-augmented chain; comparison cell = latched canonical
13/20. Design direction if no-latch holds: keep the language read LIVE at every step and buy
stability from head smoothness (embedding-noise augmentation), not from freezing — which is
also what multi-stage instructions require.

## 2026-08-09 — Paraphrase augmentation: offline UP, closed-loop DOWN (negative)
Trained the language prior with 12 TRAIN paraphrases per task mixed in (one sampled per
frame at 1/3 of frames). Offline every metric improved: held c-R2 0.9393 -> 0.9623; phases
early 0.881 -> 0.928, transit 0.966 -> 0.982, tail 0.924 -> 0.940.
Closed loop it got WORSE, and the right task collapsed (5 trials/side):
  canonical prompts:  L 5/5, R 0/5  (un-augmented: L 7/10, R 6/10 = 13/20)
  held-out paraphrase: L 2/5, R 0/5  (un-augmented: L 3/5, R 0/5)
So paraphrase rows bought aggregate fit and cost command precision exactly where it was
already marginal. Reading: spreading the language cluster over 12 surface forms per task
widens the embedding region the head must average over; the right task, which sits nearer a
command-space boundary, loses its margin entirely. This is the offline/closed-loop chasm
again (5th independent instance) — R2 improvements do not imply closed-loop gains, and this
time the offline gain was uniform across phases and still inverted in flight.
NOT a fix. Options that remain for paraphrase robustness: (a) more DISTINCT TASKS rather
than more surface forms of four tasks (the language subspace is 3-dimensional with 4
prompts — see the PCA measurement — so paraphrases add no new dimensions, only spread);
(b) contrastive training that pulls paraphrases of one task together while pushing tasks
apart, instead of plain regression over a widened cluster.

## 2026-08-09 — Deployable packaging + steering verified through the flow
src/snmvp/deploy.py (PinnedPolicy, NumpyMLP, OneHotTasks, LanguageEncoder) +
scripts/package_policy.py: one immutable bundle (params/, norm_stats/, pin_U.npy,
prior.npz, manifest.json with per-artifact sha256, config, action space, git commit). No
sim, no renderer, no experiment imports; no torch at inference (prior converted to numpy).
Packaging refuses to write unless numpy/torch parity passes and the bundle infers.
Three real bugs the guards caught while building it: naive SiLU overflow (fixed with a
branch-stable sigmoid); a parity threshold tighter than float32 can deliver (now judged on
relative error, 7.5e-07); and a checkpoint path off by one level.
STEERING VERIFIED END TO END (open loop, through the flow, not just in the pinned subspace):
nudge z +0.30 m -> generated chunk net dz +0.001 -> +0.291 m (97%), lateral axes undisturbed
(+0.002, -0.003 m). Command displacement is now reported per inference
(res["snmvp_command_displacement"]) — the interpretable trace of what the policy intends.
First bundle: /home/ubuntu/bundles/gate_record_v1 (record flow + tail-weighted prior, the
19/20 system). Docs: docs/DEPLOYMENT.md.

## 2026-08-09 — LIBERO low-data ladder launched (cross-domain replication)
LIBERO sim confirmed working on this box (see setup note). Ladder built and queued:
  data: scripts/build_libero_subsets.py -> local/libero_n{2,5} (40 tasks, 2 or 5 demos per
        task = 80 / 200 episodes; LeRobot v2.0 layout, nested, norm stats reused)
  configs: pi0_libero_n{2,5} (clone of pi0_libero_shared — NOT low_mem_finetune, whose
        extra_delta_transform changes the action space)
  arms: pin (SNMVP_PIN_U=pin_U_rrr_k5_shared.npy, the RRR basis fit on LIBERO) vs scratch,
        5k steps each, both GPUs, queued behind the running drone jobs.
Why these rungs: the drone result says the pin's advantage is at LOW data and that structure
is useless-to-harmful once data solves the task (prior LIBERO finding: -40pt on full data).
2 and 5 demos/task sit well below LIBERO's ~42/task, i.e. inside the regime where the claim
applies. Eval will use the LIBERO sim through scripts/libero_eval_client.py with a pinned
server; command source for the pin arm still to be built (state prior on LIBERO demos).

## 2026-08-09 — Live steering GUI (in sim) + client-side nudge protocol
Denis: do the steering demo live in sim rather than as a pre-rendered page. Built:
- serve_gate_pin_prog4.py now accepts obs["snmvp_nudge"] = [dx,dy,dz] METRES per call, so
  steering is client-driven and stateless (env NUDGE still works for batch sweeps).
- experiments/rung3/steer_gui_server.py: stdlib HTTP server (no new deps) running in the tv
  env; a background thread flies the scene continuously through the pin server and publishes
  an MJPEG stream; the browser UI has x/y/z nudge sliders, zero/restart, and a live readout
  of position and per-chunk displacement. Reachable over an SSH tunnel.
- gate_video_overlay.py made importable scene-only (NCH=0: no client, no rollout, no video)
  so the GUI reuses the exact renderer, cameras and observation construction the batteries
  use — demo and measurements share one code path.
- run_steer_gui.sh brings up both servers and prints the tunnel command.
Two self-inflicted bugs fixed on the way: an inline comment appended to a semicolon-chained
line commented out five assignments; and the gsplat import needs PATH=/tmp/tv/bin +
CUDA_HOME (ninja for the CUDA extension) — the battery scripts set this, a bare python does not.

## 2026-08-09 — CORRECTION: the no-pin arm DOES complete tasks at 60 demos/gate task
gate_n240_scratchstrat (no pin, plain serving), 10 trials/side: transit 20/20, FULL 9/20
(L 9/10, R 0/10), clean 18/20. This CONTRADICTS the claim "the no-pin arm completes zero
tasks at any size", which I had generalised from the 3/10/40-demo rungs where it was true.
Corrected single-prior ladder (full success per 20; pin | no-pin):
  3 demos/task   3/20 | 0/20        transits 20/20 | 12/20
  10             4/20 | 0/20                 20/20 | 14/20
  40             3/20 | 0/20                 20/20 | 10/20
  60            13/20 | 9/20                 20/20 | 20/20
  ~100          19/20 | (not run)            20/20 | -
Revised reading: the pin's advantage is LARGE below ~40 demos per gate task (transits
20/20 vs 10-14/20; completions 3-4/20 vs 0/20) and LARGELY CLOSES by 60 (13/20 vs 9/20,
p~0.2 — not significant at n=20; transits tie at 20/20). This is the same conditional the
LIBERO line already found ("structure helps only when the policy is the bottleneck; it is
useless-to-harmful when data solves the task") — now measured as a crossing point on one
curve rather than as two separate regimes. It is a better result for the paper than the
over-general version: the claim becomes "the factorization buys competence in the
data-starved regime and its advantage decays as data grows", with the crossing located.
Right-scene asymmetry persists in the no-pin arm too (L 9/10 vs R 0/10).
Artifact + paper outline corrected accordingly.

## 2026-08-09 — No-latch ablation: latching buys nothing measurable; drop it
Same prior, same flow, language embedding recomputed EVERY step (LATCH_N=0), 10 trials/side:
  live:    FULL 10/20 (L 7/10, R 3/10), clearance 7/20
  latched: FULL 13/20 (L 7/10, R 6/10), clearance 5/20
10/20 vs 13/20 is not significant at n=20 (p~0.34); left is identical, right differs by 3
trials (p~0.18). So the latch has NO established benefit for this regression head — it was
imported from the classifier's argmax thrash, a discontinuity this head does not have.
DECISION: default to live embeddings (LATCH_N=0 in the deployable LanguageEncoder). It is
the generalizable choice — latching assumes single-stage instructions and breaks both
multi-stage commands and the prompt-switch composition experiments — and it costs nothing
demonstrable. If the right-task hint proves real at higher n, the fix should be command
smoothing (EMA over c, or head smoothness), not freezing the language input.

## 2026-08-09 — Steering command-response: AMPLIFIED, not attenuated (prediction wrong)
Sweep on the record system (3 trials/setting, left scene), mean trajectory offset vs baseline:
  z -0.30 -> -0.244 (0.81x)   z -0.15 -> -0.151 (1.01x)
  z +0.15 -> +0.247 (1.65x)   z +0.30 -> +0.565 (1.88x)
  y -0.30 -> -0.364 (1.21x)   y +0.30 -> +0.658 (2.19x)
I predicted realized < commanded because the state prior opposes displacement (restoring
gain 0.39-0.73). WRONG: gains are 0.8-2.2x. Mechanism: the nudge is a PER-CHUNK displacement
bias applied on every command, so it accumulates across ~40 chunks; accumulation beats the
restoring pull at these magnitudes. Asymmetries: climbing gains more than descending
(1.88 vs 0.81 — descent is opposed by the route/goal structure), and the lateral axis gains
most (2.19).
Consequence for the interface: the nudge is a well-behaved monotonic control but NOT a
metre-for-metre trajectory offset; calibrate per axis if a specific offset is wanted, or
apply the nudge for a bounded number of chunks. Open loop (single chunk) it remains 97%
faithful, which is the honest "exactness" claim.
Page (slider + baseline comparison + path overlays): see steering artifact.

## 2026-08-09 — LIBERO command prior trained: 0.85 held c-R2 (vs 0.97 on the drone)
c = MLP([model_state, task_onehot40]), tail-weighted (TAILW=4 from the start), trained once
on all 1693 episodes and reused at every rung. Held c-R2 +0.8535 (train +0.8822); phases
early +0.867, transit +0.813, TAIL +0.709.
PRE-FLIGHT READ, recorded before running the closed loop: the drone's prior sits at 0.97 and
that bought 19/20; LIBERO's command prediction is materially worse and its tail is worst
(0.71) even with tail weighting. Since command error passes through the pin 1:1, this
predicts weak closed-loop numbers for the LIBERO pin arm — and a null ladder result would
then be UNINTERPRETABLE (bad commands vs no benefit from the pin).
So the LIBERO evaluation ships three arms, mirroring the drone component battery:
  pin + prior   (the ladder cell)
  pin + DEMO-ORACLE c  (k-NN over demo state->c of the same task; the execution ceiling)
  scratch, plain serving (the control)
serve_libero_pin.py implements all three behind --source {prior,oracle,plain}.

## 2026-08-09 — LIBERO: commands are NOT the bottleneck (oracle ~= prior), and why
libero_spatial, 10 tasks x 5 trials, flows trained on 2 demos/task:
  pin + prior       0.16
  pin + DEMO-ORACLE 0.18   <- near-perfect commands buy ~2 points
This inverts the drone finding (where oracle commands took the record flow from 0 to 3/5 on
the right scene and exposed the prior as the bottleneck). On LIBERO at 2 demos/task the
EXECUTION is the binding constraint, so supplying the coarse action cannot help much.
Mechanistic check on the LIBERO RRR basis (pin_U_rrr_k5_shared, same measurement as the
drone's): the pin captures only 41.9% of chunk variance overall, and per action dim
  x 40.0  y 56.6  z 59.5  roll 28.9  pitch 25.0  yaw 45.7  GRIPPER 43.4  (% of that dim)
versus the drone basis, which expresses per-axis translation nudges at 0.985-0.994.
Two structural reasons the drone result does not transfer as-is:
(1) LIBERO manipulation is gated on CONTACT EVENTS (grasp/release timing), which are
    discrete and not expressible as coarse chunk displacement — the pin's currency;
(2) the basis carries under half the chunk variance here, consistent with the 2026-07
    finding that LIBERO actions are contact-rich/bang-bang rather than smooth low-rank.
Honest cross-domain statement so far: the factorization's benefit depends on whether the
task's difficulty lives in the COARSE, low-rank part of the action. Drone navigation: yes
(the route is the task). LIBERO manipulation: largely no (the decisive part is contact
timing). Awaiting the scratch arms to complete the comparison.

## 2026-08-09 — LIBERO LADDER (hard pin): the pin HURTS — 0.16/0.18 vs 0.30 for no-pin
libero_spatial, 10 tasks x 5 trials, 2 demos/task:
  pin + prior 0.16 | pin + demo-oracle 0.18 | SCRATCH (no pin) 0.30
The no-pin arm is ~2x the pin arm, and the pin arm loses even with near-perfect commands. So
on LIBERO at low data the pin is not merely unhelpful, it is HARMFUL — the opposite of the
drone ladder.
CONFIGURATION ERROR (mine): this pass used a HARD pin, a straight port of the drone recipe,
ignoring our own 2026-07 LIBERO finding — "a pin-trained VLA is hostage to the prior... soft
pin (sigma=0.7) fixes the hostage-to-prior catastrophe, LIBERO object 0.00 -> 0.87, vs hard
pin 0.40/0.00". The hard pin was the right choice on the drone (commands 0.97 accurate) and
is the known-wrong choice on LIBERO (commands 0.85, contact-gated tasks). The measured harm
is exactly the predicted failure mode: pass-through faithfully executes imperfect commands
with no capacity to correct.
Relaunched: libero_n2_pin_soft07 and libero_n2_pin_soft035 (sigma 0.7 / 0.35, same data and
steps). The fair cross-domain test is soft-pin vs scratch; the hard-pin numbers stand as a
measurement of what pass-through costs when commands are imperfect and the basis captures
only 42% of chunk variance.

## 2026-08-09 — RETRACTION: my LIBERO "oracle" is not a ceiling — phase aliasing
n5 pin + demo-oracle scored 0.10, BELOW n5 pin + learned prior (0.30). An execution ceiling
cannot be worse than a learned predictor; the instrument is broken, not the flow.
Cause: the oracle retrieves c by k-NN over the raw 8-d LIBERO state. In manipulation the
same arm pose recurs at different task stages (descending to grasp vs lifting after grasp),
so nearest-state retrieval returns WRONG-PHASE commands. On the drone this did not bite
because position along a flight path nearly determines phase. The n2 numbers (oracle 0.18 vs
prior 0.16) are equally unreliable.
THEREFORE RETRACTED: the claim "on LIBERO commands are not the bottleneck; execution is"
rests on this instrument and is not supported. What still stands (measured independently of
the oracle): pin 0.16 vs scratch 0.30 at 2 demos/task (hard pin), and the basis capturing
only 41.9% of LIBERO chunk variance.
Fix for the instrument: retrieve on (state, progress) or restrict k-NN to a monotone
progress window / DTW-align to a demo, so retrieval cannot jump task stage. Queued behind
the soft-pin arms, which are the more important experiment.

## 2026-08-09 — LIBERO hard-pin ladder COMPLETE: no benefit; harm at the smallest rung
libero_spatial, 10 tasks x 5 trials (50 episodes/arm):
  demos/task   pin+prior   scratch (no pin)
      2          0.16         0.30
      5          0.30         0.28
(oracle cells 0.18 / 0.10 are discarded — the retrieval instrument phase-aliases, see the
retraction entry.) At 2 demos/task the pin loses by 14 points (8/50 vs 15/50, p~0.10); at 5
they tie. Note the shapes differ: the pin arm nearly doubles with 2.5x the data (0.16->0.30)
while scratch is flat (0.30->0.28) — the pin arm is data-limited in a way the plain
fine-tune is not, which is what a constraint that must be learned to satisfy looks like.
Contrast the drone at comparable rungs: pin 20/20 transits vs no-pin 12/20, and pin the only
arm completing tasks. The cross-domain claim therefore does NOT replicate as a general
low-data benefit. Best current statement of scope, pending the soft-pin arms:
  the factorization pays when the task's difficulty lives in the coarse, low-rank part of
  the action (drone navigation: route IS the task; basis expresses per-axis motion at
  98.5-99.4%), and does not when the decisive part is contact timing (LIBERO: basis captures
  41.9% of chunk variance; success is gated on grasp/release events).
Soft-pin arms (sigma 0.7 / 0.35) training — the configuration our 2026-07 LIBERO work found
necessary; the hard-pin numbers above stand as the cost of pass-through under imperfect
commands in a domain the basis fits poorly.

## 2026-08-09 — LIBERO basis probes (Denis's two questions)
GRIPPER: only 15.0% of the shared basis's squared norm sits on the gripper dimension (85% on
the six arm dims), and removing it costs little capture (42.7% -> 38.2% pooled). So the
gripper is NOT consuming the pin — but capture is not harm: a small basis component on a
binary, decisively-timed dimension can still wreck grasping by commanding the wrong gripper
trajectory with pass-through fidelity. Built pin_U_rrr_k5_nogrip.npy (gripper rows zeroed,
re-orthonormalised) and launched the closed-loop arm; that is the actual test.
SUITES (LIBERO is single-embodiment — Franka Panda — so the analogue of "per embodiment" is
its four task suites): suite-specific PCA-5 bases capture far more than the shared basis:
  long   37.2 -> 50.9    goal 45.1 -> 59.0    object 49.8 -> 69.7    spatial 38.3 -> 56.9  (%)
and the suites' subspaces genuinely differ — principal angles between suite bases run
[14-24, 20-41, 28-53, 36-76, 71-88] deg, i.e. ~2 shared directions and 3 divergent ones. The
shared basis is a poor compromise: it is 13-20 points below each suite's own basis.
Caveat carried from the drone: better CAPTURE did not mean better commands there
(per-domain U was negative because RRR selects PREDICTABLE directions, not high-variance
ones). Launched the closed-loop arm with the spatial-suite basis on the same n2 data, so the
comparison isolates basis choice: pin(U_shared) 0.16 vs pin(U_spatial) vs scratch 0.30.

## 2026-08-09 — K was under-provisioned on LIBERO (Denis): 5 pin dims for 7 active axes
Active action dims (std > 0): DRONE 4 (x,y,z,yaw — pitch/roll/gripper constant) with K=5, so
the pin had MORE directions than active axes; measured per-axis expressivity 0.985-0.994.
LIBERO 7 active dims with K=5 — under-provisioned by two, and measured expressivity is
0.48-0.88 per axis. The drone's K was quietly generous; LIBERO's was quietly starved.
Candidate bases measured on LIBERO chunks (capture / per-axis expressivity):
  RRR K=5 (used)     42.8%  /  0.48-0.88
  displacement K=7   41.0%  /  1.00 exact by construction
  PCA K=7            60.2%  /  0.44-0.93
The displacement basis captures no more variance but captures the RIGHT variance — exact
per-axis authority, which is the property the drone basis had and this one lacked.
Queued: libero_n2_pin_disp6 (one direction per ARM axis, gripper excluded — Denis's
proposal, which also answers the gripper question in its cleanest form) and
libero_n2_pin_disp7 (all seven axes). Priors for both bases building on CPU now
(libero_prior.py is parameterised by UPATH/OUT, K inferred from the basis).

## 2026-08-10 — Drone K=8 (temporal shape) trained; prior slightly worse than K=5
pin_U_half8_gate.npy = early/late chunk displacement per active axis (4 axes x 2 halves).
Capture on gate-task chunks 86.6% vs the deployed RRR K=5's 78.5% (PCA K=8 bound 95.6%).
Prior on the K=8 basis: held c-R2 0.9601 vs 0.9705 for K=5 — slightly HARDER to predict, as
expected: the extra directions encode WHEN motion happens within the chunk, which is less
determined by the current state than net displacement is. So K=8 trades a little command
accuracy for more expressive commands; the battery decides whether that trade pays.
Queued the battery (10 trials/side) behind the center/compound run on GPU1.

## 2026-08-10 — Center-gate and compound on the CURRENT best drone system (19/20 config)
gate_both_pin_rrr/4999 + tail-weighted prior, 10 trials each, strict judge + clearance:
  center-from-LEFT   10/10 FULL success, 9/10 clearance-clean
  center-from-RIGHT   9/10 FULL success, 6/10 clearance-clean
  COMPOUND (left gate -> center gate, prompt switched at the crossing)
                      0/10 — every run passes gate 1 (steps 70-86) and NEVER reaches gate 2
So the four single-gate tasks are now all measured on the current system:
  L 10/10, R 9/10, CFL 10/10, CFR 9/10 = 38/40 FULL success (clearance 9/10, 6/10, 9/10, 6/10
  — right-side and center-from-right grazes remain the quality gap). The tail weighting fitted
  on left/right transfers to the center tasks rather than regressing them, which was the risk.
COMPOUND IS THE STANDING FAILURE: 10/10 pass the first gate and 0/10 reach the second. The
run does not stall randomly — it completes stage 1 and then does not proceed, i.e. the
prompt switch does not redirect the flight. This is the multi-stage problem in its cleanest
form and it is NOT a latch artifact (this prior takes a one-hot that changes with the
prompt). Candidate causes to separate next: (a) the command after the switch is correct but
the state prior's geometry maps the post-gate-1 state to a goal-phase command (it was tail-
weighted to settle, and after gate 1 the drone is deep in "tail" states); (b) the switch
arrives at a state far outside the CFL demos' start distribution, so c is extrapolated.
(a) is testable offline in minutes: compare prior c after the switch against demo c at
matched states from the CFL demos.

## 2026-08-10 — RETRACTION + REPLICATION: soft pin sigma=0.7 BEATS no-pin on LIBERO (0.48 vs 0.30)
libero_spatial, 10 tasks x 5 trials (50 episodes/arm), all flows trained on 2 demos/task:
  soft pin sigma=0.7      0.48   <- best
  no-pin scratch          0.30
  soft pin sigma=0.35     0.24
  suite-spatial basis     0.22  (hard pin)
  hard pin K=5            0.16
  gripper-free K=5        0.14  (hard pin)
  displacement K=6        0.18  (hard pin)
  displacement K=7        0.10  (hard pin)
RETRACTED: "the LIBERO ladder shows no benefit / harm, therefore the factorization only pays
in coarse-dominated domains." That conclusion came from the hard-pin pass and was a
CONFIGURATION artifact — exactly the error flagged when the hard pin was found in the recipe.
With the pin configured as our own 2026-07 LIBERO work prescribed, the low-data benefit
REPLICATES cross-domain: +18 points over a matched no-pin fine-tune (24/50 vs 15/50,
p~0.08 — suggestive, needs claim-tier n).
Mechanism reading, now consistent across both domains: sigma must scale with COMMAND ERROR.
  drone   prior c-R2 0.97 -> hard pin best, sigma=0.35 near-equal, sigma high unnecessary
  LIBERO  prior c-R2 0.85 -> sigma=0.7 best (0.48), sigma=0.35 worse (0.24), hard pin worst (0.16)
The pin is pass-through, so with accurate commands hardness is free and with inaccurate
commands the flow MUST retain capacity to correct. That is a single predictive rule rather
than two domain-specific recipes.
IMPORTANT CAVEAT on Denis's basis questions: gripper-free, suite-specific and displacement
K=6/K=7 were ALL trained with HARD pins, so they inherit the hard-pin handicap and their
numbers do NOT answer the K or gripper questions. The fair test is soft pin x basis; queued
soft07 x displacement-K=7 next.

## 2026-08-10 — Reference numbers + the MISSING full-data control (Denis)
Published LIBERO, full data: pi0 paper Spatial 96.8 / Object 98.8 / Goal 95.8 / Long 85.2,
avg 94.2. openpi docs for pi0.5 @30k: Spatial 98.8, avg 96.85. Note a known reproduction
dispute (huggingface/lerobot issue #2114: fine-tuned pi0/pi0.5 reportedly fail to replicate
these rates), so treat them as a soft reference rather than a hard target.
Denis: "we should also be training with the same amount of data, why weren't we?" — the
ladder is deliberately starved (that is the experiment), but he is right that we ran it with
NO POSITIVE CONTROL. Without a full-data cell in our own pipeline we cannot distinguish
"0.30 no-pin at 2 demos/task is a credible low-data point" from "our LIBERO pipeline is
misconfigured". Every LIBERO number today rests on that unverified assumption.
Queued (GPU0, after the soft-pin follow-ups): pi0_libero_shared on all 1693 episodes at the
SAME 5k steps, both scratch and soft07, evaluated at 100 episodes on spatial.
  - scratch@full tells us whether the pipeline reaches sensible numbers with data, isolating
    data quantity at matched optimisation;
  - soft07@full tests whether the pin still helps once data is plentiful — our 2026-07 work
    says it should NOT (-40pt at full data), so this is also a check on the sigma rule.
A 30k-step run would be needed for a like-for-like comparison with the published numbers;
5k is a floor on our own ceiling, not a replication attempt.

## 2026-08-10 — COMPOSITION (enumeration-free prior, repo prompts, no switch): 0/10
Ran the two eval-only compositional prompts from falsify-pi/configs/prompts (no training
counterpart), one instruction for the whole episode, language prior with LIVE embeddings, no
prompt switch and no ground-truth milestone:
  left_then_center   0/5 — all five pass gate 1 (steps 78-87), none reaches gate 2
  right_then_center  0/5 — gates=0/2, they do not even clear the FIRST gate
So the conjoined instruction reproduces stage 1 on the left scene and then stalls exactly as
the one-hot + scripted-switch version did; on the right compound scene it fails outright,
consistent with this prior's known right-side weakness (held-out paraphrases: right 1/5
transit). Composition does not work, and the failure is not caused by the one-hot: it
survives replacing the command source with the grounded one.
What this isolates: the second stage begins from states no demo covers, and the command
source has no phase variable, so at the same position it cannot distinguish "before gate 1"
from "after gate 1". Both the one-hot and the embedding prior are pure functions of
(state, instruction) — neither can express "which part of the instruction am I on".

## 2026-08-10 — DRONE K=8 (temporal shape): clearance PERFECT, completions DOWN
gate_pin_half8 + K=8 prior (early/late displacement per axis), 10 trials/side:
                    transit   full    clean
  K=8 (this run)     20/20    10/20   20/20   (L 2/10 full, R 8/10 full)
  K=5 record         20/20    19/20   15/20   (L 10/10, R 9/10)
Every K=8 run transits, and clearance is 20/20 with the closest approach 0.279 m — inside the
demo band (0.28-0.38) and a clean fix for THE remaining quality gap (K=5 grazes: 5/20 dirty,
several at 0.10-0.18 m). But completions fall 19/20 -> 10/20, and the loss is almost entirely
on the LEFT scene (10/10 -> 2/10) where every failure is transit-then-no-goal-dwell.
Reading: the extra temporal dimensions buy flight quality (the command can say WHEN to move,
so it stops cutting corners) and cost goal-phase reliability, consistent with the K=8 prior
being harder to predict (held c-R2 0.9601 vs 0.9705) precisely in the tail where timing
matters most. The natural follow-up is K=8 with a tail-focused prior fix, or K=8 for the
transit phase and K=5 semantics near the goal.

## 2026-08-10 — WHY K=8 fails: goal OVERSHOOT with residual motion (measured, not inferred)
Goal box centre (left_gate.yaml): [1.525, -0.615, 1.0]. Final positions, 10 trials:
  K=5  (1.57, -0.62, 0.87) +/- (0.11, 0.07, 0.03)   motion over last 40 steps 0.31 m
  K=8  (2.36, -1.46, 0.63) +/- (0.45, 0.63, 0.14)   motion over last 40 steps 0.60 m
K=8 does not fail to arrive — it flies PAST the goal (~0.8 m over in both x and y), ends
lower, is still moving at episode end, and its spread is 4-6x wider.
MECHANISM: under K=5 "hover here" is one number per axis (net displacement ~ 0). Under K=8
stopping is a TWO-PART command — decelerate in the first half of the chunk, hold in the
second — so the prior must get both halves right, and any error in the late-half directions
is residual motion executed verbatim, because the pin is pass-through and K=8 pins 8
directions instead of 5. More command authority => more exposure to command error, and it
bites hardest exactly where precision matters (the goal box).
PREDICTION (same rule the LIBERO sigma sweep established): K=8 needs sigma > 0 so the flow can
correct late-half command error. Queued gate_pin_half8_soft035 (sigma=0.35, same data/steps)
+ battery. If it recovers completions while keeping K=8's perfect clearance, that is the best
drone system so far (K=5 hard: 19/20 full, 15/20 clean; K=8 hard: 10/20 full, 20/20 clean).
Also queued: centre-gate cells for K=5 vs K=8, both served with their enumeration-free
language priors so the comparison uses current components rather than the one-hot scaffold.

## 2026-08-10 — CLAIM TIER walks back the LIBERO replication: +8 points, NOT significant
100 episodes/arm (10 trials x 10 spatial tasks), 2 demos/task:
  soft pin sigma=0.7   0.49
  no-pin scratch       0.41
  soft07 x disp K=7    0.42  (n=50)
At n=50 this looked like 0.48 vs 0.30 (+18, p~0.07). At n=100 the SCRATCH arm rose to 0.41
and the gap fell to +8 points (49/100 vs 41/100, z~1.14, p~0.25) — NOT significant. The
n=50 scratch estimate (0.30) was simply low; the pin estimate barely moved (0.48 -> 0.49).
REVISED STATUS: the cross-domain benefit on LIBERO is NOT established. Point estimate favours
the soft pin, the effect is within noise at n=100, and honest reporting is "no significant
difference; if anything a small advantage". Yesterday's "+18 points, replicates" was an
n=50 artifact, the same trap the drone ladder hit when 5-trial cells over-read completions.
What DOES survive from the sigma sweep, because those cells differ by much more than noise:
  hard pin 0.16 vs soft07 0.49 (16/100-equivalent vs 49/100) — pin HARDNESS matters a lot on
  LIBERO, and the sigma-scales-with-command-error rule stands on that contrast, not on the
  pin-vs-scratch gap.
K QUESTION under correct hardness: soft07 x displacement-K=7 = 0.42 vs soft07 x RRR-K=5 =
0.49 — raising K to match the 7 active action dims does NOT help on LIBERO once the pin is
soft. So the earlier K=6/K=7 deficits were mostly the hard pin, and matching K to DOF is not
the fix here either.

## 2026-08-10 — COMPOSITION with K=8 language prior: 0/10, and worse than K=5
Repo compositional prompts, single instruction, live embeddings, no switch:
  K=8 flow + K=8 language prior:  left_then_center 0/5 (gates=0/2 — never clears gate 1)
                                  right_then_center 0/5 (gates=0/2)
  K=5 flow + K=5 language prior:  left_then_center 0/5 but cleared gate 1 in ALL five
                                  right_then_center 0/5 (gates=0/2)
So more coarse dimensions did not help composition and cost stage 1 on the left. Consistent
with the K=8 single-gate finding (overshoot, residual motion): in the compound scene the
first gate arrives at a different point along the route, and an overshooting flight misses it.
CONTROL STILL NEEDED and now queued (k8hard_single): the same K=8 flow + K=8 language prior
on SINGLE-gate tasks. Without it, "never clears gate 1" cannot be separated from "this
flow+prior pair cannot fly one gate either" — the K=8 numbers we have used the ONE-HOT prior.
PROCESS ERRORS this run exposed, both mine:
 1. compose_lang_k8 wrote its done-marker BEFORE an appended baseline block, so the next
    chain (k8soft) began training while the baseline was still on the GPU — GPU1 hit 94.9/97.9
    GB. Rule: the done-marker goes at the true end of the script.
 2. A pkill pattern matched my own shell because the HEREDOC BODY being written contained the
    string the pattern matched. Rule: build scripts with the file tool, not heredocs, when the
    body mentions process names; and never mix a pkill with script authoring in one command.

## 2026-08-10 — U's TWO PURPOSES measured separately (Denis's framing)
Denis: "U serves two purposes — a high-variance summary of the chunk, and something simpler
to predict from our inputs. We're doing well on the former, not great on the latter."
Measured on identical rows (9000 rows, 200 gate-set episodes, frozen 160/40 episode split);
predictability = held-out R2 of an MLP from the DEPLOYED inputs to c:
  basis                     capture   R2 from state   R2 from state+language
  deployed RRR (VLM feats)   82.8%        0.905            0.964
  state_rrr (fit on state)   69.7%        0.927            0.976
  lang_rrr (state+lang)      82.3%        0.918            0.969
  PCA-5 (max capture)        82.9%        0.909            0.962
  half-split K=8             87.0%        0.889            0.954
Findings:
1. The two purposes genuinely TRADE. Ranking by capture (K=8 87.0 > PCA 82.9 > RRR 82.8 >
   state_rrr 69.7) is almost exactly the REVERSE of the predictability ranking (state_rrr
   0.927 > lang_rrr 0.918 > PCA 0.909 > RRR 0.905 > K=8 0.889). Denis's read is right: we have
   been optimising the summary property and taking whatever predictability followed.
2. A REAL MISMATCH in the deployed basis: RRR was fit as the top-K eigenvectors of the chunk
   variance predictable from VLM PREFIX FEATURES, but the command source that flies is a
   function of (state, instruction). Refitting the same construction against the deployed
   inputs (state_rrr) raises predictability 0.905 -> 0.927 from state and 0.964 -> 0.976 from
   state+language, at the cost of 13 points of capture.
3. lang_rrr is the interesting middle: capture 82.3% (equal to the deployed basis) with
   predictability above it on both input sets — the deployed basis is DOMINATED by a basis fit
   against the inputs we actually use, at no capture cost.
Saved pin_U_state_rrr_gate.npy and pin_U_lang_rrr_gate.npy. Whether predictability beats
capture in CLOSED LOOP is the open question: pass-through says predictability should matter
more, the K=8 result says capture buys flight quality (clearance 20/20). Queuing lang_rrr
closed loop — it dominates the deployed basis offline, so it is the cheapest available win.

## 2026-08-10 — FULL-DATA LIBERO CONTROL: the pipeline is sane, but every cell is UNDERTRAINED
libero_spatial, 100 episodes/arm, pi0_libero_shared, 5k steps everywhere:
                        no-pin    soft pin sigma=0.7
  2 demos/task (80 eps)   0.41          0.49
  FULL data (1693 eps)    0.53          0.62
Significance: pin effect at 2 demos p=0.256; at full data p=0.198; POOLED across both
regimes 111/200 vs 94/200, +8.5 points, p=0.089. Data effect on the no-pin arm
(0.41 -> 0.53) p=0.089.
Two conclusions, one reassuring and one that reframes the whole LIBERO line:
1. THE PIPELINE IS SANE. More data moves the no-pin arm in the right direction by a sensible
   amount, so the low-data cells are points on a curve rather than symptoms of misconfiguration.
   That was the control we lacked, and it passes.
2. BUT 0.53 AT FULL DATA vs the published 96.8 (pi0, 30k steps) means our 5k-step budget —
   not data — is the binding constraint. So our "full data" cell is data-rich and
   OPTIMISATION-POOR: it never reaches the regime where data solves the task. Every LIBERO
   number we have is therefore from a bottlenecked policy, INCLUDING the full-data cells.
   Consequence: this does NOT contradict the 2026-07 finding that structure hurts by 40 points
   once data solves LIBERO — we never got there. And it means the persistent +8/+9 pin
   advantage across both data regimes is evidence about the bottlenecked regime only, which is
   exactly what "structure helps when the policy is the bottleneck" predicts.
To actually test the crossover we need 30k steps per arm (~17 h each). Worth doing once, on
the full-data pair only, before any LIBERO claim goes in the paper.

## 2026-08-10 — WHY WE ARE OPTIMISATION-POOR: the schedule was never matched to the run
Diagnosed from the config rather than assumed. Every run in this project (drone AND LIBERO)
stopped at 5k steps against a schedule built for 30k:
  warmup_steps = 1000            -> 20% of a 5k run is warmup
  decay_steps  = 30000 (FIXED, independent of --num-train-steps)
  LR at step 5000 = 2.396e-5 vs peak 2.5e-5  -> we stop at 96% OF PEAK, never annealing
  batch_size   = 32 (TrainConfig default; openpi's other recipes use 256)
  samples seen = 5k x 32 = 160k vs published 30k x 256 = 7.7M  -> 48x fewer
  LoRA adapters only (paligemma gemma_2b_lora + action expert gemma_300m_lora, trunk frozen),
  ema_decay = None, and extra_delta_transform=False in the shared configs (our deliberate
  cross-domain choice, but it also differs from the published LIBERO recipe's action space).
So "0.53 at full data vs published 96.8" needs no exotic explanation, and the same handicap
applies to the DRONE results: the 19/20 record flow, every ladder rung and every basis
comparison was a snapshot taken mid-trajectory at near-peak LR with batch 32.
FIX ORDER (cheapest first):
 1. Match the schedule to the run: warmup 500, decay_steps = num_train_steps. FREE — no extra
    compute — and it is the difference between a settled checkpoint and a random high-LR one.
 2. Raise batch size (32 -> 128/256): 4-8x samples per step; batch 32 badly underuses a 98 GB GPU.
 3. Full fine-tune instead of LoRA (memory is available), EMA back on.
 4. extra_delta_transform for LIBERO-only work — but it changes the normalised action space, so
    U and every prior must be refit; defer until 1-3 are done.
 5. Longer runs, last, since 1-2 may make them unnecessary.
TEST LAUNCHED on the record recipe, where the reference is strongest (19/20 full, 15/20 clean):
  gate_pin_sched  = 5k steps, batch 32,  warmup 500 / decay 5000  (isolates the schedule)
  gate_pin_schedb = 5k steps, batch 128, warmup 500 / decay 5000  (schedule + more samples)
Both evaluated with the same tail-weighted prior and serving path, 20 trials each. If the
schedule alone moves the record cell, every number in the project is a floor rather than a
measurement, and the ladder should be rerun on the fixed recipe before anything is published.

## 2026-08-10 — PREDICTOR-MATCHED BASIS (lang_rrr): offline Pareto win does NOT transfer
gate_pin_langrrr (5k steps, same data/recipe) + language prior on the same basis, 10 trials/side:
                          transit   full   clean
  lang_rrr basis           7/20     6/20   15/20   (L 7/10 transit, 6/10 full; R 0/10 transit)
  deployed RRR basis      20/20    13/20    5/20   (L 7/10 full, R 6/10 full)
So the basis that DOMINATED offline — equal capture (82.3 vs 82.8) and better predictability
(0.918/0.969 vs 0.905/0.964) — is WORSE closed loop: completions 6/20 vs 13/20, and the RIGHT
scene collapses to 0/10 transits (it never reaches the gate at all).
This is the SIXTH independent instance of the offline/closed-loop chasm, and the cleanest yet:
a strictly Pareto-better basis on both offline axes lost half the completions and an entire
scene. Direct answer to the two-purposes question: raising predictability at equal capture did
not help; something else about WHICH directions are chosen matters more than either metric.
Working hypothesis for the right-scene collapse: fitting RRR against (state, language) with a
linear map favours directions that are LINEARLY predictable from state, and the right route's
distinguishing geometry may be exactly what a linear state map cannot separate (the right task
has always been the fragile one — +x aim bias, label bug, paraphrase collapse, goal phase).
Selecting for linear-state-predictability can therefore discard the very direction that
distinguishes the harder route.
Caveat: the clearance numbers flatter both weak arms — a flight that never approaches the gate
scores "clean" (the zero-pin artifact). lang_rrr's 15/20 clean includes 10 right-scene runs
that never transit, so it is not evidence of better flight quality.
Also: this run carries the same optimisation handicap as everything else (LR at 96% of peak at
the 5k stop). The schedule-fix arms now running will say whether basis comparisons at this
budget are even measuring the basis.

## 2026-08-10 — BASIS DIAGNOSIS: the two bases are the SAME SUBSPACE; the gap is RUN VARIANCE
Compared deployed RRR vs lang_rrr on everything offline capture/predictability cannot see:
  principal angles                 [0.8, 0.9, 1.8, 4.1, 25.2] deg — 4 of 5 within 4 deg
  per-axis expressivity x/y/z/yaw  0.99/0.99/0.98/0.97   vs  0.98/0.99/0.95/0.97
  temporal weighting early/late     52/48                vs   47/53
  command scale (per-dim std of c)  7.52 5.17 4.86 4.07 3.64  vs  7.51 5.17 4.83 4.05 3.59
  left-vs-right task separability   7.25                 vs   7.58  (lang_rrr better)
  per-task predictability L / R      0.957 / 0.951        vs   0.974 / 0.961  (lang_rrr better)
The two bases are nearly identical and lang_rrr is equal-or-better on EVERY measured property,
including right-task predictability — yet closed loop gave 13/20 vs 6/20 with the right scene
going 6/10 -> 0/10. That difference cannot be attributed to the basis.
CONCLUSION: what we measured is TRAINING-RUN VARIANCE, not basis quality. Consistent with the
optimisation diagnosis (checkpoints sampled at 96% of peak LR are draws from a moving
trajectory), and it puts every single-run comparison in this project in doubt — K=8 vs K=5,
gripper-free, per-suite, disp6/disp7, lang_rrr, and plausibly the LIBERO basis arms too.
Launched the measurement that should have existed from the start: run_seed_variance.sh trains
the IDENTICAL record recipe at seeds 7 and 123 (only --seed differs) and runs the same 20-trial
battery. The spread across seeds is the error bar that belongs on every basis/config comparison
we have reported. With the schedule-fix arms also running, the pair of results tells us both
how bad the variance is and whether annealing removes it.

## 2026-08-10 — ORACLE CONTROL on both bases: the lang_rrr FLOW is broken, not its basis
Demo-oracle commands (k-NN over demo state->chunk-c in each flow's OWN basis), 5 trials/side:
                                    transit   full   clean
  deployed RRR flow + oracle         10/10    10/10   8/10
  lang_rrr flow + oracle             10/10     1/10  10/10
Both flows TRANSIT perfectly under oracle commands (10/10 each) — so the lang_rrr flow can fly
routes. It cannot FINISH: 9 of 10 runs transit and then fail the goal dwell, and the same flow
under its learned prior could not even transit the right scene.
So the 13/20 vs 6/20 gap is NOT a command-prediction difference. Given near-perfect commands,
one flow scores 10/10 and the other 1/10. Since the two bases agree to within 4 degrees on 4 of
5 directions and lang_rrr is equal-or-better on every offline property, the difference is a
property of the TRAINING RUN, not of the subspace. The seed-variance measurement (identical
recipe, seeds 7 and 123) will size that.
Two further notes:
 - The deployed flow + oracle scored 7/10 full in the component battery and 10/10 here — same
   checkpoint, same command source, different stochastic rollouts. 3/10 of swing at n=10, so
   ROLLOUT variance is also large and 10-trial cells cannot resolve small differences.
 - lang_rrr's 10/10 "clean" is again the conditional-metric artifact in mild form: its flights
   transit but do not settle, so they spend less time near structure.
Practical rule going out of this: report clearance CONDITIONAL on transit, and never compare
configs on single training runs at n=10.

## 2026-08-11 — process note: I violated my own saved rule and lost ~5 h of GPU1 time
run_seed_variance.sh was edited (to re-gate it) WHILE a bash process was executing it, so bash
resumed at a shifted byte offset and died on garbage ("r k in $(seq 1 200)"). The file on disk
was valid; the running interpreter was not. GPU1 then sat idle from roughly 23:50 to 04:58.
This is exactly the failure recorded in memory as never-edit-running-script after the same
thing killed run_mech10.sh on 2026-08-07. The rule exists; I did not apply it. Concrete
practice to follow instead: to change a chain that is already running, kill the waiter FIRST,
then edit, then relaunch — or write the change to a new filename.
Relaunched; seed 7 training now.

## 2026-08-11 — MATCHED SCHEDULE: not the fix, and a pattern emerges across fresh runs
gate_pin_sched = record recipe, 5k steps, batch 32, LR schedule MATCHED to the run
(warmup 500, decay_steps 5000) instead of truncating a 30k schedule at 96% of peak.
                              transit   full   clean
  matched schedule (this run)  20/20     9/20   18/20   (L 9/10 full, R 0/10 full)
  truncated schedule (record)  20/20    19/20   15/20   (L 10/10, R 9/10)
So annealing did NOT reproduce the record, and the loss is entirely the RIGHT scene: 10/10
transits there but 0/10 goal dwell. Left is essentially unchanged (9/10 vs 10/10).
THE PATTERN THAT NOW MATTERS: this is the SECOND fresh 5k-step run of this recipe to transit
both scenes and then fail the right goal phase completely — lang_rrr did exactly the same
(right 0/10 under its prior, 1/10 under oracle). The record flow (gate_both_pin_rrr) handles
both scenes. Two of two fresh runs fail the right goal phase; the record run does not.
Leading interpretation: the 19/20 record is an OUTLIER RUN rather than a reproducible property
of the recipe, and the right goal phase is a bistable mode that most runs lose. If the two
seed repeats (identical recipe, seeds 7 and 123, training now) also collapse on the right, then
the record-board entry is not reproducible and every comparison drawn against it — including
the ladder crossing and the basis results — was measured against a lucky checkpoint.
Alternative to rule out: something in the training path changed since the record flow was
trained (pi0.py has since gained _PIN_ZERO and a steer_c import; both are inert unless their
env vars are set, but "inert" should be verified rather than assumed).
Also note the schedule fix was NOT free in the sense I claimed: it changed the outcome, just
not favourably, so schedule choice is now another axis that needs seed repeats to evaluate.

## 2026-08-11 — WHY THE RIGHT GOAL PHASE IS FRAGILE: it is MARGINAL, not bistable
Key facts established:
 - The goal is the SAME PHYSICAL POINT for left and right: goal_position [1.525,-0.615,1.0]
   in both safety YAMLs (a shared "stuffed animal" hover). Only the ROUTE differs (which gate).
 - Judge tolerance is a box with half-extents [0.3, 0.3, 0.5] m, and "goal" requires >=1
   post-transit frame inside it.
 - The demos are equally tight for both tasks: terminal position mean (1.52,-0.61,1.00) with
   sd ~0.02 m and final-25-step motion 0.06-0.07 m for LEFT and RIGHT alike. So the DATA does
   not distinguish the two goal phases at all.
Closest approach to the goal (mean over 10 rollouts):
                     LEFT              RIGHT
  record run         0.16 m (pass)     0.27 m (pass, 9/10)
  matched-schedule   0.22 m (pass)     1.23 m (fail, 0/10)
  lang_rrr           --                0.38 m (fail, 0/10)
So on the LEFT every run lands 0.16-0.22 m from the goal, comfortably inside the 0.3 m box.
On the RIGHT the record run gets 0.27 m — just inside — and the other runs get 0.38 m and
1.23 m, just outside and far outside. The left task has ~40% margin; the right task has ~10%.
INTERPRETATION: the right goal phase is not a bistable skill that some runs learn and others
do not; it is a MARGINAL one. Terminal accuracy after the right gate sits right at the judge's
tolerance, so ordinary run-to-run variation flips it between 9/10 and 0/10 while the same
variation is invisible on the left. That explains the whole history of right-side fragility
(aim bias, paraphrase collapse, goal phase) as one thing: less margin, same noise.
WHY less margin on the right: the goal is a single fixed point, so the approach geometry
differs — after the right gate the drone reaches the same hover point from a different
direction and distance, accumulating more terminal error. This is a property of the task pair,
not of the policy.
CONSEQUENCES:
 1. Report right-side completion as marginal and always with the closest-approach distance,
    which is the continuous quantity underneath a binary that sits on a threshold.
 2. Comparisons that hinge on right-side completions (the K=8 verdict, lang_rrr, the schedule
    fix, most single-run basis results) are threshold-crossing noise, not effects.
 3. The actionable target is terminal PRECISION after the right gate — closest approach
    0.27 -> 0.15 m would give the right task the same margin the left already has. That is a
    goal-phase precision problem, i.e. exactly what the tail-weighted prior addressed on the
    left, and it argues for a right-specific terminal fix rather than more basis search.

## 2026-08-11 — DENIS'S HORIZON HYPOTHESIS IS CORRECT, and it is a data-pipeline defect
Claim tested: "the pin is 50 chunks, so 10 steps from the goal it still predicts a 50-step
thing and ~40 of those steps carry phantom movement."
MEASURED, ten steps from the end of a demo:
  true remaining displacement   (0.021, 0.039, 0.015) m
  padded 50-step chunk encodes  (0.154, 0.357, 0.092) m   -> 8.5x inflation
Cause: LeRobot pads out-of-episode chunk indices by CLAMPING to ep_end-1, i.e. REPEATING THE
LAST ACTION (LeRobotDataset._get_query_indices). Actions are per-step deltas, so repeating the
last delta 40 times injects 0.13-0.36 m of motion that never happens. Our prior's c_of pads
identically, so prior and flow agree with each other — and both encode the phantom.
END TO END: the deployed prior, queried ten steps from a demo's end, commands
  LEFT   (-0.358, 0.068, 0.157) m   while only (0.005, 0.048, 0.004) m remains
  RIGHT  (-0.142, 0.214, 0.095) m   while only (-0.028, -0.034, 0.006) m remains
i.e. tens of centimetres commanded where centimetres remain. The client executes only 8 of the
50 steps per inference, but the pin constrains the chunk's TOTAL coarse displacement, so the
flow spreads that phantom motion across the chunk — including the 8 executed steps. That is the
overshoot-at-the-goal mechanism, and it explains why terminal precision is marginal (right task
closest approach 0.27 m against a 0.3 m box).
A SECOND, INDEPENDENT DEFECT found while checking: the one-hot prior (the record system's
command source) trains on rows range(0, Tn-H, 4), so rows stop at progress 0.783 — ZERO rows
above 0.85. It has never seen a state in the final ~22% of an episode. The tail weighting that
produced 19/20 therefore upweighted rows at 0.75-0.783, i.e. 20-60 steps BEFORE the end, not
the terminal hover. The language prior (range(0, Tn-5, 6)) does cover it (500 rows above 0.85).
FIXES, in order:
 1. Compute the pin target from the REAL part of the chunk only (zero the padded tail in delta
    space, or mask with actions_is_pad, which openpi already puts in the batch). This changes
    the pin target definition, so it needs a flow retrain + prior rebuild, but it removes a
    systematic terminal bias instead of tuning around it.
 2. Extend the one-hot prior's rows to Tn-5 so the terminal phase is represented at all.
 3. Re-examine the tail-weighting result afterwards: it may have been compensating for (1)-(2).
This also predicts the K=8 overshoot: a temporal basis makes the phantom tail commandable in
its own right, so the late-half directions carry the padding artifact explicitly.
YAW: cannot test Denis's other hypothesis yet — gate_video_overlay saves only positions
(traj = 3 columns), so no rollout yaw exists to compare against the demos' 1.80 rad span.
Adding yaw to the saved trajectory is a one-line change and is needed for the side-by-side.

## 2026-08-11 — PADDING FIX IMPLEMENTED (SNMVP_ZERO_PAD_ACTIONS=1)
Applied in three places so the flow's target, the pin's command and the prior's target all
agree, and all reflect real motion:
 1. openpi data_loader.create_torch_dataset — a transform that zeroes action steps flagged by
    `actions_is_pad` in RAW DELTA space ("stay put after the episode ends"). Verified on a
    terminal item: 47/50 steps padded, max |action| on padded steps now 0.0, and the chunk's
    net displacement drops from the inflated value to (-0.026, -0.019, 0.002) m.
 2. make_progress_prior4.py — same padding rule for the prior's targets, AND rows extended
    from range(0, Tn-H, 4) to range(0, Tn-5, 4) so the terminal phase is represented at all
    (previously rows stopped at progress 0.783, zero rows above 0.85).
 3. langprior_rebasis.py — same padding rule, for the enumeration-free prior.
All three are gated behind the env var, so previous behaviour is reproducible for comparison.
First result: the corrected prior fits BETTER than the old one despite covering 22% more of
each episode — held c-R2 0.9712 (old: 0.9705) on strictly harder coverage, which is what you
would expect if the phantom tail was noise the old prior had to absorb.
Flow now training with the fix; battery to follow (10 trials/side) against the record's
19/20 full and 15/20 clean. Whether this recovers or exceeds the record matters less than
whether it makes the RIGHT task less marginal: the prediction is that removing a systematic
outward bias at the goal should pull closest-approach in from 0.27 m toward the left task's
0.16-0.22 m, which is what would make right-side results reproducible rather than
threshold-flipping.

## 2026-08-11 — SEED VARIANCE: the 19/20 record is an OUTLIER; completions are seed-unstable
Identical recipe (pi0_gate, 5k steps, RRR basis, hard pin, tail-weighted prior, same serving
path); ONLY --seed differs. Completions per 20:
  seed 42 (the record)  19/20     (L 10/10, R 9/10)
  seed 7                 9/20     (L  8/10, R 1/10)
  seed 123               2/20     (L  0/10, R 2/10)
  -> mean 10.0/20, sd 7.0, range 2-19. The record sits 1.3 sd above its own recipe's mean.
For reference the other fresh runs of the same recipe family land in the same band:
matched-schedule 9/20, lang_rrr 6/20.
CRITICAL SEPARATION — transits are stable, completions are not:
  every one of these runs transits 20/20 (seed 7, seed 123, matched-schedule, record,
  lang_rrr's flow under oracle, the K=8 arms). Seed 123 scores 0/10 COMPLETIONS on the left
  while still transiting 10/10.
So route-following is a reproducible property of the recipe; terminal goal-dwell is not. This
is exactly consistent with the marginality analysis (the goal box is 0.3 m and runs land
0.16-1.23 m away, straddling the threshold) and with the padding defect (a systematic outward
bias at the goal, now fixed and retraining).
WHAT THIS INVALIDATES (all single-run comparisons decided on COMPLETIONS):
  - the K=8 vs K=5 verdict (10/20 vs 19/20) and its "clearance for completions" trade
  - lang_rrr vs deployed basis (6/20 vs 13/20)
  - the matched-schedule verdict (9/20 vs 19/20)
  - the gripper-free / per-suite / disp6 / disp7 LIBERO arms
  - the LADDER's completion numbers (3/4/3/13/19 per rung — one training run each)
WHAT SURVIVES, because it rests on transits or on much larger gaps:
  - pin vs no-pin TRANSITS in the ladder (20/20 across five pin runs vs 10-14/20 across three
    no-pin runs) — a large, repeatedly reproduced gap on the stable metric
  - zero-pin 0/20 transits vs pin 20/20 (transit-level, and a huge gap)
  - the additive-edit control's steering equivalence (transit-level)
  - the sigma sweep on LIBERO (0.16 / 0.24 / 0.49) — much larger than plausible seed noise,
    though it deserves seed repeats before publication
  - the demo-oracle ceiling result (deployed flow + oracle 10/10) — but this too is one flow
REQUIRED CHANGES TO PRACTICE, effective now:
  1. Report completions as mean +/- sd over >=3 seeds, never from a single run.
  2. Report closest-approach distance alongside the binary, since the binary sits on a threshold.
  3. State the record as "best of N runs" with the distribution, not as the recipe's performance.
  4. Judge configuration changes on TRANSITS (stable) or on seed-averaged completions.

## 2026-08-11 — Infrastructure so failed runs stop wasting GPU time (Denis)
Two incidents today cost roughly 8 h of idle GPU: (1) run_seed_variance.sh was edited while a
process was executing it, so bash resumed at a shifted offset and died; (2) the padding-fix
train died at startup because a dataset transform defined inside a function could not be
pickled by the DataLoader's spawned workers — and my single-item test had passed because it ran
in the main process with no workers. In both cases nothing noticed until I checked manually.
Added:
 1. /home/ubuntu/jobq.sh — per-GPU supervisor. Reads ctxrun/queue_gpu<N>.txt (one command per
    line, re-read after every job), runs jobs sequentially, waits for the GPU to be free before
    starting each, and ALWAYS advances on failure, appending OK/FAIL + duration to
    ctxrun/jobq_gpu<N>.status. A crashed job can no longer leave a GPU idle: the next queued job
    starts. Both supervisors are running.
 2. /home/ubuntu/preflight.sh — CPU-only, no GPU allocation, so it can run beside training.
    Builds the config, creates the dataset, and pulls a batch through a DataLoader with
    num_workers=2 and spawn context. That is precisely the check that would have caught the
    pickling failure in seconds; it also catches unknown configs, missing norm assets and
    env-gated code paths that only error when the flag is set.
 3. Practice: every long training launch goes through preflight first and into a queue file
    rather than being launched directly.
Note the preflight run itself was slow on CPU (LeRobot metadata load + worker spawn), so it is
a minutes-scale gate, not seconds — still two orders of magnitude cheaper than discovering the
same fault three hours into a chain. Also aborted a self-test that would have temporarily
reintroduced the defect into a live file; verified data_loader.py is intact and parses.

## 2026-08-11 — PADDING FIX REGRESSES: it interacts with the tail weighting
Zero-padded targets + the same recipe (prior rows extended into the terminal phase, TAILW=4):
                        completions   closest approach to goal
  zero-pad fix            0/20          1.51 m (median 1.61, worst 1.72)
  record (old padding)   19/20          0.22 m
  seed 7 (old padding)    9/20          0.74 m
  seed 123 (old padding)  2/20          0.82 m
Transits are unaffected (left 10/10, right 8/10) — it flies the routes and then stops 1.5 m
short. On the CONTINUOUS metric this is far outside the three-seed spread, so unlike the binary
comparisons this is a real effect, not noise. (Reporting closest approach is what makes that
call possible — the binary alone would just say 0/20.)
MECHANISM (hypothesis, testing now): the fix interacts with the tail weighting. Zero-padding
makes terminal chunk targets ~0, and the prior's newly-added terminal rows are the ones TAILW=4
upweights, so the prior is trained hardest on rows that say "do not move" -> it commands stop
too early. Corollary, and uncomfortable: the 19/20 record may have depended on two errors
partially CANCELLING — the phantom forward motion in padded targets offsetting a stopping bias
from tail weighting. Removing one without the other exposes the other.
TEST LAUNCHED (cheap: reuses the zero-pad flow checkpoint, rebuilds only the prior):
  zeropad flow + prior with TAILW=1, same battery.
If closest approach returns to the 0.2-0.4 m band, the interaction hypothesis holds and the
correct recipe is "zero-padded targets, no tail weighting". If it still stops short, the
stopping bias comes from the FLOW trained on zero-padded targets, not from the prior.

## 2026-08-11 — PADDING FIX, second seed: I over-called the regression an hour ago
Zero-padded targets, two seeds, 20 trials each:
                     completions   closest approach   clearance
  zeropad seed 42       0/20           1.51 m           8/20
  zeropad seed  7       9/20           0.73 m          20/20  (perfect)
Old padding, three seeds: 19/20 @ 0.22 m, 9/20 @ 0.74 m, 2/20 @ 0.82 m.
RETRACTION of "the padding fix is a clear regression": that was based on ONE zero-pad seed
(1.51 m) against the three-seed old-padding range (0.22-0.82 m). The second zero-pad seed lands
at 0.73 m / 9-20 completions — squarely inside the old range. So closest approach is ALSO
seed-unstable (zero-pad 0.73-1.51, old-pad 0.22-0.82), and I treated a continuous metric as
though it were noise-free after spending the day establishing that binaries are not. Two seeds
per arm cannot separate these distributions: zero-pad mean 4.5/20 vs old-pad mean 10.0/20, with
per-arm spreads of 7-9 points.
What is NOT explained by seed noise: zeropad seed 7 achieved 20/20 CLEARANCE-CLEAN, which no
old-padding run reached (best 15/20, typical 8-16/20), while also transiting 20/20. If that
holds up it would be the fix doing what the mechanism predicts — removing a systematic outward
bias tightens the flight path — with terminal dwell still limited by something else.
STATUS OF THE INTERACTION HYPOTHESIS (tail weighting x zero padding): still worth testing, but
the motivating observation (seed 42's 1.51 m) is now partly seed noise, so the TAILW=1 arm
should be read as "does removing tail weighting help on top of zero padding", not as a fix for
a regression that may not exist. That battery finishes shortly.
HONEST BOTTOM LINE for the day: nothing in the padding line is resolvable at 2 seeds. Any claim
here needs >=4-5 seeds per arm, which is ~14 h of GPU per arm at 5k steps. That cost is the real
finding: this recipe's run-to-run variance is large enough that cheap comparisons cannot work.

## 2026-08-11 — RECIPE DECISION (Denis): no tail weighting. It was a hack for the padding bug.
Denis: "we shouldn't do tail weighting. i think tail weighting was a hack for a bug." The
evidence supports that: tail weighting upweighted rows whose targets contained phantom forward
motion from repeat-last padding, so it was compensating for a data defect rather than fixing a
goal-phase modelling gap. Both prior builders now default to TAILW=1 (env-overridable).
Result of the first no-weighting arm (zero-padded targets, same flow checkpoint as seed 42,
prior rebuilt without weighting, 20 trials):
  completions 7/20 (L 4/10, R 3/10) | closest approach 0.69 m (median 0.56) | clearance 7/20
Against the same flow WITH weighting: 0/20, 1.51 m. So removing the weighting moved that flow
from 1.51 m to 0.69 m closest approach and from 0 to 7 completions — and it is the first arm
where the RIGHT scene completes at all under the corrected padding (3/10).
Current picture of the corrected recipe (zero-padded targets):
  seed 42 + tail weighting   0/20   1.51 m
  seed 42 + NO weighting     7/20   0.69 m
  seed  7 + tail weighting   9/20   0.73 m   (20/20 clearance-clean)
Interpretation, stated with the variance caveat: no-weighting looks better than weighting on the
same flow, and the corrected-padding arms cluster around 0.7 m closest approach versus the old
recipe's 0.22-0.82 m. Nothing here separates at 1-2 seeds; the value of these runs is that they
remove a KNOWN defect and a KNOWN compensating hack from the recipe, which is progress in
correctness even where it is not yet progress in score.
CANONICAL RECIPE going forward: zero-padded action targets, prior rows through the terminal
phase, NO tail weighting.
SEEDS (Denis, 2026-08-11): ONE seed while iterating — that is enough to choose what to try
next. Spend seeds only when DECLARING something (record board, paper, a claimed improvement).
The mistake this week was not running single seeds; it was narrating mechanisms from
single-seed differences. Iterate at n=1, hold conclusions loosely, validate before claiming.

## 2026-08-11 — EXECUTION-WINDOW PIN (acting on Denis's horizon hypothesis)
Measured on zero-padded chunks: of the displacement the pin COMMANDS, only 0.16 lands inside
the 8 steps the client executes before replanning (deployed RRR K=5). The other 84% is motion
in steps 9-50 that are discarded. So each replan executes ~1/6 of what it asked for, the drone
converges too slowly to reach the goal inside the episode, and the old repeat-last padding's
8.5x inflation was accidentally compensating for exactly this ratio. That is the unified
explanation for stopping short, and it makes Denis's "the 50-step chunk is bad near the goal"
the root cause rather than a side effect.
New basis pin_U_exec8full_gate.npy (K=8): per-axis displacement over the EXECUTED window
(first 8 steps) PLUS per-axis full-chunk displacement.
  capture 78.2% (deployed RRR 78.3% — no loss)
  executed/total commanded displacement: 1.00 for the executed block (0.16 for the deployed basis)
Launched with the corrected recipe (zero-padded targets, terminal-phase prior rows, TAILW=1),
single seed per the iteration policy. Prediction: if the convergence-rate account is right, this
should reach the goal rather than stalling ~0.7 m short, because the commanded displacement is
now what actually gets flown.

## 2026-08-11 — THE EXECUTION FRACTION WAS THE BUG. APC=50 gives 10/10 with 0.07 m precision
Denis: "what if you just tried increasing from 8 to like 25 or even 50" — instead of changing
the basis. He was right, and it needed no retraining: APC is purely serving-side (how many of
the 50 chunk steps the client executes before replanning).
Same flow (gate_pin_zeropad), same prior (no tail weighting), same basis; only APC differs.
Total executed steps held comparable (8x40=320, 25x14=350, 50x7=350), so this is NOT extra
flight time:
  APC   completions   clearance-clean   closest approach   final distance
   8      7/20           7/20              0.69 m            0.88 m
  25      9/10          10/10              0.16 m            0.21 m
  50     10/10          10/10              0.07 m            0.13 m
For comparison the old 19/20 "record" (APC=8, old padding, tail weighting) reached 0.22 m
closest and 0.58 m final — i.e. APC=50 is better on the continuous metric than the best run we
had ever seen, and it does it on BOTH scenes with perfect clearance.
WHY: the pin commands the whole chunk's coarse displacement, but at APC=8 only 0.16 of that
displacement is ever flown before the next replan overwrote it. The drone was executing about a
sixth of every command, converging too slowly to arrive. Raising APC honours the command.
THIS REFRAMES MUCH OF THE PROJECT'S HISTORY. Every drone result to date used APC=8:
 - the "goal phase problem" and the right task's marginal 0.27 m approach
 - tail weighting (a hack that inflated commands to compensate)
 - repeat-last padding's 8.5x phantom inflation, which was ALSO compensating
 - the K=8 temporal basis's goal overshoot, and probably several seed-noise verdicts
A serving parameter, not the method, was the dominant limiter. Note also that flight time is not
a constraint (Denis), so APC can be raised freely; finishing sooner is a bonus, not a criterion.
STATUS: n=5/side, single seed — a strong lead, not yet a claim. Next: 10/side plus a second
seed at APC=50, then re-run the comparisons that were decided at APC=8 (K=8 vs K=5, the ladder,
the language prior, LIBERO's replan setting) since their conclusions may not survive.

## 2026-08-11 — NEXT STEPS after the execution-fraction finding (mostly INFERENCE-ONLY)
Because the execution fraction is a SERVING parameter, nearly everything that needs redoing can
be redone on checkpoints we already have. Ordered plan:
 1. RUNNING — APC=50 with the enumeration-free language prior: single gates, centre gates, and
    the two repo compositional prompts. Establishes the headline on current components.
 2. QUEUED  — LIBERO replan sweep (inference only). LIBERO used replan_steps=5 of a 50-step
    chunk = ~1/10 execution fraction, WORSE than the drone's 1/6. Every LIBERO number was
    collected there, and the handicap is asymmetric: a no-pin policy gets a fresh chunk each
    replan, whereas the pin's entire mechanism is a chunk-level coarse command. soft-pin vs
    no-pin at replan 25/50 on the existing 2-demo checkpoints may change the cross-domain story
    (currently a non-significant 0.49 vs 0.41).
 3. Ladder re-evaluation at APC=50 — inference only; every rung's checkpoint still exists, so
    Figure 4 can be rebuilt without retraining. Note transits were already saturated at APC=8,
    so the interesting change is the COMPLETION curve.
 4. Claim tier for the headline: 10 trials/side plus a second seed at APC=50.
 5. Only then, revisit basis questions (K=8, exec-window, per-suite) — all were decided at
    APC=8 and several were inside seed noise anyway.
 6. Update docs/PAPER_OUTLINE.md: the ladder completions, the K=8 trade, the tail-weighting
    result and the LIBERO numbers are all superseded or pending re-measurement.

## 2026-08-11 — APC=50 with the ENUMERATION-FREE prior: nothing clean (review page 056d94a5)
Flow `gate_pin_zeropad/4999`, basis `pin_U_gate_rrr_k5`, command source `langprior_zeropad.pt`
(MLP over model state + PCA-64 of the post-fusion language-token embedding; no task list, live
embeddings), 5 trials per group, strict judge + `gate_clearance.py`:
  single left    0/5 transit   (min clearance 0.069-0.154 m -> passes the gate off-centre)
  single right   0/5 transit   (min clearance 0.200-0.252 m -> misses the opening entirely)
  centre from L  5/5 transit   0/5 clearance-clean (0.020-0.062 m)
  centre from R  4/5 transit   0/5 clearance-clean (0.003-0.025 m)
  compound L/R   0/5, 0 of 2 gates latched, both prompts
Under the scoring rule (transit + clearance + video) NO group is a success. Contrast, same flow and
same APC=50, one-hot scaffold prior: left 5/5 and right 5/5, all clearance-clean.
DIAGNOSTIC SIGNATURE: the grounded prior's flights run ~1 m further downrange -- max x 2.35-2.85 m
vs 1.52-1.84 m for one-hot -- then return to hover near the goal box. That is what open-loop
execution of a slightly over-long command looks like once replanning stops truncating it: APC=8 was
flying only 0.16 of each command, which HID the error. So raising the execution fraction is not a
free win; it removes a safety net and is only correct when the command is right.
OFFLINE METRICS DO NOT PREDICT IT. Per-task held c-R2 0.9027-0.9589, command MAE 4-17 cm per axis
(early/transit/tail) on chunks whose commanded displacement is 20-30 cm -- as good as the one-hot
scaffold. The chasm between offline command accuracy and closed-loop competence is reproduced here
with a well-fit grounded prior, so it is not explained by command L2.
METHOD NOTE (cost me two wrong conclusions before I caught it): the langprior feature cache
`langprior_feats.npz` is built at STRIDE 6, not 4. Reconstructing targets at stride 4 misaligns
every row within an episode and yields per-task R2 of -1.8 to -0.2, which looks exactly like a
basis or padding mismatch. Verify alignment against the cached `frac` array (frac == t/Tn) before
believing any offline number recomputed from a cache.
Also added: `experiments/rung3/extract_scene_cloud.py` -- whole-scene splat cloud export (positions
+ SH DC colours, scene edits applied, cropped to the flight volume) for the 3D trajectory viewers.

## 2026-08-11 — LIBERO replan sweep (inference only, 2 demos/task, libero_spatial, 5 trials/task)
                      replan=5 (as published)   replan=25   replan=50
  soft-pin sigma 0.7        0.49                  0.50        0.42
  no-pin scratch            0.41                  0.44        0.18
At the standard setting the two are equal (0.49 vs 0.41, inside protocol noise). As the execution
fraction rises the no-pin policy COLLAPSES (0.44 -> 0.18) while the pin degrades gently (0.50 ->
0.42): at fully open-loop 50-step chunks the pin is 2.3x the baseline. This is the LIBERO analogue of
the drone "pin training buys chunk coherence" finding — the pin's benefit is specifically robustness
to executing a whole chunk, which the standard replan_steps=5 protocol hides by re-planning ten times
per chunk. It does NOT show the pin beating the baseline at the published protocol.
Caveat: single run per cell (iteration tier), 50 episodes each; treat as a lead, not a claim.

## 2026-08-11 — BASIS AUDIT (clean) and a guard so it stays that way
Asked whether any current predictor is paired with the wrong RRR basis. Audited three ways:
 1. chain scripts — every gate chain uses `pin_U_gate_rrr_k5.npy` for SNMVP_PIN_U (flow training),
    UPATH (prior building) and --pin-u (serving), including all five ladder rungs; LIBERO uses
    `pin_U_rrr_k5_shared.npy` with its matching prior.
 2. numerical fingerprint — score each deployed prior's predictions against targets recomputed under
    all 27 candidate bases: `langprior_zeropad.pt` -> pin_U_gate_rrr_k5 at pooled +0.9449 (every
    other basis -0.6 to -54); `langprior_rrr.pt` -> same basis at +0.9281.
 3. the +0.9449 reproduces the value printed at training time EXACTLY, which also rules out a
    padding-convention mismatch (zero-pad vs repeat-last moves the tail rows and would shift it).
So no basis bug. But the only guard was matching env vars across chain scripts by hand — the class of
bug that already cost us the LIBERO arms once. Fixed structurally:
 - `experiments/rung3/pin_basis.py`: `stamp(upath)` records {pin_u, pin_u_sha256, pin_u_shape} in the
   prior checkpoint; `verify(d, path)` refuses a mismatch at serve time (K mismatch too) and warns
   loudly for unstamped legacy priors instead of implying the pairing was checked.
 - builders stamp: make_progress_prior4.py, langprior_rebasis.py, libero_prior.py
 - servers verify: serve_gate_pin_langprior.py, serve_gate_pin_prog4.py, serve_libero_pin.py
 - backfilled the stamp onto the five gate priors whose basis is established by both fingerprint and
   chain script. Tested: correct pairing passes, wrong basis and wrong K both refused.

## 2026-08-11 — ladder re-evaluated at APC=50 with the grounded prior: overshoot at EVERY rung
Transit judge, 5 trials/side, `langprior_rrr.pt` + pin_U_gate_rrr_k5 (padding convention matched to
the old-padding rung flows):
  rung   left        right     max x reached (left/right)
  n12    5/5 transit  0/5      2.62 / 2.44     (left 0/5 clearance-clean — clips)
  n40    5/5 transit  0/5      2.32 / 2.35
  n160   0/5          0/5      2.50 / 2.51
  n240   0/5          0/5      2.59 / 2.70
  full   5/5 transit  0/5      2.16 / 2.07     (left 5/5 clean, right 3/5 clean)
Every rung overshoots to x 2.07-2.70 m and stops PAST the goal box (ends x 1.85-2.13 vs goal
1.525 +- 0.3). Five different flows, one prior, one signature -> the defect is command-side, and it
is the TERMINAL command (when to stop/turn) not the route: early flight reaches the gate correctly.
Left-side non-monotonicity (5/5, 5/5, 0/5, 0/5, 5/5) is not a data-quantity effect; with one run per
cell and this overshoot dominating, the ladder cannot be read at APC=50 until the prior is fixed.
HYPOTHESIS UNDER TEST (`run_cmdshift.sh`, both GPUs): the prior is fit only on demo states, so once a
long command carries the drone 50 steps off the demo manifold both of its inputs are extrapolations.
latch12 (embedding frozen after 12 frames, state the only drifting input) vs latch0 (live embedding)
separates the two inputs; CLOG records (position, c, e64) per inference for the quantitative read.

## 2026-08-11 — the grounded prior's failure is NOT live-embedding drift
`run_cmdshift.sh`, flow gate_pin_zeropad/4999 + langprior_zeropad, APC=50, 5 trials/side:
  latch12 (embedding averaged over 12 frames then FROZEN)  left 0/5, right 0/5
  latch0  (embedding recomputed live every inference)      left 0/5, right 0/5
Identical. Freezing the language input — which removes any in-flight drift of the image-dependent
embedding, and with it the "e64 acts as a second copy of the state" story — changes nothing. So the
defect is not the language channel. Left clearance also degrades (latch12 2/5 clean, latch0 0/5),
i.e. the flights clip the gate either way.
Remaining candidates: (a) the prior's STATE-conditioned mapping off the demo manifold, (b) the
serving code path, not the prior at all. (b) is a real confound that has been sitting in the
comparison unremarked: the one-hot 10/10 came through serve_gate_pin_prog4.py at NCH=7 and the
grounded 0/10 through serve_gate_pin_langprior.py at NCH=8, so TWO things differ.
`run_prior_swap.sh` closes it — one-hot prior, grounded run's exact client config (NCH=8, APC=50,
5 trials, same flow, same basis, both scenes) — leaving the command source as the only difference.
Both servers now support CLOG (per-inference model state + c) so the two priors can be compared
offline at the states the failing flights actually visited, not only at demo states; the analysis is
`experiments/rung3/clog_analysis.py` (bins command error by distance off the demo manifold).

## 2026-08-11 — PRIOR-SWAP CONTROL: the command source is the defect, and the RIGHT GATE is clean
`run_prior_swap.sh` — one-hot prior served at the grounded run's EXACT client config (NCH=8, APC=50,
5 trials/side, flow gate_pin_zeropad/4999, basis pin_U_gate_rrr_k5), i.e. the only difference from
the 0/10 grounded run is the command source:
  left  5/5 success, 5/5 clearance-clean
  right 5/5 success, 5/5 clearance-clean
So the serving code path is exonerated and the command source is the defect. NOTE THE RIGHT COLUMN:
the right gate has been 0/10 on the record board with a persistent ~1 m +x aiming bias, and it is
clearance-clean here. Full-chunk execution appears to fix it. 5 trials = a lead; `run_swap10.sh`
(10 trials/side) is running for claim tier.

## 2026-08-11 — the measurement of command error, VALIDATED against a working command source
`clog_analysis.py` on the one-hot prior's own flight logs (the arm that flies 10/10 clean), same
direction-aware demo matching as the grounded prior's:
                        off demo manifold        command error vs the demo's own c
  one-hot   left        0.054 m (max 0.208)      0.081 m
  one-hot   right       0.011 m (max 0.033)      0.057 m
  grounded  left        0.285 m (max 0.720)      0.486 m  (0.527 m even at on-manifold states)
  grounded  right       0.505 m (max 1.224)      0.596 m  (0.475 m even at on-manifold states)
The metric is therefore diagnostic, not an artefact of position matching: a command source that works
lands within 6-8 cm of the demo command and never leaves the demo manifold; the grounded one is 6-9x
worse, INCLUDING at states close to demo states, where its offline error is 4-17 cm.

## 2026-08-11 — TRAIN/SERVE FEATURE SKEW in the grounded command source (retracts my latch reading)
I earlier read the latch result as exonerating the language channel. That was wrong: latching tests
only TIME-VARIATION of the embedding, not whether the embedding pipeline differs between training and
serving. It does.
 - At the FIRST inference of an episode the drone is at the origin, where all 100 demos also start, so
   viewpoint novelty is impossible. There the flight embedding sits at mean |z| 2.34 (left) / 3.60
   (right) against its own training distribution, with 25% / 42% of dims beyond 3 sigma. Held-out DEMO
   rows measured the same way: mean |z| 0.80, 0.1-0.2% beyond 3 sigma.
 - Cause is upstream of the prior: `langprior_pipeline` builds the cache from the STORED demo frames
   (`data_gate_synth` image/wrist), while the server embeds LIVE gsplat renders. Re-rendering each
   demo's own frame through the serving path (`render_skew_probe.py`) gives mean |Δ| ~55/255 and pixel
   correlation 0.05-0.20 vs the stored frame — same room, visibly different framing. Tried and
   REJECTED as explanations: squash vs centre-crop, 256-then-224 vs direct 224, 1024x1024 render,
   both yaw signs. So it is a camera/pipeline difference in how the synth dataset was generated, not a
   resize convention.
 - If this holds, it is a candidate explanation for the standing record-board entry that VLM-feature
   command maps score 0/5 everywhere while their offline metrics look excellent: the features trained
   on are not the features served. The flow tolerates the same skew because state + pin carry this
   task (cf. scratch pi0 transits 20/20 but completes 0/20), whereas the prior's input is 64 PCA dims
   OF that feature.
 - `feature_skew_probe.py` (two-stage: gsplat render, then openpi features) quantifies the effect on c
   at identical state and prompt. Running.
FIX IF CONFIRMED: rebuild the language cache from LIVE renders at the demo states (serving
convention), retrain the prior, re-fly. The demo states are known, so this needs no new demos.

## 2026-08-11 — ROOT CAUSE of the grounded command source's closed-loop failure: FEATURE-SOURCE SKEW
`langprior_feats.npz` — the cache BOTH language priors are fit on — was extracted with flow checkpoint
**gate_both_pin_rrr/4999** (`langprior_pipeline.py` line 23). Every recent run served
**gate_pin_zeropad/4999**. `gc.lang_pool` reads post-fusion language tokens out of the SERVED model, so
the embedding is a function of that model's VLM weights: the prior has been consuming a representation
produced by different weights than it was trained against.
Evidence chain:
 - `feature_skew_probe.py`, at identical demo states and identical prompt, embedding through the SERVED
   checkpoint: even from the STORED demo frames the features sit at mean |z| 1.7-2.0 with 18-22% of dims
   beyond 3 sigma of the cache's own distribution — while held-out rows OF that cache sit at 0.80 and
   0.1-0.4%. The stored-frame path should be identical to the cache; the only remaining difference is
   the model that produced the features. That is the skew.
 - Live-vs-stored IMAGE path is a real but SECONDARY term: it moves the commanded displacement by
   0.113-0.133 m (max 0.44), and command error vs the demo's own command goes 0.134 -> 0.207 m (left),
   0.121 -> 0.145 m (right). So the earlier "stored vs live renders" reading was directionally right
   but not the main term; the pixel comparison (corr 0.05-0.20) overstated it.
 - Explains every observation: excellent offline metrics (the cache is self-consistent), failure from
   the FIRST inference at the origin where viewpoint novelty is impossible, latching changing nothing
   (the mismatch is not time-varying), and the one-hot prior being immune (its inputs never touch the
   VLM). Candidate explanation for the standing record-board entry that VLM-feature command maps score
   0/5 everywhere with excellent offline metrics.
TEST RUNNING (`run_featmatch.sh`): serve gate_both_pin_rrr/4999 — the checkpoint the cache came from —
with langprior_rrr.pt at APC=50, 5 trials/side. Matched pairing, no retraining.
STRUCTURAL FIX (same shape as the basis stamp): `pin_basis.stamp(upath, feat_ckpt=...)` now records the
feature-source checkpoint, `verify_features(d, serve_ckpt)` reports a mismatch loudly at serve time
(non-strict by default — serving a different flow deliberately is a legitimate experiment, e.g. the
ladder rungs; silence is what we cannot afford). langprior_rebasis.py records it; the langprior server
checks it; all five existing language priors backfilled. Tested both ways.

## 2026-08-11 — CONFIRMED: matching the feature source fixes the grounded command source (left gate)
`run_featmatch.sh` — serve gate_both_pin_rrr/4999 (the checkpoint langprior_feats.npz was extracted
with) + langprior_rrr.pt, APC=50, 5 trials/side, NO retraining and no change to basis or client:
  LEFT   5/5 success, 5/5 clearance-clean   (min clearance 0.216-0.258 m, transit @80-87)
  right  0/5 success, 2/5 clearance-clean   (min clearance 0.034-0.223 m)
Left goes 0/5 -> 5/5 clean purely by pairing the prior with the flow whose VLM produced its features.
That is the enumeration-free command source flying cleanly for the FIRST time — the one-hot scaffold is
no longer the only thing that works. 5 trials = a lead; `run_fm10.sh` (10 trials) is running.
The RIGHT gate's failure MODE changed, which is itself informative: mismatched it missed the opening
entirely (closest 0.200-0.252 m, ZERO steps inside the body radius = clean miss), matched it now flies
AT the gate and clips it (15-32 steps inside the radius on 3 of 5, crossing at z~1.31-1.35 vs the demos'
z~1.5). So the right gate is now an AIM problem in z, not a route-selection problem.
NEXT (`run_featfix.sh`, running): the flow we actually want is gate_pin_zeropad (it carries the padding
fix), so re-extract the language cache with THAT checkpoint's VLM, retrain the prior under the same
basis with zero-padded targets and no tail weighting, and fly it — matched features AND matched padding.
langprior_pipeline.py CKPT and the cache path are now env-parameterised (FEAT_CKPT / CACHE); a hardcoded
feature checkpoint is what caused this in the first place.

## 2026-08-11 — the RRR basis is (nearly) PCA, so it does NOT inherit the checkpoint problem
Denis asked whether fine-tuning the VLA invalidates the RRR basis, since training moves the VLM as a
side effect and the basis was fitted on VLM features. Re-fitted it on POST-FUSION language-token
features (`refit_rrr_basis.py`, same RRR recipe: OLS features->chunk, top-K eigenvectors of Cov(Yhat)),
and measured principal angles (the only basis-invariant comparison):
  vs deployed pin_U_gate_rrr_k5 (PRE-fusion, gate_both_pin)   mean  1.7 deg  max  4.8 deg
  vs plain PCA of the chunks (no features at all)             mean  2.9 deg  max 14.0 deg
  vs pin_U_lang_rrr_gate                                     mean  5.6 deg  max 20.9 deg
  vs pin_U_gate_k5                                           mean 14.2 deg  max 43.9 deg
Held-out chunk variance captured: refit 0.8227, PCA 0.8242 (upper bound), deployed 0.8235.
FOUR OF FIVE directions are within 0.2 deg of PCA. The reason is structural, not a bug: the chunk is
~91% predictable from the features on held-out demos (per-task 0.860-0.936, mean 0.907 — checked
per-task precisely because pooled R2 misled me earlier today), so Yhat ~= Y, Cov(Yhat) ~= Cov(Y), and
"most predictable subspace" coincides with "highest-variance subspace". A ridge sweep confirms the
mechanism: as lambda grows the basis walks away from PCA (6.8 deg at 1e2, 23 deg at 1e6) while
held-out regression R2 falls (0.86 -> 0.25), i.e. the only way to make RRR differ from PCA here is to
make the regression worse.
CONSEQUENCES
 - The basis does not need re-fitting per checkpoint. It is effectively determined by the ACTION
   statistics, which do not drift when the VLM does — unlike the prior, whose input IS the feature.
   Fortunate, since U cannot be changed without retraining the flow.
 - The "U serves two purposes" tension (high-variance summary vs easy predictability) is empirically
   VACUOUS at the global level on this data: there is almost no unpredictable component for RRR to
   discount. Our RRR has been PCA with extra steps.
 - It also explains the historical basis noise: every variant we have compared is within a few degrees
   of every other, so basis A/B tests were always going to be dominated by run variance (cf. two bases
   4 deg apart giving 13/20 vs 6/20).
 - PER-TASK RRR bases DO differ sharply from the global PCA (mean 17-26 deg, max 48-84 deg), so
   per-task/per-embodiment bases remain a real avenue — but one flow trains with one global basis.
 - The 5th direction is where all variants disagree (14-29 deg); it is the low-variance tail.

## 2026-08-11 — DECISION (Denis): frozen encoder now, clearly temporary; joint training as the target
Recorded in `docs/command_source_design.md`, with the interim status also noted in CLAUDE.md so it
cannot be mistaken for the destination.
 - INTERIM (Option A): the command path reads a FROZEN encoder (base pi0 VLM or a standalone
   PaliGemma/SigLIP). Cache never goes stale, the command source is fitted once and reused across
   flows/seeds/embodiments, and nothing in the stack depends on the fine-tuned VLM any more (the basis
   already does not, since RRR ~= PCA). Cost: features are not task-adapted. That cost is a CLAIM TO
   TEST, not an assumption — the featfix arm gives the fine-tuned-feature number and a frozen-encoder
   arm gives the comparison. If task adaptation buys nothing, Option A stops being temporary.
 - TARGET (Option B): train the command head INSIDE the flow train loop. The forward pass already
   runs, so the extraction that currently takes hours as a separate pass is free in-loop, and
   c = U^T a is already computed each step for the pin. The real prize is not speed: the head is saved
   into the flow's checkpoint directory, so the pairing becomes impossible to get wrong by
   construction rather than merely detected by a stamp.
   B1 detached (stop_gradient on the features) first — solves staleness, cannot regress flow quality.
   B2 coupled (head loss backprops into the VLM, weight lambda) second, gated on flow loss not
   regressing: this is the properly-realised version of the RRR idea — MAKE the representation predict
   the subspace instead of hunting for a subspace that happens to be predictable, which on our data
   selects nothing because RRR degenerates to PCA.
   Moving-target handling: train the head throughout, then re-fit it over the final ~10% of steps when
   features are nearly stationary.
   MUST NOT CHANGE: the flow's pin keeps oracle c from the ground-truth chunk during training. Feeding
   the head's PREDICTION into the pin would let the flow co-adapt to the head's errors. (Deliberate
   scheduled-sampling/DAgger on predicted c is a separate later experiment, not part of this.)
 - ORDERING RULE, in force either way: U from action statistics -> train flow with oracle c -> only
   then extract features and fit the prior. Nothing can go stale because nothing consumes features
   before training finishes. Verified one-way coupling: the training patch extracts c from `actions`
   (`_snmvp_extract(actions)`), never from the prior, so there is no circularity to resolve.
 - EXPLICITLY NOT FIXED BY ANY OF THIS: the covariate shift. The command source errs ~0.49 m at states
   near the demo manifold and drifts 0.26-0.51 m off it in flight, vs the one-hot scaffold's 0.06-0.08 m
   error and <=0.05 m drift. Needs its own fix (perturbed states/renders, or DAgger from rollouts) and
   must not be claimed as a benefit of joint training.
 - Practical note: extraction is currently CPU-bound, not GPU-bound — the running job saturates ~8
   cores at 0% GPU (npz decompression + PIL resizes). Pre-resizing demo frames once to 224 uint8 would
   cut the interim cost substantially if Option A stays in force for long.

## 2026-08-11 — JOINT TRAINING BUILT (B1): command head inside the flow, U held FIXED
Denis: do the joint training. Implemented and launched; design in `docs/command_source_design.md`.
WHAT ALREADY EXISTED, AND WHY IT WAS THE WRONG SHAPE: `SNMVP_LEARN_U` (pi0.py) already trains a head
jointly — but it FREES U, and that is exactly what broke `gate_aug_pin_learnu/4999` (learned U
19/65/90/90/90 deg from RRR, yaw-dominant, DCT modes 0-2 carrying 16-32% of column energy): the flow
co-opts the pinned channel for fine/high-frequency detail because pinning pays most where the flow
predicts worst, and lam=1 predictability barely resists. With U FIXED that failure mode is impossible
— coupling can only reshape the representation, never change what is pinned — so the old learned-U
result is NOT evidence against coupling.
NEW (pi0.py, env-gated, default inert):
  SNMVP_HEAD=1          enable the head with U taken from SNMVP_PIN_U (K = U.shape[1])
  SNMVP_HEAD_DETACH=1   B1: stop_gradient on the features; solves staleness, cannot regress the VLA
  SNMVP_HEAD_LAM=f      loss weight (B2 = DETACH=0 with lam swept, gated on flow loss not regressing)
Target c is the SAME oracle c the pin uses, from the ground-truth chunk; the head's PREDICTION is never
fed to the pin (that would let the flow co-adapt to head error — scheduled sampling is a separate
experiment). Readout is the existing 4-query attention pool over `prefix_out`, which is already
POST-fusion — better than the mean-pool I had assumed we would need to add.
INERTNESS: reconstructed the pre-edit file by inverting the five replacements and diffed — 36 lines,
every added path gated on _HEAD_K, and the two ungated structural edits reduce to the original when it
is 0. Smoke (30 steps) passed: head logs, loss finite.
THE STRUCTURAL PRIZE, CONFIRMED: the head params appear in the trainable tree — snmvp_q (4,256),
snmvp_k (2048->256), snmvp_v, snmvp_head_in (1024->256), snmvp_head_out (256->5), with snmvp_W absent
as intended. So the command source is written INSIDE the flow checkpoint and cannot be paired with the
wrong weights, which is the bug that cost 0/10 closed-loop at offline c-R2 0.94.
SERVING: `serve_gate_pin_joint.py` + `joint_head.py` call the model's OWN submodules rather than
re-implementing the readout, so serving cannot drift from training. One assumption: prefix_out from a
prefix-ONLY forward equals the prefix half of training's prefix+suffix forward (true because the
attention mask forbids prefix->suffix attention; the same property lang_pool relies on). It is GATED,
not assumed: `joint_head.py --check` requires per-task c-R2 > 0.5 against oracle c before flying, which
would collapse if the readout were wrong. Chain: `run_joint_b1_eval.sh` (gate -> fly 5/side at APC=50).

## 2026-08-12 — disk full killed the first B1 run; root cause was save_interval, not the head
`gate_pin_joint_b1` died at step 2000 with ENOSPC writing an intermediate checkpoint (1.9T volume at
100%). The head code was fine — the smoke passed and the run reached 2.9k steps before the write.
ROOT CAUSE: openpi's `save_interval` defaults to 1000, so a ~6 GB checkpoint was written every 1000
steps, while we only ever serve `/4999`. All gate training scripts now pass `--save-interval=5000`,
which writes only the final checkpoint (the trainer still saves at `num_train_steps-1`).
Freed with Denis's approval: `pi0_libero_low_mem_finetune` (173 GB, 13 superseded fs_pin/fs_scratch
t21 + snmvp_src runs, replaced by the pi0_libero_n2 line that today's replan sweep used) and
`pi0_libero` (146 GB, Phase-1 arms A/B/C seeds 42-44 — the 45 result JSONs in
experiments/phase1/results/ preserve the numbers; the weights are gone). 326 GB free now.
Also fixed a self-inflicted footgun: `pkill -f "run_joint_b1s.sh"` matched the command line of the
shell issuing it and killed that shell before the relaunch ran — the same class as the earlier broad
pkill that killed live DataLoader workers. Resolve PIDs with pgrep and skip $$ instead.

## 2026-08-12 — FEATFIX: matched features are NECESSARY BUT NOT SUFFICIENT
Rebuilt the language cache with the SERVED checkpoint's VLM (gate_pin_zeropad/4999), retrained the
prior on it under the same basis with zero-padded targets, flew it at APC=50, 5 trials/side.
Offline it is as good as anything we have: held c-R2 +0.9491 (early 0.8923 / transit 0.9675 / tail
0.9690). Closed-loop:
  left   3/5 transit, 1/5 full success, 0/5 clearance-clean (grazes 0.034-0.116 m at [1.13, 0.44, 1.48])
  right  0/5 transit, clearance 0.098-0.411 m
Compare, all at APC=50 with the enumeration-free command source:
  mismatched features (langprior_zeropad on gate_pin_zeropad)      left 0/5 transit   right 0/5
  matched by SERVING the cache's own flow (langprior_rrr on
      gate_both_pin_rrr)                                           left 5/5 CLEAN     right 0/5
  matched by REBUILDING the cache for the served flow (this)       left 3/5 transit   right 0/5
So fixing the feature source recovers a large part of the loss (0/5 -> 3/5 transit) but does not
reproduce the 5/5-clean result, and the two matched arms differ by FLOW (gate_both_pin_rrr vs
gate_pin_zeropad) as well as by padding convention. Note the zeropad flow is NOT the problem in
general: with the one-hot scaffold it is 10/10 clean on both gates. So this is a command-side
quality difference that survives correct pairing, and offline c-R2 again fails to predict it
(0.9491 with 1/5 success).
Reading: matched features are necessary, not sufficient. The remaining gap is what joint training and
the covariate-shift fix are for. 5 trials = a screen, not a claim.

## 2026-08-12 — joint B1 trained; readout GATE caught a serving bug, then verified
B1 (`gate_pin_joint_b1/4999`, fixed U, detached head, no state input) trained to 5k steps after the
save-interval fix. Its eval gate FAILED first — on MY code, not the head: `PaliGemma.llm` returns
((prefix_out, suffix_out), kv_cache) and I unpacked only the outer pair, so the "features" handed to
the head were the list [prefix_out, None] and the None blew up in flax's dtype promotion.
`gate_ctx_common.lang_pool` already had the correct double unpack; I should have copied its form
instead of writing the call fresh. Value of the gate: without it this would have flown 10 rollouts on
a garbage command and produced a "joint training doesn't work" result instead of a one-line fix.
After the fix, the readout verifies against the oracle c the head was trained on (held-out demo
frames, 6 episodes per task):
  center_from_left  +0.9670    center_from_right +0.9508    left +0.9367    right +0.8638
So the IN-LOOP head learns the command about as well as our best post-hoc priors (per-task 0.90-0.96)
— and by construction it cannot be mispaired, since it lives in the flow's own checkpoint. Closed-loop
eval re-running; the post-hoc control arm (same flow, prior fitted afterwards on its own features) is
queued behind it on the same GPU.

## 2026-08-12 — JOINT B1 CLOSED-LOOP: the RIGHT gate works with a grounded command source, first time
`gate_pin_joint_b1/4999` served from its OWN in-checkpoint head (no external prior file), APC=50,
5 trials/side. Readout gate passed (min per-task c-R2 +0.8501).
  LEFT   judge 3/5, clearance-clean 1/5  -> STRICT (both) 0/5   grazes at 0.003-0.175 m
  RIGHT  judge 3/5, clearance-clean 5/5  -> STRICT (both) 3/5   trials 1, 2, 5
Every enumeration-free arm at APC=50, side by side:
                                                     left            right
  mismatched features (langprior_zeropad)            0/5 transit     0/5
  matched by serving the cache's flow (langprior_rrr) 5/5 STRICT     0/5
  matched by rebuilding the cache (featfix)          1/5 judge,0 clean  0/5
  JOINT B1 (head in the checkpoint)                  0/5 strict      3/5 STRICT
This is the FIRST grounded result on the right gate at all — every previous arm was 0/5 there, and the
right gate is the one the record board has carried as 0/10 with a ~1 m +x aim bias. It is also the first
arm to beat the left/right asymmetry in the opposite direction: B1 is better on the right than the left,
where the langprior arms were the reverse. Left failures are grazes, not misses (crossings at z~1.26-1.35
vs the demos' ~1.5 — the same low-crossing signature the matched-feature right gate showed), so the left
gap is an AIM error, not route selection.
5 trials = a screen. Two things follow directly: 10 trials on B1-right for claim tier, and B1s (head +
STATE input) whose eval is now running on GPU1 — the low-crossing signature is exactly what
proprioception should help with.

## 2026-08-12 — B1s (head + STATE input): NEGATIVE, proprioception is not the missing ingredient
`gate_pin_joint_b1s/4999`, head input = attention-pooled prefix PLUS the 32-d normalized state
(head_in 1056x256 vs B1's 1024x256, verified in the param dump). Readout gate passed (min per-task
c-R2 +0.8632). APC=50, 5 trials/side, same client config as B1:
              judge   clearance-clean   STRICT
  B1  left    3/5     1/5               0/5
  B1  right   3/5     5/5               3/5
  B1s left    2/5     0/5               0/5
  B1s right   3/5     2/5               0/5
So adding the state does NOT help and costs clearance on the right (5/5 -> 2/5 clean), turning 3
strict successes into 0. Offline it is a wash too: per-task c-R2 0.863-0.932 vs B1's 0.850-0.967.
READING: the head's problem is not lack of proprioception. That is consistent with the earlier
within-task ablation (state-only reached per-task 0.775 while state+lang and state+onehot both reached
~0.91) — the post-fusion features already carry the drone's position well enough, so the state adds
little and the extra 32 inputs apparently cost a little. My hypothesis that the low crossings (z~1.3 vs
demos' ~1.5) came from missing altitude information is REFUTED: B1s knows z explicitly and still
crosses low. The aim error is therefore in the command CONTENT, not in the head's access to state.
5 trials per cell; B1's claim-tier run (10 trials/side) is now on GPU1.

## 2026-08-12 — JOINT B1 AT CLAIM TIER: right 7/10 strict; and ROLLOUTS ARE NOT REPRODUCIBLE
10 trials/side, same checkpoint/server/client as the 5-trial screen:
  LEFT   judge 10/10, clearance-clean 2/10 -> STRICT 2/10
         clearances 0.134 0.142 0.149 0.152 0.154 0.154 0.161 0.172 0.189 0.278
  RIGHT  judge 10/10, clearance-clean 7/10 -> STRICT 7/10
         clearances 0.092 0.173 0.174 0.246 0.263 0.290 0.303 0.311 0.336 0.373
So the enumeration-free command source, living inside the flow's own checkpoint, gets 7/10 STRICT on
the RIGHT gate — the gate the record board carries as 0/10 with a ~1 m aim bias, and which every other
grounded arm missed 5/5. Left is 10/10 on the judge but grazes: the clearances CLUSTER at 0.134-0.172
against the 0.18 m threshold, i.e. a systematic ~4 cm shortfall rather than scatter, so a small aim
correction should convert most of them.
METHOD FINDING, and it changes how the earlier screens should be read: THE ROLLOUTS ARE NOT
REPRODUCIBLE RUN TO RUN. The 5-trial screen of this identical config gave judge 3/5 on both sides; the
10-trial run gives 10/10. I diffed the two eval scripts — only names and ports differ — and compared
trajectories directly: same config, same server seed, trials 1-3 differ by up to 0.63 m (left 1: 0.633,
right 3: 0.527). Cause is almost certainly float nondeterminism (GPU reduction order, gsplat atomics)
amplified over ~400 closed-loop steps. Consequences:
 - a 5-trial cell is much noisier than the +-5-6 pt protocol noise we had assumed, and 3/5 vs 10/10 on
   the same config is within its range; do not compare 5-trial cells across arms at all.
 - the two-tier rule (>=10 for claims) stands for a sharper reason than training-seed variance: the
   ROLLOUT itself is nondeterministic.
 - B1s's negative result rests on 5 trials and should be re-run at 10 before it is treated as settled;
   the same applies to featfix's 1/5 and featmatch's right-gate 0/5.
Combined over both runs (15 trials/side): left judge 13/15, clean 3/15; right judge 13/15, clean 12/15.

## 2026-08-12 — B2 COUPLED and the post-hoc control; plus a self-inflicted repeat of the skew bug
RESULTS (all APC=50, strict = judge AND clearance):
  B2 coupled, head loss backprops into the VLM, 5 trials   LEFT 5/5 STRICT   right 0/5
  post-hoc prior on B1's OWN features, 5 trials            LEFT 5/5 STRICT   right 1/5 judge, 0 clean
  joint B1 in-checkpoint head, 10 trials                   left 2/10 strict  RIGHT 7/10 STRICT
  B1s (+state), 10 trials                                  left 0/10         right 1/10
B1s's negative result now HOLDS at claim tier: 0/10 left and 1/10 right vs B1's 2/10 and 7/10. The
state input hurts.
A CONSISTENT PATTERN across constructions, not noise: EXTERNAL post-hoc priors are good on the LEFT
and fail on the RIGHT (featmatch 5/5 L, posthoc_b1 5/5 L, both ~0 R), while the IN-LOOP head is the
reverse (B1 7/10 R, 2/10 L). Two ways of building the same command source have OPPOSITE gate biases.
That is worth understanding before optimising either — it suggests the left and right routes need
different things from c, and each construction happens to supply one of them.
B2's total training loss is 0.079 vs B1's 0.246, but that is NOT evidence the flow improved: both
losses include the head term, and in B2 the coupling lets the VLM reduce that term directly. I said
this arm would be gated on "flow loss not regressing" and the log only records the SUM, so the gate as
written cannot be evaluated. Needs the two terms logged separately before B2 can be compared on flow
quality.
BUG (mine, third instance of the same class today): `langprior_pipeline.py` wrote its prior to a
HARDCODED `langprior_rrr.pt`. Every cache re-extraction therefore silently replaced the prior other
experiments were serving. featfix's extraction (01:59) and posthoc_b1's (08:43) each overwrote it, so
the 10-trial "rerun" of the matched-feature arm actually served gate_both_pin_rrr with a prior fitted on
B1's features — a feature-source mismatch, which is precisely why it scored 0/10 judge where the
original scored 5/5 strict. That run is INVALID (renamed INVALID_ev_fm10b_scores.txt) and the original
langprior_rrr.pt is LOST; redoing it needs a fresh extraction under FEAT_CKPT=gate_both_pin_rrr into a
distinct filename. The featmatch 5/5 result itself stands — it ran before the overwrite.
FIX: the prior path is now `PRIOR_OUT` (env), the chains pass a per-cache filename, and the docstring
records why. Same lesson as the hardcoded FEAT_CKPT: any artifact with a default shared path will
eventually be paired with the wrong feature source.

## 2026-08-12 — CLAIM TIER ACROSS ARMS: B2 gets LEFT 10/10 STRICT; B1 and B2 are COMPLEMENTARY
All 10 trials/side, APC=50, strict = transit judge AND gate clearance >= 0.18 m:
  command source                                       left strict   right strict
  one-hot scaffold (enumerates tasks -- not the goal)      10/10         10/10
  B2 coupled, head loss backprops into the VLM            10/10          0/10
  B1 detached head, both in the flow's checkpoint           2/10          7/10
  B1s detached + 32-d state input                           0/10          1/10
  matched-feature external prior (rebuilt properly)         5/10          0/10
  featfix external prior (cache rebuilt for served flow)     0/10          0/10
B2's left clearances are 0.21-0.38 m -- not marginal, comfortably clear of the 0.18 m body radius, and
it is 10/10 on the judge too. That is the FIRST enumeration-free command source to match the one-hot
scaffold on a gate.
THE STRUCTURE OF THE RESULT: B1 (detached) and B2 (coupled) are COMPLEMENTARY -- B2 owns the left gate,
B1 owns the right, and neither does both. The same split appeared in the external-prior arms (good left,
zero right), so "external vs in-loop" was the wrong axis; the real axis is COUPLING. Letting the head's
loss reshape the VLM buys the left route and loses the right one. Both directions are now at claim tier,
so this is not sampling noise.
Also settled at 10 trials: featfix is 0/10 both sides (its 1/5 screen was optimistic, not pessimistic);
the properly-redone matched-feature arm is left 5/10, right 0/10 -- so the original featmatch 5/5 was a
lucky draw of a ~50% left rate, and its right-gate 0 is real.
NEXT: the open problem is no longer "can a grounded command source fly" -- it is "can ONE model fly both
gates". Launched b2long (coupling + 15k steps + annealed LR, 10 trials) as the cheapest shot: it tests
whether more optimisation lets the coupled arm recover the right gate it currently trades away.

## 2026-08-12 — MORE STEPS DOES NOT HELP: b1long (15k, annealed LR) loses the right gate
Same recipe as B1 (detached head, fixed U) but 15k steps with lr_schedule.decay_steps=15000 so the LR
actually anneals over the run (the stock 1,000,000 keeps a 5k run at essentially peak throughout).
  training loss   B1 5k = 0.246   ->   b1long 15k annealed = 0.197
  left   judge 5/5, clean 1/5 -> STRICT 1/5   (clearances 0.09 0.10 0.13 0.17 0.19)
  right  judge 0/5, clean 5/5 -> STRICT 0/5   (clearances 0.29-0.33, i.e. it flies PAST cleanly)
So 3x the optimisation and a properly annealed schedule LOWERED the loss and DESTROYED the right gate
(B1: 7/10 strict; b1long: 0/5, and not by grazing — it stops transiting at all). Left is unchanged
within noise. The two changes (steps, schedule) are confounded by design, but neither ordering of that
confound rescues the arm.
This is the optimisation/closed-loop disconnect again, now on the training axis rather than the metric
axis: loss down, behaviour worse. Caveat: 5 trials, so a screen — but 7/10 transits going to 0/5
transits is a qualitative change, not a rate shift.
NEXT EXPERIMENT, chosen from the structure of the results rather than from a hunch: B1 (detached, lam
irrelevant) owns the RIGHT gate and B2 (full coupling, lam=1) owns the LEFT, and the distinguishing axis
is COUPLING STRENGTH. So sweep it — launched b2lam03 (DETACH=0, lam=0.3, 5k steps, 10 trials): partial
coupling is the natural candidate for keeping the left gate while recovering the right.

## RRR-from-VLA claim CONFIRMED for the PREDICTOR, WASH for the basis; C1/C2 factored arms built (2026-08-12)
Denis set the destination architecture: a VLA predicts c, a separate action head denoises given c;
the VLM trains ONLY through the c loss, the action head only through the flow loss; U comes from RRR
with the VLA's features as predictor. Pre-registered confirmation of the underlying 2026-08-01 claim
(confirm_vlm_rrr.py, CPU, cached vlm_feat_context.npz, row alignment re-verified 3483==3483): clean
2x2 {basis: RRR(vla-feat) vs PCA(chunks)} x {prior: MLP(vlm-ctx) vs MLP(state+onehot)} on in-dist + 3
disjoint task-heldout splits of data_libero_multi. RESULTS (held c-R^2 all-suite):
  in-dist: all four cells 0.818-0.820 (reproduces original 0.823/0.808; nothing distinguishes them).
  held 18,19,28,29: VLM prior +0.425/+0.383 (RRR/PCA basis) vs onehot -3.47/-3.63  <- original split
  held 12,13,22,23: VLM prior +0.189/+0.304 vs onehot -3.82/-3.73
  held 15,16,25,26: VLM prior +0.313/+0.235 vs onehot -3.80/-3.41
VERDICT: the WIN is the VLM-feature PREDICTOR (+0.2..+0.4 vs -3.4..-3.8 on unseen tasks, all three
splits — stronger and more robust than the original single-split claim). The BASIS is a WASH here:
RRR-vs-PCA differences are small and change sign across splits (capture nearly identical, 0.38-0.52),
consistent with the RRR~=PCA geometry (chunks ~91% predictable). Unseen LANGUAGE goals remain the
frontier (goal-suite heldout +0.09/-0.29/-0.29). U_vla is the principled choice but not the lever on
this data; the routing is the experiment.
BUILT AND VERIFIED (test_flow_detach.py, dummy pi0, float32):
  C2 = SNMVP_FLOW_DETACH=1 (openpi pi0.py): training runs the same two-pass computation as inference
  (live prefix pass -> head loss reaches the VLM through prefix_out; stop_gradient(kv_cache) -> the
  flow loss cannot). Forward EQUALS the joint pass (rel 2.3e-7 f32; the bf16 run differs by one ulp
  which amplifies — equivalence checks on this model must be f32+relative). Flow loss alone: VLM grad
  exactly 0.0 on all 32 tensors, expert grad nonzero and bit-identical with/without head loss. Guard:
  FLOW_DETACH without a coupled head raises (nothing would train the VLM).
  C1 = config pi0_gate_freezevlm (freeze_filter .*(llm|img).* minus llm *_1, minus lora): 32 frozen /
  27 trainable on dummy, zero leak either way. Head detached = pure readout on pi0_base features;
  basis-feature pairing fixed at init cannot drift.
ROUTING 2x2 now complete as an experiment family: B1 flow-only->VLM (owns right), B2 both (owns
left), C2 c-only (Denis's factored proposal), C1 neither (frozen control).
QUEUED (run_c2_chain.sh GPU0 behind b2lam03; run_c1_chain.sh GPU1 behind b2long + waits for U_vla):
Phase 0 extracts pi0_base post-fusion features (langprior_pipeline FEAT_CKPT=pi0_base, distinct
filenames) -> refit_rrr_basis --out pin_U_vla_base_k5.npy (angles vs deployed U + PCA logged to
c2_basis_audit.log) -> C2 5k/decay1e6/10 trials, then C1 same on GPU1. Oracle c = U^T a stays the
pin target everywhere; the head prediction is never fed to the pin in training. run_joint_arm2.sh
adds UPATH + TRAIN_CFG params (run_joint_arm.sh was mid-execution — new file by rule). wandb: no
credentials on the box; left disabled per Denis (flip on after `wandb login` if wanted).

**b2lam03 (lam=0.3 coupling) LANDS: first enumeration-free source to transit BOTH gates 10/10 clean
(2026-08-12).** 10 trials/side, readout gate min per-task c-R2 +0.90. LEFT 10/10 STRICT (transit+goal+
clearance). RIGHT: 10/10 transit AND 10/10 clearance-clean (0.31-0.38 m — demo-level, vs lam=1's 0/10
transit) but only 3/10 strict: the 7 failures fly through the gate and OVERSHOOT the hover box
(goal x<=1.825; failures end x 1.88-2.44, +0.06..0.62 m past) — the known no-stop endgame flaw, not
aim. So the coupling dial trades cleanly: lam=1 owns left/kills right route; lam=0.3 keeps left strict
AND recovers the right route completely; the residual moved from aim to endgame. Failure STAGE changed,
which is the structural read. Videos overlay_armb2lam03_* pending Denis review before any record-board
claim. Meanwhile run_c2_chain 1st attempt EXTRACT_FAILED: pi0_gate config is LoRA, raw pi0_base lacks
lora_a/b and create_trained_policy checks structure strictly — fixed with pi0_gate_full (non-LoRA twin,
matches pi0_base exactly = the arms' init features since LoRA is ~zero at init) + PIPE_CONFIG env in
langprior_pipeline; relaunched as run_c2_chain2.sh. Tracker artifact updated (430ab907-7c05: running/
queue/Theme-4 routing card).

**lam=0.3 CENTER + COMPOUND (2026-08-12).** Center (trained prompts, in-dist, 10 trials each): CFL
0/10 transit (5/10 clean), CFR 0/10 (10/10 clean) — every flight approaches the center gate to
0.11-0.43 m and never crosses; one-hot reference rescored properly (old ctrcomp run had a broken
--side arg): CFL 10/10 judge / 9 strict, CFR 9/10 / 6 strict. So the lam=0.3 coupled head owns
LEFT+RIGHT routes but NOT center — third representation trade-off on this axis. Compound (novel
conjoined prompts, never trained, 5-trial screen): left-compound gets FIRST gate 5/5 and BOTH gates
IN ORDER 2/5 (dwell 0 -> 0/5 strict) — real zero-shot composition signal from one static sentence, no
milestone switching (the langprior reference got 0/5, zero gates); right-compound selects nothing 0/5.
Explorer updated with center + both compound scenes + goal boxes (artifact 2c0f3000).
**CLE THROUGHPUT (Denis approved all suggestions).** Measured: ~75% of rollout wall time was mp4-only
rendering (25 extra 512x384 renders/chunk), ~15% per-trial process startup, ~10% inference; the policy
consumes only 2 renders per chunk. New rung3/gate_rollout_batch.py: TRIALS looped in-process (scene
loads once), VIDEO=0 traj-only screen mode, VIDEO=1 default now VIDFRAME_STRIDE=4 fps=9 (Denis ok with
low-fps review; same full-length playback), strut-overlay load now HARD-FAILS if the asset is missing
(was a silent except->None that would strip the strut from the wrist obs). OBS PATH BYTE-IDENTICAL
(fwd Tbc_f + downward wrist Tbc_d, 1024x768->256->224, overlay after squash — verified asset present,
RGBA 256x256). Smoke-tested both modes on the b2lam03 checkpoint (screen: 2 trials no frames, video:
26 frames = 2 chunks stride 4, same verdict line). run_joint_arm2.sh + run_eval10.sh switched to the
batch client with left/right sides in PARALLEL against one server. Expected: screens ~6-8x, claim
batteries ~2.5-3x. gate_video_overlay.py untouched (was executing; also still owns steer_gui import).

**Viewer scene clouds were HIDING the gates (2026-08-12).** extract_scene_cloud's uniform voxel thin
reduced the thin gate structure to ~47 points in the center/compound clouds (left cloud had 1713 in
the gate region; the edited scenes were also exported smaller) — so the center viewers showed flights
aiming at nothing. Fix: gate-aware decimation (gate Gaussians from the scene-edit mask kept at full
density up to 3000 pts, background voxel-thinned separately); all five clouds re-exported; explorer
viewers now also draw the judge's TRUE aperture rectangles (miss_gate.corners, cyan) alongside the
goal box. Same bug family as the region-box scorer bug: a uniform instrument silently erasing exactly
the structure under study.

**COMPOUND-RIGHT WAS DOUBLY BROKEN; RESULT RETRACTED AND REVERSED (2026-08-12).** Denis noticed the
right_and_center viewer gate looked wrong; pulling the thread found TWO bugs. (1) SCENE-SELECTION:
right_and_center is defined on the RIGHT splat (its YAML gsplat_path) but gate_video_overlay/
extract_scene_cloud selected the base checkpoint with `SCENE in ("right",)` — false for
"right_and_center" — so every compound-right rollout rendered the LEFT scene: the drone was told
"through the gate on the right, then the center gate" while looking at a scene with NEITHER. The l03
and a50 compound-right runs are INVALID (quarantined ctxrun/invalid/); my "the novel right wording
selects nothing / left-bias" conclusion is RETRACTED. (2) REGION-BOX (again): right_and_center's
safety-YAML gate_1 corners spanned a 0.39 m segment matching no physical opening (left_and_center
follows the convention gate_1 = parent aperture; right did not), so even valid right-gate crossings
scored gates=0. Both fixed (clients: SCENE in ("right","right_and_center"); YAML corners =
right_gate.yaml's validated aperture). REFLY (b2lam03, corrected scene+judge, 5 trials, first outing
of gate_rollout_batch.py): first gate 5/5 (crossings dead-center in the aperture), center copy 0/5
(fly past it, ends x 3.6-4.5) -> compound-right = SAME shape as compound-left's residual: novel prompt
composes the FIRST hop, second-hop center is the weak task, matching center-standalone 0/20.
Explorer split into separate CFL/CFR center viewers (Denis) and republished with the corrected data +
invalidation notice. Lesson repeated: every scorer AND every renderer scene-selection needs a
calibration check against its parent task before first use.

**SCRATCH CONTROL ON CENTER+COMPOUND; APC=25 PROBE (2026-08-12, VIDEO=0 batch battery, ~40 min for
50 rollouts).** Scratch (gate_both_scratch, no pin): CFL 10/10 judge / 1/10 clean, CFR 10/10 / 0/10 —
the scratch signature (reaches and transits, hits the structure) reproduces on a third scene, so
lam=0.3's 0/20 center transit is a COMMAND-SIDE deficit of the coupled head, not scene difficulty.
Compound: scr left 0/5 (contacts 0.003-0.134 m); scr right 2/5 pass the ordered judge INCLUDING dwell
but BOTH graze the center copy (0.010/0.035 m) -> 0/5 strict — the compound judge without clearance
overstates scratch. gate_clearance had the SAME right_and_center left-splat bug (4th sighting of the
scene/region bug class today); fixed, and compound rows now carry clearance columns. l03 cmp_right
refly: 5/5 clean through gate 1 (0.26-0.39 m). APC=25 (b2lam03, NCH=20, matched step count): CFL 0/10
(2/10 clean), CFR 0/10 (10/10 clean) — same as APC=50, so the center miss is NOT a replanning-rate
problem; the head commands a consistently short/offset center route at every planning frequency.
Explorer updated with scratch + APC=25 groups (2c0f3000).

**GRID PAGE + SCRATCH L/R BASELINE (2026-08-12).** Explorer rebuilt as a synced 6-panel grid (Denis:
3 arms only — one-hot / lam03 / scratch; ONE global legend toggling all panels; middle-drag pan;
compact panels; goal box + judge apertures as toggles). Old per-viewer legends had a latent
cross-talk bug (every viewer bound every legend by index) — new design has one intentional handler.
Scratch (gate_both_scratch) on plain L/R under today's protocol: LEFT 9/10 judge AND clean (8/10
strict), RIGHT 0/10 — so today's full grid: left gate is easy for every source; right gate separates
them (one-hot 10, lam03 3 strict/10 clean-transit, scratch 0); center separates differently (scratch
transits dirty, lam03 clean-but-short, one-hot does both). NOTE the historical "scratch transits
20/20 completes 0/20" line does NOT describe this checkpoint under the current scorer — scratch is a
strong LEFT performer now; the record-board control line needs re-verification before reuse.
Training-smoothness check (Denis): b2lam03 loss 0.661->0.045 smooth, grad_norm 0.33-0.42 stable, no
spikes; b2long's train log contains NO loss lines at all (tqdm progress only, pbar.write lines
missing — logging quirk, worth folding into the separate-loss-terms logging fix); NO val loss exists
anywhere (openpi train.py has no validation pass; the readout gate + closed-loop are our only
held-out signals). Queued: sim-vs-real c-consistency probe at matched states (data_gate_real vs
synth through the b2lam03 head).

**PHASE 0 BASIS AUDIT LANDS (2026-08-12 22:24).** U_vla (RRR on pi0_base post-fusion features, the
init all joint arms start from): deployed pin_U_gate_rrr_k5 within [0.4,0.8,0.9,1.6,7.0]deg (mean
2.1) — four directions essentially identical, the 5th rotates 7deg; PCA max 16.4deg; held-out chunk
variance 0.8222 vs deployed 0.8235 vs PCA 0.8242. As predicted by the LIBERO confirm: the basis is
NOT the lever on this data — C2 trains on U_vla anyway (principled choice, comparability preserved
by the tiny angles). C2 training running (1.5k/5k, ~1h55m left). Sim-real c-probe
(sim_real_c_probe.py, direction-aware matched frames real<->synth through the b2lam03 head) chained
behind arm_c2.done on GPU0.

**TAIL-C DESIGN DISCUSSION -> TWO GENERALIZING FIXES CHOSEN OVER TAIL-WEIGHTING (Denis, 2026-08-12).**
Denis rejected tail-weighting as regime-specific hand-tuning; root cause named: c CONFLATES route
content with motion magnitude (fixed-window displacement sum -> "stop" is only expressible as
smallness, and variance-normalized MSE can't see errors there while the pin's 1:1 passthrough makes
them maximally expensive). Constraint: any basis fix must keep c LINEAR in the chunk (exact pin).
Chosen directions: (A) MULTI-HORIZON BASIS — U over dyadic-window displacements (6/12/25/50 steps),
linear, pinnable, RRR-compatible; "stop" becomes a readable PATTERN (short-horizon == long-horizon)
instead of a small number; slow regimes auto-scale, no reweighting. Offline audit queued before any
flow train. (B) SCALE-INVARIANT HEAD LOSS (SNMVP_HEAD_LOGMAG=1, implemented + dummy-verified finite
grads incl. zero-command batch): magnitude-weighted cosine direction term + squared log-magnitude
(eps floor 0.1) — relative error costs equal at 5 cm and 1 m; PIN UNTOUCHED (oracle c still pinned
in training; head output recomposed at serve). Arm b2logmag queued (b2lam03 recipe, ONLY loss
changed, six-cell eval via run_joint_arm3.sh) behind tonight's GPU1 chain — watch training
stability per Denis. (C) speed-referenced magnitude and (D) uncertainty-scaled trust discussed and
held in reserve; APC=25 right-gate probe (tonight) will size the pure execution-amplification share.

**C2 (FACTORED ROUTING) FIRST CLOSED-LOOP: OFFLINE-EXCELLENT, LEFT GATE 0/10 (2026-08-13 00:30).**
Readout gate min per-task c-R2 +0.938 (better than lam03's 0.90), yet LEFT 0/10 transit: flights
head toward the goal region directly (ends ~(1.5-1.7, -0.9, 1.3)) without threading the aperture —
route content missing from the executed command despite on-demo c accuracy. The chasm instrument
strikes the in-checkpoint head the moment the VLM stops receiving flow gradients: features shaped
ONLY by c-prediction seem to lose what the closed loop needs off-manifold (and/or the expert's
perceptual inputs degrade — C1 frozen control, ~1h out, separates these: if C1 flies better than C2,
c-only VLM gradients actively HARM; if both fail, flow gradients into the VLM are necessary).
RIGHT side data missing entirely: the parallel batch clients raced — first client's handshake+JAX
compile blocks the server loop, second handshake times out (websockets TimeoutError). Left data
valid. Re-fly launched (c2_right_refly); run_joint_arm2b.sh staggers clients 120 s and
run_joint_arm3.sh now routes through it (arm2 untouched — C1's bash has it open). C1's own right
side may hit the same race — will re-fly if so.

**C2 RIGHT GATE 10/10 STRICT — FIRST FULL RIGHT-GATE COMPLETION BY ANY ENUMERATION-FREE ARM
(2026-08-13 01:30, re-fly after the handshake race, pending video review).** 10/10 success (transit
AND goal) + 10/10 clearance-clean: the factored routing (VLM trained by c loss only) solves the
ENDGAME that defeats every coupled arm at every lam and step count — and simultaneously fails LEFT
0/10 (flights skip the aperture and head for the goal region directly; suspicious of route collapse
toward the right-task shape — CLOG/video forensics queued). C2 = mirror of B2. The routing 2x2 row
is now: B1 right 7/10-ish, B2 left-only, C2 right-only-but-COMPLETE, C1 readout-gated. C1: frozen
pi0_base features FAIL the readout gate on LEFT only (+0.47; CFL 0.85, CFR 0.73, right 0.75) — VLM
adaptation matters most for the left task's readout; center add-on continues (those readouts pass).
SIM-REAL C PROBE (Denis's question) ANSWERED with decomposition: at matched real/sim states the
head's predictions differ cos 0.52-0.62, gap 1.9-2.2 std — NOT similar. But the ORACLES diverge
MORE (cos 0.46-0.54, 2.5-2.6 std): real and synth DEMOS genuinely behave differently at the same
spot; the head is domain-faithful (R2 0.86 real / 0.74-0.76 sim vs own oracle; cross-domain R2
~0) and even smooths the demo gap slightly. The sim-real command gap is dominated by BEHAVIOR
difference in the data, not by the head misreading real pixels. Grid updated with C2 row
(2c0f3000); table row bug fixed (ARMS[:3] hardcode dropped rows silently).

**OVERNIGHT BATCH LANDS (2026-08-13 05:45).** (1) APC=25 RIGHT PROBE DECISIVE: b2lam03 right at
APC=25/NCH=16 -> transit 10/10, goal 0/10, clean 10/10 — the overshoot SURVIVES double replan rate;
endgame failure is NOT open-loop commitment, the head commands past the goal at every planning
frequency. Tail-c content is the culprit; execution-side fixes ruled out. (2) C1 center add-on:
CFL/CFR 0/10+0/10 (2/4 clean) — frozen features fail center closed-loop despite offline readout
0.85/0.73 there (chasm again). (3) b2long center add-on: 0/20 transit (clean approaches), compounds
0 — full coupling at 15k also has no center route. (4) mh16 FULL ROW — THE ROUTE GENERALIST: 
transits EVERYTHING (L 10/10, R 10/10, CFL 10/10, CFR 9/10 — FIRST learned arm to route center,
which every other arm fails 0/20) + best composition yet (cmpL 3/5 both gates, cmpR 1/5) BUT
finishes little: strict L 0/10 (goal misses), R 4/10 (7/10 success, 4/10 clean — grazes), center
0/20 strict, readout 0.894. The multi-horizon code made the center route EXPRESSIBLE as
hypothesized; the endgame/hover and clearance precision did not come along at lam=0.3/5k. 
(5) b2logmag: TRAINING STABLE (smooth 0.66->0.041, grad_norm 0.28-0.42), readout gate PASS at
0.926 > MSE-twin's 0.90 — the scale-invariant loss costs nothing offline; closed-loop eval in
flight. C2-right human-review page published (447bd6f4, 10 re-encoded full-length videos).
Grid updated with mh16 row (2c0f3000).

**C2-LEFT FORENSIC: ONE COMMAND DIMENSION COLLAPSES AT THE AMBIGUOUS START; b2logmag NEGATIVE
CLOSED-LOOP; COMPOUND PROMPTS CORRECTED (2026-08-13 morning).** CLOG forensic on C2's left flights:
the start-position command matches the LEFT oracle at cos 0.81 on 4 of 5 dims (task selection is
CORRECT, route collapse REFUTED) but component c2 outputs -0.21 where LEFT demands +2.88 and RIGHT
-4.69 — c2 IS the route-discriminating dim, and at the start state (identical across tasks; only
language separates them) the c-only-trained head regresses it toward the task midpoint,
deterministically (spread 0.00). Ties three facts together: C2 right works (its c2 binding holds),
C2 left fails route (c2 mid-collapse), C1 frozen features fail readout on LEFT specifically — base
features encode the right task better; LEFT's language->c2 binding at ambiguous states is what
flow-gradient coupling (B-arms) provides and c-loss-only training does not. b2logmag CLOSED-LOOP
NEGATIVE at 10 trials: left 6/10 success 6/10 clean (MSE twin 10/10 strict), right 2/10, clearance
1/10 clean — stable training + better offline R2 did not transfer; the ||c||-weighted direction term
likely under-trains direction precision mid-flight (grazes). COMPOUND PROMPTS: corrected to carry the
second hop's direction ("...then go through the center gate from the left/right...") matching CFL/CFR
vocabulary + judge dy-signs; re-screens (b2lam03/c2/mh16, 5 trials/scene) running as cmpfix on GPU0.
b2logmag row added to grid.

**MH16 "GOES CRAZY AT THE END" — FORENSIC (Denis, 2026-08-13).** CLOG on mh16's left flights: the
h6-band command magnitude NEVER decays (mean 2.0-2.8 normalized units across all 8 chunks — cruise
level even at the goal, where the demo is nearly stationary), the long band stays 3.3-5.0, and late
chunks show SIGN FLIPS between consecutive replans (h6:x -0.26 -> +0.85). Diagnosis, two compounding
causes: (a) the head's tail failure is UNCHANGED and is a PHASE/PERCEPTION problem — off-manifold
near the goal the features never say "you're at the end", so commands stay cruise-like (the audit's
tail R2 was measured ON-manifold, where h6 is predictable — closed loop leaves that set); (b) the
multi-horizon basis AMPLIFIES the consequence: hard-pinned short-horizon components convert "wrong
magnitude" into VIOLENT IMMEDIATE MOTION executed 1:1 within 6 steps, and chunk-to-chunk sign flips
into thrash. Flat basis fails gracefully (smooth overshoot); mh16 fails energetically. The basis
delivered ROUTE expressiveness (center transits, best composition) but cannot fix a command source
that doesn't know the phase — and it raises the price of that ignorance. Fix candidates: per-band
soft pin (sigma on h6/h12 so the flow — which SEES the goal below — can veto short-horizon nonsense),
phase inputs from state history, or conceding the stop to the denoiser (option D).
b2logmag30k queued (~17 h GPU1, Denis's ask): does the scale-invariant loss recover with 6x
optimization + annealing, as lam=1 coupling did between 5k and 15k?

**BIMODALITY AUDIT: THESIS NOT CONFIRMED AS PRE-REGISTERED (2026-08-13).** MLP head on the base
feature cache, held-out residuals by segment: NO segment/dim passes the pre-registered bar
(dBIC<-10 AND sep>1.2sigma AND w>0.15). Tails mostly prefer ONE component (dBIC +1.6..+10.9; only
c4 at -26.9 with sep 0.61 — weak); start/mid dBIC favors 2 components but separations 0.1-0.9 sigma
= heavy tails, not modes. HONEST READ + instrument limitation (my design error): the audit tests
residuals ON-MANIFOLD, where demo futures are nearly deterministic given the frame — the thesis
lives (a) at OFF-MANIFOLD closed-loop states (not in any cache) and (b) in the failing checkpoint's
OWN feature space (C2's c2=task-midpoint at start was measured on C2 features; this cache's
features separate prompts fine and correspondingly show no start bimodality). REFINED THESIS:
mode-averaging occurs where the head's FEATURES fail to disambiguate the future — task-at-start for
C2, phase-off-manifold for the endgame. Corollary that tempers the generative-head claim: sampling
fixes INVALID AVERAGES (no creep/thrash — always flies a valid mode) but not MISCALIBRATED
POSTERIORS — features that can't see phase will sample "cruise" with whatever weight the learned
posterior gives it off-manifold. The generative head is still the right structure; feature/phase
observability remains a separate axis. Next instrument: manifold-distance of closed-loop tail
states (clog_analysis style) to size the extrapolation share directly.

**MANIFOLD-DISTANCE TAIL PROBE (Denis-approved instrument, 2026-08-13): the endgame story changes
again — the dominant term is a MISSING RESTORING FIELD, not cruise-blindness.** Per-replan matching
of closed-loop states to demo frames (direction-aware) with the arm's own oracle at the match:
- b2lam03-LEFT (the success control): flights deviate up to 0.53-0.56 m from the tube MID-flight and
  COME BACK (d: 0.01 -> 0.56 -> 0.09 by the last chunk); command error small on-tube (0.32-0.35),
  rising with d then correcting. The left command field is CONTRACTIVE — that is what success looks
  like under this instrument.
- b2lam03-RIGHT (the overshoot): flights stay NEAR the tube the whole time (tail d 0.15-0.18) but
  the commands are wrong ON-manifold — err 0.68/std in the d<0.1 bin, 2x left's 0.34 — and
  systematically UNDERSIZED (|c|/|c*| 0.34-0.72). The failure is not "cruise past the goal": they
  settle PAST the box and then command ~hover (matched-demo oracle is hover; demos contain NO
  return-to-goal data), so there is no restoring command from wrong-hover states. Coverage hole at
  recovery states + on-manifold precision deficit on the right task.
- c2-LEFT: off-route from chunk 1 (d 0.21 immediately — the c2 start error), err grows with d
  (0.37 -> 0.65); once off, nothing restores it.
- mh16 (position-only matching, aliasing caveat): err elevated at ALL d on right (0.65 on-tube ->
  1.01 far), consistent with 16-dim on-manifold imprecision amplified by hard pin.
REVISED SYNTHESIS: three separable failure terms, now each measured: (1) ON-MANIFOLD per-task
command precision (right/center worse than left; feature binding); (2) NO RESTORING FIELD off the
nominal path (demos give zero supervision for recovery; the old union-tube finding was this same
term at mid-flight — coverage, not architecture); (3) mode-averaging at ambiguous conditioning
(C2-c2 start; real but narrowest). The generative head addresses (3) and the VALIDITY of commands
under (1)-noise; it does NOT create a restoring field (2) — that needs coverage (union-style
return-to-goal data derived from demos' own continuations) or conceding off-nominal correction to
the flow. Instrument now standing: manifold_tail_probe.py.

**GENERATIVE COMMAND HEAD BUILT + LAUNCHED (2026-08-13).** Denis's design settled: sample c~p(c|o)
independently at each replan, NO commitment mechanism — the state is the memory; revisability free.
Measurement that motivated it (answering "does averaging explain mh16-left's wrong magnitudes"):
demo h6-band magnitude by phase is cruise 3.28 / late 6.84 / stop 3.02 — in NORMALIZED units the
stop is a NONZERO signature vector (zero raw delta -> -mean/std offset), a fact my earlier "commands
never decay" reading missed — and mh16's goal-region command is 2.48 (p10-p90 2.03-2.94), BELOW all
three modes = the shrinkage signature of averaging different directions. Restoring-field/coverage
line explicitly DEFERRED by Denis (data problem). Implementation: SNMVP_HEAD_GEN=1 in pi0.py — the
MLP readout's regression is replaced by a K-dim conditional flow (CFM, same t=1-noise convention as
the action flow; ctx = same attention pool; loss lam*||v-(e-c)||^2; oracle c still pins as always);
serve = 10-step Euler, one draw per replan (joint_head.head_c auto-detects gen params; readout gate
uses the 8-sample mean via SNMVP_GEN_SAMPLES to test the distribution's center). Verified on dummy:
loss finite, gen params + VLM get gradient, OLD regression head gets exactly 0; sampler shape/
finite/stochastic. Arms launched: gen1 = gen head x FLAT K=5 basis (single-variable vs b2lam03,
GPU0) and gen16 = gen head x mh16 basis (combined bet, GPU1 after the seed-7 replication).
cmpfix final: corrected directional prompts give b2lam03 cmpL 5/5 both gates (2/5 before), mh16
cmpR 3/5 both gates (best yet), C2 unchanged (feature-level blindness, as the refined thesis
predicts); mh16 cmpL dipped 3/5->0/5 (screen noise or binding quirk — flagged, 5-trial tier).

**gen1 FULL ROW (2026-08-13 11:10): sampling recovers CENTER ROUTING ON THE FLAT BASIS.** Center:
CFL 8/10 TRANSIT (6/10 clean), CFR 4/10 transit — every previous flat-basis arm was 0/20 center
transit; only mh16 (basis change) had ever routed center. So center's route failure was also
substantially MODE-AVERAGING (blended commands veering short), fixed by committing draws — the
multi-horizon basis and the generative head independently recover it via different mechanisms.
Right: 5/10 strict (6 judge / 10 transit / 8 clean) — ~2x the MSE twin. Left: 0/10 (the 50/50
posterior coin-flip; binding is the isolated bottleneck). Compounds ~0 (flat basis, one 1/5 both-
gates). Grid updated. Next lever on GPU0: gen1lam1 (lam=1) — does stronger coupling calibrate the
posterior's mode weights (B2's flow-gradient binding, now measurable directly as the start-draw
histogram); gen16 still queued on GPU1 behind the seed-7 replication (5k reached, eval running).

**SEED-7 REPLICATION: TRAINING-SEED VARIANCE IS LARGE (2026-08-13 14:00).** b2lam03's exact recipe,
only --seed=7: LEFT 5/10 success (10/10 clean; seed-42 was 10/10 strict), RIGHT 1/10 success
(seed-42: 3/10 strict, 10/10 transit). Training-seed noise on strict cells is ~+/-5 points at 10
rollouts — the two-tier statistics rule covered ROLLOUT noise but training-run noise was unmeasured
until now. CONSEQUENCE: single-training-run deltas of <5 strict points between arms are NOT
interpretable; today's directional findings that survive this bar: gen1-right vs b2lam03-right
(5-6 vs 1-3 across seeds, marginal), center-transit recoveries (0/20 -> 8-10/10 transit, far above
noise), C2-right 10/10 strict vs everything (far above), left ownership patterns (0 vs 5-10,
marginal at the low end). Claim tier now formally requires seed replication in addition to >=10
rollouts for cross-arm deltas under ~5 points.

**14:00 CONTENTION WINDOW — TWO EVAL FAILURES, BOTH RETRIED (2026-08-13).** gen1lam1's eval server
took >450 s to bind (SERVER_TIMEOUT; manual repro shows slow load, no crash) while gen16's addon ran
its own server + clients — overlapping 6 GB checkpoint loads; the addon's clients died after ~2
trials/cell and its compounds are empty, but it still wrote DONE (the done-marker check greps for
"clearance-clean", which 2-trial output satisfies — marker is completion-shaped, not
completeness-shaped; noted as a harness wart). gen16 addon re-run clean on GPU1; gen1lam1 eval
re-run via run_eval10 on GPU0. Lesson: the GPU-memory gate serializes within a GPU; cross-GPU disk
contention during simultaneous server loads is unguarded.

**gen1lam1 (lam=1 generative): POSTERIOR COLLAPSES TO THE WRONG MODE (2026-08-13 14:30).** Left-start
c2 draws, all 10 rollouts: [-1.1..-4.8], left-mode fraction 0.0 — stronger coupling did NOT calibrate
the 50/50 posterior toward left; it collapsed it entirely onto the RIGHT mode under left prompts
(left 0/10, 6/10 clean). Coupling strength is NOT monotone in binding quality under the CFM
objective (B2-MSE lam=1 owned left; gen lam=1 loses it completely). The start-draw histogram
instrument continues to pay: this diagnosis took one numpy read, zero extra rollouts. Right side
data lost to the SAME unstaggered-client race in run_eval10.sh (arm2b was fixed, eval10 wasn't —
now staggered too); right re-fly running. Current generative-line calibration table (left-mode
fraction under left prompt): gen1 lam=0.3 flat: 0.5 | gen1lam1 lam=1 flat: 0.0 | gen16 lam=0.3
mh16: TBD (left 5/10 success suggests >0.5).

**gen16 FULL ROW: FIRST LEARNED-ARM CENTER COMPLETIONS (2026-08-13 14:55, single run, seed-rep
queued).** CFL 1/10 + CFR 2/10 SUCCESS (transit+goal) — no learned arm had ever COMPLETED center
(one-hot scaffold excepted); with left 5/10 (grid strict: 3/10) gen16 is also the first learned arm
with strict successes on three different tasks. Right 0/10 (its posterior leans left — the mirror of
gen1lam1). Compounds 0. GRID row: L 3/10 strict (8 transit), R 0/10 (3), CFL 1/10 (2 transit note:
judge-vs-clearance join), CFR 1/10 strict (5 transit). gen16s7 (seed 7) launched on GPU1 — tests
whether mode ALLOCATION (which gate a run owns) is itself training lottery, before any basis/lambda
causal claims. Generative-line summary at 5k/lam scan: sampling fixes validity everywhere it runs
(full-magnitude draws, center routes back on flat basis, first center completions on mh16 basis);
posterior CALIBRATION is the single open axis and is NOT monotone in lambda.

**FEATURE-SEPARATION PROBE: COLLAPSE HYPOTHESIS REFUTED; FAILURE LOCALIZED TO CONDITIONING NEGLECT
IN THE CFM (2026-08-13 15:30).** Task separability of each head's OWN pooled features (LOEO
nearest-centroid probe + Fisher), by phase: START = probe-acc 1.00 on EVERY checkpoint (gen1 4.8,
gen1lam1 6.1, gen16 3.6, b2lam03-MSE 3.1 Fisher) — the conditioning information reaches the CFM
input fully intact; the sampler coin-flips anyway. Feature collapse is NOT the mechanism.
Localization: the velocity field under-USES ctx — CFM conditioning gradients concentrate at low t
(near t=1 the optimal v is conditioning-independent), our conditioning coupling is the weakest
possible (one additive concat), and the field was unconverged (loss plateau 0.70, grad 6.6).
gen1lam1's all-starts->right-mode is a degenerate routing (fitting pathology, not information).
UNIFORM secondary finding: TAIL separability degrades to 0.56-0.59 across ALL five arms — the
endgame phase-observability deficit is feature-side and head-independent (quantified, parked).
Principled fixes on the table (both task-agnostic): (1) FiLM/AdaLN multiplicative ctx conditioning
(pi0's own action expert already conditions on time via adaRMS — same mechanism one level up) +
annealed training; (2) conditioning-dropout + classifier-free guidance at serve. Ordering: (1)
first (fixes the training-time cause), (2) composes later. Held for Denis's read.

**EXTRACTION READINESS (Denis, 2026-08-13 17:40).** docs/EXTRACTION.md written: five-part physical
inventory (repo, openpi patch, checkpoints, ctxrun artifacts, data/renderer) with pull commands and
the expensive gotchas. openpi working tree captured completely as
patches/openpi_joint_gen_head_full.patch (549 lines vs upstream 15a9616 — pin, joint head
MSE/logmag/GEN, C2 routing, freezevlm, zero-pad, asset plumbing). Scratchpad-resident viz code +
scene clouds MOVED INTO THE REPO (experiments/rung3/viz/ — /tmp is volatile) and the arm/eval
harness scripts copied to scripts/. Every rung3 instrument carries a header docstring naming what
it measures and the finding it produced. Repo left uncommitted per Denis's convention — the live
git status IS the state; EXTRACTION.md says so explicitly.

**gen16s7 FULL ROW + C2 SEED-REP LAUNCHED (2026-08-13 18:35).** gen16 seed 7: L 1/10, R 7/10 (seed
42: L 5/10, R 0/10) — MODE ALLOCATION FLIPPED ON SEED ALONE; lottery confirmed, basis/lambda do not
control which task the posterior favors; hyperparameter search for calibration is dead, mechanism
fixes (FiLM conditioning / CFG) are the only live path. gen16s7 center: 0/20 success (6+6 clean) —
seed 42's center completions (1+2) did NOT replicate at seed 7; "first learned-arm center
completions" stays a single-run observation, not a claim. gen1det: readout gate PASS at 0.797
(detached CFM, 8-sample mean), rollouts starting. GPU1 -> c2s7: seed replication of the C2
right-gate 10/10 strict headline (the seed rule applies to our best result most of all).

**FiLM GENERATIVE HEAD BUILT + QUEUED (Denis go, 2026-08-13 evening).** SNMVP_HEAD_FILM=1: CFM trunk
input is (c_t, temb) ONLY; conditioning enters exclusively as per-layer (1+gamma)*h+beta from three
explicit channels — STATE (restores the record system's geometry input), LANGUAGE-TOKEN mean pool
(in-checkpoint descendant of the readout that grounded language; the full-prefix pool can fit our
scene-confounded data while ignoring language), IMAGE attention pool. Rationale measured, not
assumed: concat conditioning is ignorable (conditioning neglect), pooled-pixel localization floor
0.14-0.20 m matches the arrival scatter that kills the ending, and the record MLP + langprior both
had state+language inputs the joint heads dropped. Dummy-verified: finite loss; grads reach
film/state/lang/trunk, 0 to unused plain-gen modules; FiLM modulates (same noise, different state
-> different sample). Arm genfilm queued on GPU0 behind gen1det (mh16 basis, lam=0.3 — single
variable vs gen16). Also queued context: c2s7 (C2 replication) training GPU1.

**ACCESS-LOSS PREP (2026-08-13 21:05, ~2h warning from Denis).** docs/status_latest.md written (the
catch-up page: story, autonomous runs, decision state, instruments, artifacts). openpi patch
regenerated WITH the FiLM head (588 lines). Chain scripts copied into scripts/. Running
autonomously past cutoff: c2s7 (eval ~22:10 — the C2 stop replication) and genfilm (train to
~22:20, then gate + six cells overnight; read order documented in status_latest.md). Both write
scores to the standard ctxrun locations; nothing requires supervision.

**C2 REPLICATES: THE SOLVED RIGHT-GATE ENDGAME IS REAL (2026-08-13 21:50).** c2s7 (identical recipe,
training seed 7): RIGHT 9/10 success, 10/10 clearance-clean -> 9/10 STRICT (seed 42: 10/10 strict).
Across two independent training runs: 19/20 right-gate strict — far above the ±5 seed floor. LEFT
0/10 both seeds (1/10 clean s7): the left blindness replicates too. So the factored routing's
profile is STRUCTURAL, not lottery (contrast gen16, whose gate ownership flipped on seed):
FLOW_DETACH + c-only VLM shaping reliably solves the right task INCLUDING the stop, and reliably
fails left. Readout seed-stable (0.938/0.942). Implication for the mid-flight-precision account:
C2's arrivals land in the box on right consistently — its command accuracy on the bound task is
the object of study. Videos for s7 at overlay_armc2s7_right_*.mp4 (claim tier needs Denis's eyes on
both seeds' videos).

**REPO RESTORED OFF-BOX; COMMAND-HEAD TOY: GMM PARITY + THE CALIBRATION LOTTERY DOES NOT
REPRODUCE WITH CLEAN CONDITIONING (2026-08-19, local CPU machine).** Working tree + .git restored
from box-code-backup-2026-08-13 (code/docs/configs only — no checkpoints, no episode data beyond
meta.json, no feature caches, no clog arrays; genfilm/c2genfilm overnight results were never
captured). New toy (experiments/toy_cmdhead/): MSE vs concat-CFM vs FiLM-CFM vs GMM(MDN) heads at
matched capacity on a synthetic branch-state task (start disambiguated only by language; tail
phase observability degraded; nonzero stop signature), 5 training seeds, Bayes-reference tail
posterior. Results: (1) startL-mode fraction 1.00 for EVERY head and seed — the box's 0.0-0.6
coin-flip does not reproduce when ctx is clean and low-dim; conditioning neglect is not intrinsic
to concat-CFM; the calibration lottery is feature/coupling-side (dilution, confounds). (2) MSE
mode-averaging replicated: ~10-sigma invalid means, |fwd|~2.0 below all modes (the mh16 shrinkage
signature). (3) GMM == CFM on validity/calibration (both at the 8-sample noise floor). (4) The
GMM strategy's real payoffs are serve-side: explicit pi(o) turns the start-draw-histogram
instrument into a direct offline readout, and argmax-mode serve is 0.3-sigma valid with ZERO
sampling jitter — while the deployed k=8-mean+EMA smoothing is shown to re-introduce mode
averaging at ambiguous states (8-sample mean ~8.7 sigma invalid). Proposed box arm when access
returns: GMM head with genfilm's information diet, argmax+pi-hysteresis serve.

**LOCAL 4090 PIPELINE REBUILT END-TO-END; K=5 RRR BASIS REGENERATED (RRR~=PCA REPRODUCES); GMM
HEAD IMPLEMENTED; FIRST TWO ARMS LAUNCHED (2026-08-19 evening).** Machine bring-up on top of
LOCAL_CONTINUATION.md, in order: (1) raw npz mirrors regenerated from the rebuilt local/gate_nav
(gate_extract_raw.py needed a local-lerobot fix: frames carry no "task" string, resolved via
ds.meta.tasks; synth=200 real=100, and the regenerated episode layout MATCHES the box convention
exactly — CFL 0-49 / CFR 50-99 / L 100-149 / R 150-199, so joint_head.TASK_EPS and
clog_analysis.EPS stay valid). (2) Renderer venv rebuilt at ~/code/tv (PERSISTENT, unlike the
box's /tmp/tv): py3.10 + torch 2.1.2+cu121 + gsplat 1.5.3 prebuilt wheel (cp310 is the only linux
wheel on the gsplat index) — rasterization smoke-passed on the left splat (6.1M gaussians);
~/code/falsify-pi symlinked to ~/code/falsify so gate_success/gsplat_scene_edit resolve; literal
/home/ubuntu paths rewritten to /home/dfliu in gate_rollout_batch/gate_clearance/joint_head/
refit_rrr_basis; falsify.safety.posthoc imports clean in the openpi venv. (3) Basis:
make_u_rrr_gate_local.py (repo-resident adaptation of the rescue recipe; per-episode language from
meta.json's real task strings instead of the binary is_left labeler — the known label-bug class)
on features from gate_both_pin: held c-R2 +0.966 (feat prior) / +0.970 (state+task prior),
coverage 0.826, principal angles RRR-vs-PCA [0.13 0.15 0.17 0.28 16.5] deg — the box signature
(~=PCA on 4/5 dirs) REPRODUCES on regenerated data + restored checkpoint. Saved
pin_U_gate_rrr_k5.npy sha256 ac49ae6b16bc..., feat_ckpt=falsify/local/checkpoints/gate_both_pin.
(4) SNMVP_HEAD_GMM=1 implemented in openpi-snmvp pi0.py (MDN sibling of GEN/FILM: FiLM information
diet [state, language-token mean pool, prefix attention pool] -> 2x256 MLP -> M*(1+2K) mixture
params, NLL, logsig clamp [-5,2], M=4 via SNMVP_HEAD_GMM_M; exclusive with GEN/FILM). Dummy-
verified: finite loss, grads reach MDN+pool+VLM, exactly 0 to the unused regression head. Serve:
joint_head.head_c MDN branch (argmax-mode default, SNMVP_GMM_MODE=mean for the readout gate =
distribution-center analogue of the CFM 8-sample mean; return_gmm=True exposes (pi, mu));
serve_gate_pin_joint.py does argmax + pi-HYSTERESIS (switch only when incumbent pi falls
SNMVP_GMM_HYST=0.2 below argmax), latch keyed by a client-sent snmvp_trial tag (two batch clients
interleave on one server — a global latch would leak commitment across trials), and CLOG rows are
[pos(3), c(K), pi(M)] so posterior calibration is a direct offline readout (clog_analysis indexes
only [:3+K], unaffected). gate_rollout_batch.py sends snmvp_trial=SCENE_SIDE_t. (5) Two venv-compat
local fixes in the worktree (same class as the local_files_only fix): checkpoints.py
CallbackHandler.async_save ported from 16affa3 (venv orbax 0.11.1 lacks
CommitFutureAwaitingContractedSignals — bit at first save, step 29 of the smoke). 30-step GMM
smoke train PASSED (finite joint loss, checkpoint saved, readout-gate mechanics run on the real
checkpoint). (6) scripts/run_joint_arm_local.sh = single-GPU six-cell chain (arm2b+addon merged;
120 s client stagger kept; JAX server at XLA fraction 0.45 to colocate with the torch render
clients on the one card). ARMS LAUNCHED, sequential: ctl = MSE control, b2lam03's exact recipe
(lam=0.3, DETACH=0, zero-pad, no tail weighting, 5k steps, decay 1e6, seed 42) on the NEW basis;
gmm = SNMVP_HEAD_GMM=1 M=4 same recipe, queued behind ctl. NOTE: all box-era strict scores were
flown on the LOST basis/checkpoints — ctl re-baselines the MSE twin on this machine before the
gmm comparison; cross-machine deltas vs box numbers are NOT interpretable (new basis fit, new
training runs, seed rule applies).

**ctl (MSE control, b2lam03 recipe, LOCAL REBUILD) FULL ROW (2026-08-20 ~02:00): THE B2 SIGNATURE
REPLICATES ON THE NEW MACHINE/BASIS/TRAINING RUN.** Readout gate min c-R2 +0.924 (box b2lam03
~0.93). Six cells, APC=50, seed 42: LEFT 10/10 STRICT (10/10 clean, clearances 0.21-0.33 —
demo-band); RIGHT 10/10 transit + 10/10 clean but 0/10 goal — every flight crosses the right gate
at step ~98-104 then settles OUTSIDE the goal box (the b2-right endgame failure, box: 10/10
transit 3/10 strict; 0 vs 3 is within the +/-5 seed floor); CFL 1/10 transit 0/10 clean — flights
GRAZE the center gate frame at x~2.2 (clearance 0.001-0.14), CFR 0/10 (one wrong-dir transit,
flights don't approach); CMPL 5/5 through gate 1, 0/5 both gates; CMPR pending re-fly. So: left
mastery, right transit-not-goal, center blindness — the flat-basis MSE-head profile carries over
whole. This is the re-baseline the gmm arm compares against. HARNESS NOTES: (a) ctl's first eval
died on missing tv-venv client packages (an early &&-chained install had silently skipped them
when gsplat failed; caught because the readout gate passed while both roll logs showed
ModuleNotFoundError) — packages installed, 1-trial smoke = full strict left success, then the
10-trial re-fly via new scripts/run_sixcell_eval_local.sh; (b) COMPOUND CELLS CANNOT RUN PAIRED
on 24 GB: the duplicated-gate splat clients are 7.0/4.8 GB and cmpr OOM'd next to the 11 GB
server — both chain scripts now fly compounds sequentially; ctl-cmpr re-fly queued behind the
gmm chain (scripts/run_cmpr_refly_ctl.sh). gmm arm (SNMVP_HEAD_GMM=1 M=4, same recipe/basis)
training since 01:25.

**gmm (MDN COMMAND HEAD, M=4, FiLM DIET, ARGMAX SERVE) FULL ROW (2026-08-20 ~10:30): RIGHT-GATE
ENDGAME SOLVED 10/10 STRICT WITHOUT C2 ROUTING; ENDGAME OWNERSHIP FLIPS BETWEEN THE TWINS; pi IS
DEGENERATE.** Same recipe/basis/seed as ctl, only the head swapped (NLL mixture). Readout gate
+0.9410 min per-task (mixture-mean; ctl MSE +0.9238). Six cells: RIGHT 10/10 STRICT (transit +
goal + clean) — the first arm to solve the right endgame without FLOW_DETACH (C2's 19/20 was the
only prior owner); LEFT 10/10 transit + clean but 0/10 goal — transits ~step 93 then overshoots
to endpoints (1.98,-1.60,1.26) vs ctl's successful (1.89,-0.94,1.09), a ~0.7 m -y overshoot; CFL
8/10 TRANSIT (0 goal, 5/10 clean) vs ctl 1/10 — mode-commitment recovers center ROUTING on the
flat basis, replicating gen1's CFM-sampling result but with ZERO sampling jitter; CFR 1/10
transit; CMPL 1/5 both-gates (dwell 0) vs ctl 0/5; CMPR 0/5 both arms (both twins fly to the
right gate then fail gate-1 by the compound judge — same pre-cmpfix compound signature as the
box). SO: ctl owns left-strict, gmm owns right-strict, both 10/10 transit on both sides —
ENDGAME (settle) ownership flipped on the head axis alone at fixed seed. Single training runs:
the 10-point strict flips are far above rollout noise but training-seed rep is still required
before any structural claim (gen16's ownership flipped on seed alone).
**pi(o) IS DEGENERATE — THE INSTRUMENT WORKED AND THE ANSWER IS "NO MIXTURE":** CLOG rows
[pos,c,pi] show component 2 carries 0.74-0.99 at every start and 0.98-1.0 in flight, BOTH
prompts; route identity lives entirely in mu_2(o) (start c differs by prompt: -1.87 left vs
-2.72 right). The toy-predicted feature-side start ambiguity (pi~0.5) did NOT appear at box
scale; the pi-hysteresis latch never fired (no component switches to smooth). Mechanism
hypothesis (unverified): with NLL + learned sigma the head is HETEROSCEDASTIC regression —
high-variance branch rows are down-weighted by their own sigma, so mu escapes the mode-average
corruption that drags plain MSE, which would explain argmax==mu2 beating the MSE twin
closed-loop despite both being deterministic functions of o. Testable offline: compare
per-phase sigma(o) against the measured branch-state rows.
HARNESS: gmm's compound cells hit the same paired-client OOM (its running chain predated the
sequential fix — atomic-replace preserves the old fd by design); cmpr re-flown solo
(run_cmpr_refly_gmm.sh). Score files: arm_{ctl,gmm}_scores.txt / ctr_gmm_scores.txt /
ev6_ctl_{scores,ctr_scores}.txt (+cmpr REFLY sections). Videos overlay_arm{ctl,gmm}_{left,right}_*
await Denis review (claim tier). Next per the plan ladder: (1) seed-7 replications of BOTH twins
(the 0<->10 endgame flips are the claims that matter); (2) offline sigma-phase probe; (3) if the
heteroscedastic account survives, the mh16-basis pairing and the langprior-style explicit-channel
ablation become the natural follow-ups.

**mh16 REBUILT LOCALLY AND TAIL-CAPTURE VERIFIED; gmm x mh16 LAUNCHED (2026-08-20 ~12:15).**
Denis's push-performance directive after the twin-endgame review (viz/twin_endgame.html,
artifact 2ab21263): check the multi-horizon basis expresses the tail, and if so fly the GMM on
it. pin_U_mh16.npy rebuilt from the hand-constructed recipe (cumulative displacement over
{6,12,25,50} steps x {x,y,z,yaw}, QR; sha256 9d53b141b216...). basis_phase_capture.py (new
instrument, zero-pad train-target convention): WITHIN-TASK stop-segment capture flat->mh16 =
left .57->.84, right .43->.83, CFL .62->.82, CFR .66->.83; mid-flight [.5,.75) right .43->.88,
left .51->.90 — reproduces the box's 0.34->0.81 finding in structure on regenerated data. The
flat-basis table + per-arm clog drift (each twin command-accurate only on the side it wins;
gmm-left wrong ON-manifold 0.72 m; ctl-right degrades off-manifold 0.59 m) say the settle
failure is over-determined: expressiveness AND command content. Arm gmmmh = SNMVP_HEAD_GMM=1
M=4 x pin_U_mh16 (K=16), otherwise the b2lam03 recipe, seed 42, full six-cell chain. CAUTION
carried from the box: "mh16 goes crazy at the end" — hard-pinned short-horizon components turn
WRONG tail commands into thrash (flat fails gracefully, mh fails energetically); if gmmmh's
tail commands inherit gmm-left's on-manifold error, expect energetic left failures — the
start-draw/pi CLOG and clearance columns are the first read.

**gmmmh (GMM x mh16) FULL ROW (2026-08-20 ~19:30): BEST LEFT EVER + CENTER ROUTING NEARLY
SOLVED; RIGHT/CENTER TAILS FAIL ENERGETICALLY — THE AMPLIFICATION ACCOUNT CONFIRMED.** Readout
gate min per-task c-R2 +0.62 (right worst; K=16 predictability cost, exactly as the offline
ridge sweep forecast: mh16 R2worst 0.45). Six cells: LEFT 10/10 STRICT with the cleanest flying
of ANY arm (min clearances 0.388-0.406 vs ctl 0.21-0.33; crosses DEAD-CENTER at (0.88,0.71) ~=
the anchor (0.861,0.694) — the historical off-center-crossing flaw is gone in this arm), transit
~step 88. RIGHT 10/10 transit but 0/10 goal, 4/10 clean: late (~136 vs gmm-flat's ~100),
crossing near the left post (0.30,-1.28), 6/10 grazes 0.12-0.17 m — gmm-flat's 10/10 strict
right did NOT carry to the new basis; endgame ownership flipped AGAIN, this time on the BASIS
axis within the same head. CFL 9/10 + CFR 10/10 TRANSIT (best center routing ever; ctl 1/10,
gmm-flat 8/10+1/10) but 0 goal, 0 clean, and CFR shows wrong_dir=2-3 with up to 162 steps inside
0.18 m — OSCILLATION AT THE GATE, the box's "mh16 goes crazy at the end" signature: hard-pinned
short-horizon channels turn wrong tail commands into thrash where flat merely overshoots.
Compounds 0/5 (cmpr 5/5 clean). READ: the expressiveness lever WORKS where the tail command is
right (left: precision + settle both improved) and AMPLIFIES where it is wrong (right, center
endgames). Consistent with the ceiling analysis (same day, candidate_basis_eval.py +
predictability-ceiling probe): expressiveness saturates fast; only ~0.82-0.83 of chunk variance
is predictable from the observation AT ALL (kNN==ridge -> information limit, not linearity);
predictable subspace rank ~9; flat carries 0.67, mh16 0.70 of the 0.83 ceiling; uniform
segment-displacement bases (seg16/seg24) dominate mh16 at equal K; anything past K~16 pins
channels no head can serve. Artifact 2ab21263 updated with the gmmmh overlay. NEXT candidates
(Denis to pick): gmm x seg16 (predictability-optimal basis point), seed reps of gmmmh-left +
gmm-right (the two 10/10 strict claims), and the now-binding problem: TAIL COMMAND CONTENT
(sigma-by-phase probe; tail observability signals).

**SIGMA-BY-PHASE PROBE: THE EXPRESSIVE-BASIS MDN KNOWS WHEN IT IS GUESSING; gmm x seg16 LAUNCHED
(2026-08-20 evening).** sigma_phase_probe.py (new instrument): argmax-component (mu*, sigma*) on
demo frames vs oracle c, per task x phase. gmmmh (mh16): Spearman corr(||sigma*||, ||err||) =
0.82 POOLED (0.77-0.87 every task), 0.40 tail-only. gmm (flat): 0.52 pooled, -0.03 TAIL — the
flat head's uncertainty is uninformative exactly where the endgame fails. Read: expressive
targets give the NLL something real to be uncertain about; the flat basis compresses the tail
away and sigma learns nothing there. CONSEQUENCE: a sigma-gated serve (soften the pin toward
plain denoising when the head's own sigma* blows up — observation-dependent trust from the
head's trained uncertainty, NOT a phase/regime patch) is a viable mechanism for exactly the arm
family that fails energetically when confident-and-wrong; serve-side only, no retraining. Design
next; falsifiable prediction: gmmmh-right/CFR thrash rows should show elevated sigma* closed-loop
(check clog... sigma not logged yet — add sigma to the CLOG rows in serve_gate_pin_joint before
the next MDN eval). ARM LAUNCHED: gmmseg = GMM x pin_U_seg16 (sha 22f8002927b5...; uniform
window displacements, the offline-dominant K=16 point: stop capture 0.84-0.87 >= mh16 with
flat-level predictability R2all 0.849/worst 0.475 on the ridge proxy), b2lam03 recipe otherwise,
seed 42, six-cell chain.

**gmmseg (GMM x seg16) FULL ROW: ZERO STRICT CELLS — THE OFFLINE PREDICTABILITY ORDERING DID NOT
SURVIVE JOINT TRAINING; sigma EXPLAINS IT (2026-08-21 ~02:30).** Readout gate min +0.51 (right),
loss 3.34 (gmmmh 3.19). All six cells 0 strict: left 10/10 transit (fast ~step 80, dead-center,
0.36-0.38 clean) then overshoots +0.87 m in x; right transits late (~167) then sails 2.7 m -y;
CFL 0/10 clean (grazes), CFR does not approach; compounds 0. THE NEW sigma COLUMN (CLOG rows now
[pos,c,pi,||sigma*||]): sigma* ~10-11 FLAT across start/mid/late (deciles 6.9-13.9) — the head
never fit the seg16 targets sharply, uniform high uncertainty, so argmax-mu is uniformly
imprecise. The ridge-proxy prediction (seg16 >= mh16 predictability) was WRONG under joint
training at head capacity. Mechanism hypothesis: mh16's components are PREFIX displacements,
all anchored at t=0 and monotone-nested -> robust to timing jitter; seg16's detached late
windows ("displacement in steps 37-50 alone") ALIAS across window boundaries under phase
uncertainty — a 2048-d linear probe compensates, a small joint MLP cannot. Lesson recorded:
offline basis predictability screens need a capacity-matched nonlinear head, not ridge.
Standings unchanged: gmmmh owns left(+record clearances)+center routing; gmm-flat owns right.
Next per the approved plan: sigma-GATED SERVE on gmmmh (its sigma-error corr is 0.82) —
serve-side pin softening alpha(sigma*), no retraining, re-fly the failing cells.

**SIGMA-GATED SERVE: PREDICTION NOT CONFIRMED — CLOSED-LOOP sigma IS OFF-SCALE vs DEMO
CALIBRATION AND THE GATE DEGENERATES TO A MOSTLY-UNPINNED SERVE (2026-08-21 ~04:30).**
Implementation: SNMVP_GMM_SIGGATE lo,hi,amin in serve_gate_pin_joint (alpha ramps 1->0.25 as
||sigma*|| crosses demo p60->p90 = 3.74->9.72; c_eff = alpha*c + (1-alpha)*(g@U); CLOG rows
now [pos, c, pi, sigma*, alpha]). Re-fly of gmmmh's three failing cells, 10 trials each:
right 0/10 (2/10 clean, was 4/10 ungated), CFL 0/10 (2/10 clean, was 0), CFR 0/10 (4/10 clean,
was 0 — the oscillation-at-the-gate rows did soften). THE MEASUREMENT THAT MATTERS: closed-loop
sigma* mean 9.65, deciles 6.7-12.8 — far above the demo distribution (p50 2.54, p90 9.72);
alpha NEVER reached full trust and floored at 0.25 on 71% of replans. Two readings, both
recorded: (a) part is calibration phase-mix (whole-episode demo quantiles vs replans every 50
steps), but closed-loop sigma exceeds even demo start-phase levels — the head is honestly MORE
uncertain on rollout observations than on any demo frame, i.e. sigma detects the covariate
shift itself; (b) as a GLOBAL trust dial the gate therefore unpins nearly everything, and a
mostly-unpinned gmmmh flies like the scratch control (transits, never completes) — softening
toward denoising removes command ERRORS but also removes COMMANDS; the base flow does not know
the endgame, which is the entire reason the pin exists. Net: sigma is a good DETECTOR
(rho=0.82 on-manifold; saturates high off-manifold) and a bad ACTUATOR when wired as global
trust. Refinements exist (relative/within-trial sigma normalization; per-component gating of
only the short-horizon channels) but they trend toward the regime-patch family Denis rejects —
parked pending his read. PLAN COMPLETE (probe -> gmmseg -> sigma-gate): standings are gmmmh =
left 10/10 strict w/ record clearances + center routing 19/20 transit; gmm-flat = right 10/10
strict; every claim single-training-run, seed reps owed.

**c2gmmmh LAUNCHED (Denis go, 2026-08-21 ~08:10): C2 ROUTING x GMM HEAD x mh16 BASIS.** The
composition bet: C2's FLOW_DETACH reliably buys the right endgame (19/20 strict across two
seeds, MSE head, flat basis) and gmmmh buys left precision (10/10 strict, record clearances) +
center routing (19/20 transit) — complementary failure profiles. This is the local successor of
the box's lost c2genfilm overnight (C2 x generative x mh16, never read). Config verified in the
train log: K=16, detach=False, lam=0.3, flow_detach=True, gmm=True M=4. Same recipe/seed 42,
full six-cell chain. Risks named in advance: with the flow loss detached from the VLM, the
c-loss alone shapes the representation — C2's structural left/center blindness may return; and
sigma saturates off-manifold (2026-08-21 finding), so if the composition inherits C2's
narrowness the pi/sigma CLOG columns should show it directly at the unbound tasks' starts.

**gmsig LAUNCHED (Denis go, 2026-08-21 ~08:48): THE TRAINED TRUST DIAL — sigma-CONDITIONED
GMM x mh16.** Design from the strategy discussion (use the pin when certain, devote strength to
the FM head when not; the failed serve-only alpha-gate showed the flow must be TRAINED for
reduced trust). Composition of three prior pieces: (1) isotropic per-sample pin-noise
SNMVP_PIN_NOISE=1.5 + RAND (sigma ~ U[0,1.5] c-std; isotropic per the covpin pre-registered
negative — error-matched noise teaches ignoring the command subspace); (2) PIN_NOISE_COND sigma
conditioning (the "designed lever" named 2026-08-06 after fixed soft-pin 0.35 lost the right
goal phase; never flown); (3) serve-side sigma_serve = calibrated map of the MDN's own
||sigma*|| (rho=0.82 err tracking), computed per replan, command ALWAYS delivered at full
amplitude — only trust is modulated, unlike the alpha-gate that deleted the command. Verified
before launch: head NLL target stays the CLEAN oracle c under pin-noise (code audit); sigma
threads sample_actions->embed_suffix as a traced array (no per-value recompile; dummy: default
== sigma0, sigma1.2 differs, RAND+COND loss finite). New plumbing: pi0.py sample_actions
snmvp_sigma kwarg; policy.py infer snmvp_sigma; server SNMVP_SIGMA_MAP json (piecewise-linear
sig* -> sigma_serve in c-std units, cap 1.5 = train max; built AFTER training from
sigma_phase_probe --save rows on the NEW head via make_sigma_map.py — sigma* distributions are
checkpoint-specific). CLOG rows now [pos, c, pi, sigma*, alpha, sigma_serve]. Chain:
run_gmsig_post.sh (gate -> calibration -> six cells). PRE-REGISTERED: (1) left stays 10/10
strict; (2) right/CFR energetic failures turn graceful (clean recovers, transits keep);
(3) weak-side settles improve only if partial trust + trained correction suffice — the
mechanism cannot create endgame knowledge vision lacks (scratch 0/20 bounds it).

**gmsig FULL ROW (2026-08-21 ~16:30): THE TRAINED TRUST DIAL DELIVERS — FIRST-EVER CFR
COMPLETIONS (10/10 judge), LEFT RECORD KEPT, RIGHT DOWN TO A 4 cm MISS; ALL THREE PRE-REGISTERED
PREDICTIONS HELD.** Readout gate +0.6587 min (> gmmmh's 0.62 — pin-noise training leaves the
head untouched, as designed). Calibration map monotone sig* 1.25->0.105 ... 9.92->0.957 (under
the 1.5 cap); closed-loop sigma_serve deciles 0.16/0.64/0.96 (L/R) — the dial lives MID-SCALE,
genuinely modulating per replan, not saturated. Six cells vs gmmmh: LEFT 10/10 STRICT kept
(0.387-0.388 clean, dead-center); CFR 10/10 SUCCESS (transit+goal; NO learned arm ever
completed CFR before — prior best 2/10 gen16 single-run), oscillation tamed (wrong_dir 1,
contact-steps 0-28 vs 93-162), 4/10 clearance-clean (6 graze 0.08-0.16) -> strict 4/10 pending
video; RIGHT 0/10 goal BUT 10/10 clean and endpoints (1.62,-0.70,1.54)+/-0.04 — planar position
INSIDE the goal footprint, hovering 4 cm above the box z-ceiling (1.5); the right settle
collapsed from miss-by-meters to a centimeter-scale height bias; CFL 0 goal, 7/10 clean
(graceful); CMPL 4/5 clean, CMPR 5/5 clean, 0 both-gates. gmsig is the presumptive recipe.
Single training run; center cells VIDEO=0. Next (Denis go): z-miss forensic, then seed-7 rep
with VIDEO=1 center for claim tier.

**RIGHT-CELL Z-MISS FORENSIC (2026-08-21): NOT A 4 cm PROBLEM — A WRONG-SIGN TAIL COMMAND THE
DIAL CORRECTLY DISTRUSTS, DOWNSTREAM OF A HIGH ARRIVAL; SEED-7 REP LAUNCHED.** The interleaved
L/R clog needed side-splitting (both tasks converge on the SHARED goal box: rows at
(1.55,-0.64,1.17) are LEFT trials already in-box). The true right-tail rows: at
(0.9-1.7, -0.9..-1.6, z~1.53-1.56) the served command says dz = +0.014..+0.047 (UP/flat) while
the demo oracle at matched planar states says dz = -0.179 (DESCEND); sigma* there is 7.7-10.3
-> sigma_serve 0.66-0.96, i.e. the head is wrong AND knows it, the flow heavily distrusts, and
its own prior is hover -> equilibrium 4 cm above the box ceiling. Upstream cause: rollouts
ARRIVE HIGH (z 1.53-1.67 through/past the right gate vs demos ~1.5 falling to 1.25 by the
matched states) — off-manifold in z, where no demo teaches "descend from 1.55" — the
restoring-field/coverage data gap (explicitly deferred by Denis) in miniature, now localized to
one scalar at one region. Demos settle at z 1.00+/-0.02 with -0.13 m over the last 60 steps.
LAUNCHED: gmsigs7 (identical gmsig recipe, seed 7) + run_gmsigs7_post.sh — claim-tier seed rep
for the two headline cells; center cells fly VIDEO=1 this time (overlay OUT template fixed in
the derived script; compounds stay VIDEO=0).

**gmsigs7 SEED REPLICATION (2026-08-22 ~04:30): LEFT 20/20 STRICT ACROSS SEEDS; CFR COMPLETIONS
REPLICATE (16/20 judge across seeds); SEED 7 ADDS THE FIRST LEARNED-ARM COMPOUND COMPLETIONS
(CMPL 5/5 both-gates + dwell).** Readout gate +0.674 min (seed 42: +0.659; both healthy; note
the scores-file header says seed=42 — a copied echo line, training verifiably ran --seed=7 into
gate_pin_joint_gmsigs7). Cells: LEFT 10/10 STRICT again (both seeds 10/10 strict + clean =
20/20 — RECORD-BOARD CANDIDATE pending Denis video, overlay_armgmsig*/armgmsigs7 left files);
RIGHT 0/10 again but 10/10 clean, and the MISS MODE MOVED: endpoints (2.19,-0.45,1.18) — z now
in-box, planar overshoot +0.4 m in x (seed 42: z-hover 4 cm high at correct x/y) — settle
misses persist across seeds while the miss geometry is seed-lottery; CFR 6/10 judge success
(strict join 5/10: trials both success+clean; seed 42: 10/10 judge, 4/10 join) — CENTER
COMPLETIONS REPLICATE, 16/20 judge across two independent training runs, VIDEO=1 this time;
CFL 3/10 success (seed 42: 0) — even CFL completes sometimes; CMPL 5/5 SUCCESS with dwell 63-70
steps (0/5 clean — grazes) — THE FIRST LEARNED-ARM COMPOUND COMPLETIONS EVER, on the novel
compound prompt, though 5-trial screen tier and clean fails; CMPR 1/5 success 4/5 clean.
CROSS-SEED READ: the trust-dial recipe's LEFT and CFR abilities are structural (replicated);
CFL/compound completion ability is real but seed-lottery in degree; right settle is the one
cell no seed has closed — and per the z-forensic it is the coverage gap (no demos teach
recovery from the arrival states), not head class, basis, or trust wiring. CLAIM LADDER NOW:
left 20/20 strict (claim pending video), CFR 16/20 judge (claim pending video + clean-join
discussion), CMPL awaiting a 10-trial + video re-fly to leave screen tier. gmsig recipe =
GMM(M=4) x mh16 x SNMVP_PIN_NOISE=1.5+RAND+COND x calibrated sigma_serve — the standing best.

**COMPOUND 10-TRIAL VIDEO RE-FLY (2026-08-22 ~07:40): gmsigs7 CMPL 10/10 JUDGE SUCCESS —
15/15 ACROSS BATTERIES WITHIN SEED — BUT 0/10 CLEAN, AND SEED 42 GOES 0/10: COMPOUND
COMPOSITION IS TRAINING-SEED LOTTERY, NOT YET STRUCTURAL.** gmsigs7 CMPL x10 VIDEO: 10/10
ordered both-gates + dwell 40-84 frames, transits at nearly identical steps (gate1 88-90,
gate2 232-241 — the argmax serve is deterministic and the route is committed); min-clearance
0.008-0.149 ALL FAILING, mostly at steps 436-500 near (2.3,-0.37,1.1) — AFTER the dwell latch,
i.e. post-goal drift back toward the center gate frame; human video will adjudicate what that
looks like (overlay_c10gmsigs7_cmpl_*.mp4). gmsigs7 CMPR 1/10 (9/10 clean) — right-side
compound still blocked by the right-tail gap. gmsig (seed 42): CMPL 0/10 (6/10 clean),
CMPR 0/10 (10/10 clean). VERDICT per the standing rule: within-seed the seed-7 compound
ability is rock-solid (15/15 judge) and now sits at >=10 trials with video, but it FLIPS to
0/10 on training seed alone — recorded as a seed-lottery ability of the recipe, like gen16's
gate ownership; the STRUCTURAL claims of the trust-dial recipe remain LEFT 20/20 strict and
CFR 16/20 judge (both cross-seed). The novel-prompt composition demonstrably EXISTS in the
learned policy class — the first time ever — but which training run gets it is uncontrolled;
candidate control levers are the calibration posterior at the compound prompt (pi/sigma CLOG
comparison seed 7 vs 42 at the switch point) before any hyperparameter hunting.

**PIN SIM-vs-REAL PROPERTIES (Denis directive, 2026-08-22): A ~7-DIM SHARED CORE TRANSFERS;
THE DIVERGENT REMAINDER IS PRICED ALMOST 1:1 BY THE HEAD'S OWN sigma — THE TRUST DIAL IS THE
TRANSFER BRIDGE.** New instruments pin_real_vs_sim.py (chunk space, CPU) + real_head_probe.py
(head on real frames); real = data_gate_real 100 teleop eps, all zero-pad convention. (1)
CAPTURE: mh16 on real = 0.67 overall / 0.76-0.78 stop (synth 0.92 / 0.83-0.84); flat K5 real
stop 0.30. Real EARLY capture drops hardest (0.55-0.59 vs 0.86-0.91) — teleop high-frequency
corrections are off-span; note real per-channel c std is 2-3x synth. (2) SUBSPACE: within-task
PCA16 real-vs-synth principal angles [1.3..13.9] for the first SEVEN dirs, then 49-90 deg — a
~7-dim shared core (matches the predictable-rank ~9 ceiling finding); real variance is
heavier-tailed (own-PCA16 captures only 0.84 of real vs 0.985 of synth). (3) MATCHED-STATE
ORACLE GAP at K=16: cos 0.38-0.44, ~1.0-1.2 std per band, UNIFORM across h6/h12/h25/h50 (h50
mildly best) — the box's "behavior-dominated gap" upheld; no band is spared. (4) HEAD ON REAL
(gmsig): c-R2 0.375 left / 0.193 right, mean|err| 8.8-10.6 — BUT mean sigma* 9.6-9.8 (~1:1
with actual error; synth tail pairs also ~1:1), rank corr(sigma,err) 0.76-0.80 pooled and
0.92-0.93 AT THE TAIL — better-calibrated on real than on synth. The sigma-conditioned serve
would map real frames to sigma_serve ~0.95 (near cap) automatically: pin hard where sim
knowledge applies, hand over to the FM head where it does not — the strategy generalizes
across the domain gap WITHOUT recalibration. CAVEATS RECORDED: (a) all arms train on
local/gate_nav = synth+real MIXED, so this is not zero-shot head transfer; the low real R2 is
honest real-data unpredictability (per-domain-U 2026-08-07: real chunks harder in ANY basis),
and the K contrast explains the box's higher K=5 real R2 (0.86): flat K5 spans ~only the
shared core; K=16 also carries the non-transferable variance — which sigma prices. (b)
Closed-loop rollout sigma* (~9.65) ~= real-frame sigma* (~9.7): the head prices rendered
rollout states like real data — one uncertainty scale covers both gaps. RELATIONSHIP VERDICT:
close where it matters — the coarse/shared core IS the pin's cargo, and the non-shared
remainder is exactly what sigma already hands to denoising.

**CENTER-GATE PIN ON REAL OBSERVATIONS (Denis directive, 2026-08-22): DOES NOT TRANSFER —
REAL-PROMPTED CENTER COMMANDS ARE DIRECTIONLESS MUSH, NOT A ROUTE; SIM-PROMPTED ONES ROUTE
CORRECTLY.** New instrument center_pin_real_probe.py + page center_pin_real.html (artifact
51f8c22b): head (gmsig) prompted with CFL/CFR on early-flight frames (real eps have NO center
demos; starts match sim within 0.15 m), argmax mu* decoded to its implied 50-step path.
SIM frames: coherent command fans — CFL heading +17.7 deg (circ-std 9.8), CFR -45.1 (10.0),
|disp| 0.81-0.84 m, between the start-adjacent gate direction and the center route as expected
for a first chunk; cos vs demo oracle 0.48-0.55 (the branchy start, sigma* 5.8-6.9 — high even
in sim). REAL frames, same prompts: |disp| collapses to 0.27-0.28 m (3x shrink), heading
circ-std 88-108 deg — NO consistent direction; cos(real,sim) at matched positions -0.18/-0.19,
gap 5.4-6.2 std; sigma* 6.4-6.8 (high, but NOT higher than sim's center-prompt sigma — the
head does not separately flag the domain here, both are priced "uncertain" for the same
branch-ambiguity reason). READ: zero-shot task-command synthesis (real pixels x center
language) FAILS in the current head — the real-domain features never learned the center task
binding (real training rows are L/R only; the center binding lives in sim-frame features and
does not ride the shared visual representation). The shrunk-magnitude directionless commands
are the mode-averaging signature at an unresolved branch. Contrast the L/R real story (head
domain-faithful on tasks real data CONTAINS): the pin core transfers; TASK BINDINGS do not
cross domains without either real center data or a domain-invariant command representation.
This is the first concrete measurement of the north star's composition claim across the
sim-real boundary — negative for the current architecture, and localizable: the failure is in
the head's task-conditional, not the pin (the same mu* decoded on sim frames routes fine).

**SYNTH DATA GENERATION PIPELINE REVIVED ON THIS MACHINE (2026-08-22 ~23:00): full chain
verified with a fresh center-gate episode.** Chain per the falsify repo: plan_course_variants.py
(course YAML embedded perturbation blocks -> MPC-planned Trajectory NPZs, acados/qpOASES via
LD_LIBRARY_PATH from tools/env.sh, prebuilt libs at ~/code/SousVide/external/FiGS/acados/lib)
-> falsify.cli.export_training_data (gsplat 0.1.13 render, JIT CUDA build needs ninja BINARY on
PATH: package was installed but .venv/bin not on PATH — the venv-bin-on-PATH fix) ->
assemble_synth_dataset.py (LeRobot v2.1). Smoke: 2 variants planned (1 valid, 1 dropped by
validation — --ignore-collision exists for the documented strict-collision case), 1 episode
rendered: 301 rows, image/wrist/3pov 256x256 RGB, 7-D state/actions, geometry sane. Exporter
wart noted: it prints "[export] 1 episode(s)" even when the episode FAILED (empty dir) — check
for the parquet, not the summary line. All six course YAMLs present incl. compounds. COST
BASIS: planning ~seconds/variant; render ~1-2 min/episode on the 4090 -> a 100-episode
regeneration or augmentation batch is an overnight job. This unblocks the two data directions
the transfer analysis pointed at: (a) targeted right-tail coverage (descend-from-high-arrival
variants), (b) any center-data recut for the sim-real binding gap.

**SYNTH REGENERATION LAUNCHED (Denis directive, 2026-08-22/23): COURSE FAMILY FIXED — THE
"CLIPS ON THE WAY BACK" BUG WAS LATENT IN THREE OF FOUR COURSES; START VARIANCE NOW MATCHES
REAL.** Verifications first: exporter emits true RGB pinhole per the 2026-06-12 convention
(channel stats agree with training data; BGR/fisheye era frozen in a legacy embodiment YAML);
existing CFR demos scored 43/50 clean with wrong_dir=0 (the demos return WEST of the aperture
at x~2.0-2.1, one within 0.22 m of the post — the ROLLOUTS cut that narrow corridor to x~2.4,
through the frame). START VARIANCE: real starts std (0.151,0.100,0.122)+yaw 0.058 over a 0.7 m
range; synth was a 0.022-std pinpoint at (0,0,1.5), zero yaw — 5-7x under-dispersed, +0.2 m
x-biased. FIXES (falsify repo; .bak-20260822 copies kept): (1) plan_course_variants.py grew
--start-jitter/--start-mean (Gaussian start-waypoint sampling; filtering alone would have
TRIMMED the variance since clipping correlates with start offset); (2) CFR: cross_west return
waypoint pins the southbound re-crossing at x=1.30 (>1 m from the west post), pre_gate z
1.5->1.38, corrective cap 0.3->0.25 (up-variants grazed the 1.875 top bar); (3) RIGHT: pre_gate
funnel on the aperture normal + return_east berth — the nominal post_gate->hover leg passed
0.21 m from the EAST post (COLLISION_GATE on ~half of jittered variants, at post-transit steps
64-79, crossings themselves dead-center); (4) CFL: return_south berth (0.31 m west-post leg) +
corrective cap; (5) LEFT: pre_gate funnel. ACCEPTANCE SWEEP at 20/course under full start
jitter: 80/80 judge (wrong_dir 0), 79/80 clearance-clean. GENERATION PROTOCOL: samples 70/course
-> score every plan with the posthoc judge + gate_clearance -> keep FIRST 50 passing both
(same per-task counts as the original dataset); phase A (plan+filter) running, phase B (render,
~200 eps overnight) queued; norm stats will be REUSED from gate_nav so U/c units stay
comparable; real 100 eps carried over unchanged into the new training set.

**gate_nav2 BUILT + gmsig2 LAUNCHED (2026-08-23): the trust-dial recipe retrains on the fixed
data.** Phase A kept 50/50/50/50 with pass rates 67-70/70 (the course fixes hold at scale;
right went 25% -> 100% valid). Phase B rendered 200/200 parquets. build_gate_nav2.py wrote
local/gate_nav2 preserving gate_nav's episode-order convention (real 0-99 carried over
BGR->RGB; synth CFL/CFR/L/R 100-299 from regen1, already-RGB); verified loading: 300 eps,
79,625 frames, exact task strings, start std (0.19,0.10,0.11) in the loaded data. pi0_gate2
config added (worktree); norm stats = gate_nav's (copied, NOT recomputed — U/c comparability).
Readout/probe demo dir now env-driven (SNMVP_DATA_DIR; data_gate_synth2 mirrors extracting).
TRAINING: gmsig2 = identical gmsig recipe (GMM M=4 x mh16 x PIN_NOISE 1.5 RAND+COND, seed 42)
on pi0_gate2; post chain queued (gate on NEW demos -> own sigma calibration -> six cells).
What the row adjudicates: (a) does CFR's clean rate recover (rollout grazes were taught by the
narrow demo return corridor, now >1 m); (b) does right improve (returns no longer skim the east
post; starts span reality); (c) does left 10/10 survive the data change; (d) sigma* on rollout
states should DROP vs gmsig if the start-variance gap was part of the off-manifold pricing.

**REGEN1 DEFECTS DIAGNOSED (Denis review of the demo-fan page, 2026-08-23): THE LEFT FUNNEL
CANCELED THE COURSE'S RECOVERY VOCABULARY; POST-GATE VARIANCE COLLAPSES; STARTS ARE ACTUALLY
MATCHED (evidence below). gmsig2 training continues; regen-2 planned.** (1) LEFT SHAPE: the
left course's corrective block targets `approach` with the FAMILY'S LARGEST magnitudes
(0.2-0.5 m, p=0.7) — the old data's early bloom (|std| 0.31-0.37 at steps 15-40, incl.
right-swinging starts) IS that corrective expressing itself. My added pre_gate funnel sits
immediately after `approach` and re-converges the spline — it fixed the 2/10 collisions by
SUPPRESSING the recovery-teaching displacement (new early spread 0.26 is start-jitter, not
corrective swing; initial-heading std 30->20 deg). (2) POST-GATE COLLAPSE: new left spread
falls 0.26 -> 0.087/0.039 after transit (old kept 0.083-0.125 through the arc) — post-gate
waypoints carry only the 0.05 ball jitter + timed hover anchor -> MPC converges 50 variants to
one rope. (3) STARTS: real-vs-new percentile match on ALL axes (x p5/50/95: -0.45/-0.20/0.03
vs -0.44/-0.19/0.06; y: -0.10/0.04/0.20 vs -0.12/0.04/0.22; z: 1.23/1.48/1.60 vs
1.24/1.43/1.63) — x-before-zero and the y spread ARE emulated; the viewer under-shows it
(starts sit inside the cloud). REGEN-2 DESIGN (pending Denis): (a) left/right courses adopt
CFR's pattern — corrective target moves to the pre_gate funnel point itself (perturb the
funnel, spline recovers into the gate: recovery vocabulary preserved AND transit protected)
with left magnitudes restored toward 0.2-0.4; (b) driver gains --waypoint-jitter
"name:radius,..." for post-gate/arc/return diversity (~0.15-0.20 m) without touching transit;
(c) starts unchanged; (d) plan-to-quota filter stays as the backstop, with an
accepted-vs-planned start-distribution check to catch rejection bias.

**REGEN-2 RECIPE CONVERGED AND VERIFIED (Denis's two-waypoint-class design, 2026-08-23):
79/79 judge, 78/79 clean, PACING RESTORED, TUBES + GATE SPHERES EXPRESSING EXACTLY.**
Implementation arc: (1) per-waypoint jitter classes in falsify planning (Waypoint.jitter_m,
loader, sampler) — gate apertures 0.03 spheres, corridor waypoints 0.15-0.20 tubes, start 0
(real-matched --start-jitter owns it); (2) the PARKING bug root-caused: falsify's
_DEFAULT_POLICY_CFG hardcodes snap kT=10.0 (minimum-TIME weight) so MinTimeSnap COMPRESSES the
schedule regardless of waypoint t anchors — regen1's data is ~60% stationary hover (park step
96/241 vs old 239, real 221); driver gained --snap-kt none (pure min-snap honoring keyframe
times); (3) pure min-snap then BALLOONS over long un-anchored segments (left return spread
0.61-0.67; CFR swung OUT_OF_BOUNDS backward on its 17.5 s approach) — fixed with old-profile
time anchors: gate t per old transit (7.0/10.0/19.0/17.5), post_gate t, mid-return anchors
(left gained `descend` at the old step-170 mean; return_east/return_south/arc_left/cross_west
anchored), center courses gained early-leg anchors (approach t=8, east_leg t=12 at old means);
(4) CFR transit collided at slow crossing speed (crawling near the frame's north face at
0.03 m body margin) — pre_gate standoff 0.75->0.90 + t 16.2 + tube 0.10, post_gate t 19 ->
crossing at old speed. FINAL SWEEP (20/20/20/19 planned, valid rates 100/100/100/95):
judge 79/79, clean 78/79; park 217-281 (old 239, real 221); spread profile bloom
0.27-0.47 -> gate 0.04-0.21 -> return tubes 0.13-0.17 (old 0.06-0.13; wider per Denis's
explicit ask) -> settle 0.05-0.09; starts pooled p5/50/95 match real all axes. regen2 phase A
(plan 70/course, keep 50 judge+clean) RUNNING; render + gate_nav3 + retrain queued behind
gmsig2's row and Denis's read. NOTE: gmsig2 (in-flight) trains on regen1 = hover-skewed data;
its row gets read with that caveat and the gmsig-vs-gmsig2 comparison is NOT clean.

**CFR PORPOISE FIXED (Denis catch on the regen-2 page, 2026-08-23): min-snap spends schedule
slack VERTICALLY — anchor-ladder principle established.** Denis flagged "crazy stuff before the
gate" on CFR: measured, the pre-gate window flew z 0.57..2.19 (old 1.14..1.57) and 3.45 m of
path (old 1.82) with ZERO heading reversals — not loops, a vertical porpoise: the
east_leg->pre_gate segment covered 0.5 m with 4.2 s of slack and the polynomial spent it in z.
Fix iterations: se_corner t=14.5 splits the slack (z back to 0.97..1.72) -> residual
OUT_OF_BOUNDS back-swings on the 8 s first leg -> drift anchor t=4 at the old step-40 mean.
FINAL CFR: 19/20 valid, z 1.0..1.7, park 285/301; refresh into regen2 kept: 69/70 judge+clean,
kept 50 (replacing the porpoising set before the render window). PRINCIPLE for the course
family: under pure min-snap (kT=None), NO segment may carry more than ~2.5 s of slack relative
to its length — corridor variance comes from the TUBES (jitter_m), never from schedule slack;
slack converts to out-of-plane polynomial excursions, not meander. regen2_demos page
republished (artifact 6c731007) with the fixed CFR fan.

**gmsig2 GATE-FAILED FOR REAL — THE HOVER-SKEWED regen1 DATA DESTROYS THE MDN HEAD
(2026-08-23 ~17:50).** After fixing the joint_head norm-stats lookup (config-keyed asset path
404s for non-gate_nav checkpoints — the first GATE_FAILED was that bug), the true readout:
c-R2 +0.05/+0.09/+0.05/-0.02 across tasks, CFR |c| err 5.89 — no head formed. Mechanism
consistent with the training anomalies (loss NEGATIVE at -0.27, grad_norm 2588-2747 vs
gmsig's ~100/3.2): regen1's ~60% stationary-hover rows let sigma collapse toward the
deterministic stop signature, NLL runs degenerate-negative, and the unstable gradients wreck
the conditional. DECISIVE data-pathology negative: the pacing bug is not a skew but a killer
for NLL heads. Gate stopped the line per the operating rule; zero rollouts spent. gmsig2
closed; the pipeline proceeds directly to gate_nav3 (fixed pacing) + gmsig3.

**gmsig3 GATE OVERRIDDEN BY DENIS (2026-08-24): the pooled-R2 bar is miscalibrated for
tube-era data — sigma-probe shows the head is the BEST-CALIBRATED yet (corr 0.874 pooled /
0.929 tail) with TAIL errors at healthy-gmsig levels (L 3.0 / R 1.95) while early/mid errors
are the injected tube noise, priced ~1:1 by sigma. Six cells launched under the trust dial;
proposed future gate criterion for tube-era arms: tail-region c-R2 or sigma-calibration >=0.8
instead of pooled c-R2 >0.5.**

**gmsig3 FULL ROW (2026-08-24 ~05:30): 40/40 JUDGE SUCCESS ON ALL FOUR TRAINED TASKS — 39/40
STRICT-TIER — THE FIRST ARM TO OWN EVERY SINGLE-TASK CELL, WITH A GROUNDED COMMAND SOURCE.**
Six cells under the trust dial (own calibration map, mh16, gate overridden per Denis after the
sigma-probe showed tube-noise-only R2 dilution): LEFT 10/10 strict; RIGHT 10/10 STRICT — the
right settle CLOSED (endpoints (1.43,-0.40,0.99)+/-0.04, IN-BOX; the cell no prior arm's family
ever completed while keeping the others); CFL 10/10 strict; CFR 10/10 judge, 9/10 clean ->
9/10 strict. L/R clearances mean 0.35, worst 0.267 — comfortably demo-band. sigma_serve
deciles 0.10/0.18/0.85: the dial ran mostly HIGH-TRUST (vs gmsig's 0.16/0.64/0.96) —
closed-loop states now look on-manifold to the head, i.e. the start-variance + pacing fixes
removed most of the covariate shift the old dial was pricing. Compounds 0/5 (no compound
demos; composition remains the open frontier — and note gmsigs7's compound ability was on the
OLD data). COMPARISON TO THE RECORD BOARD: the historic 39/40 (2026-08-05) was the ONE-HOT
scaffold with no VLM in the loop and LEFT 9/10; gmsig3 is 40/40-judge/39/40-clean-join with
the GROUNDED language command source — the north-star replacement criterion (beat the one-hot
scaffold with semantics) is, at single-run tier, MET. EVERYTHING converged here: regenerated
data (return corridors, pacing, real-matched starts, two-class tubes), the MDN head, sigma
conditioning, and the calibrated trust dial. CLAIM LADDER: single training run/seed — seed-7
replication + video review required (L/R videos exist: overlay_armgmsig3_*; center cells were
VIDEO=0 — claim tier needs a video re-fly of CFL/CFR).

**COMPOUND DEGRADATION ROOT-CAUSED (Denis hypothesis confirmed with mechanism, 2026-08-24):
CLEAN CORRIDORS MADE STATE A SUFFICIENT STATISTIC — LANGUAGE CONDITIONING ATROPHIED OFF-START;
THE TRUST DIAL IS INNOCENT.** gmsig3 compound trials: both cells execute the ATOMIC task to
completion and hover (CMPL: left gate step ~60, goal by 350, ignores the center continuation;
CMPR: right-task route, wide gate-1 crossing uncredited, then goal). CLOG at the prompt
switch: the head commands the LEFT-TASK RETURN (SW, descend->goal) at the post-gate state
under the compound prompt, with HIGH trust (sigma_serve 0.14-0.25, sigma* 4-6 near goal) —
confidently wrong, not dial-suppressed. MECHANISM: gate_nav3's well-separated corridors give
every mid/late state exactly ONE in-data continuation -> state fully determines the future ->
the NLL-optimal conditional drops its language dependence anywhere off the start. The OLD
data's overlapping/sloppy corridors kept states ambiguous, FORCING language-dependence — which
is what let gmsigs7 re-route mid-flight (seed-permitting). WE TRADED COMPOSABILITY FOR
CORRIDOR CLEANLINESS. Options (Denis to pick): (1) compound demos in the next cut (courses
through_{left,right}_and_center.yaml exist with the same perturbation machinery; keep one
compound HELD OUT for the zero-shot composition claim); (2) deliberate corridor OVERLAP —
shared mid-states with prompt-dependent continuations, forcing language-dependence without
compound demos (truer to the north star's compose-from-atoms claim, subtler data design);
(3) head-side language pressure (prompt-dropout/CFG — previously parked as bandaid;
contrastive rejected). Confirmatory probe queued for GPU-free time: prompt-sensitivity
(same frame, both prompts, |dc|) gmsig3 vs gmsigs7.

**TEXT-ADHERENCE LINE OPENED (Denis correction, 2026-08-24): adherence is a MODEL property —
the corridor-redesign path is set aside; the fix goes into the head's training.** Denis's
read: the VLM should resolve the compound problem; ensure text adherence and prevent
overfitting from collapsing the conditional into the state regime. Two pieces built + dummy-
verified: (1) text_adherence_probe.py — same frames, all 6 prompts (4 atomic + 2 compound),
adherence = mean pairwise |dc|/cstd per phase; becomes an acceptance metric alongside the
readout gate (floor on mid-flight adherence). (2) SNMVP_HEAD_COND_DROP=p in pi0.py — per-
sample, the head's STATE and IMAGE branches are zeroed with prob p while language is ALWAYS
kept: on dropped samples language is the only route discriminator, so a live language->command
pathway is forced even where state is a sufficient statistic (the anti-regime-collapse
pressure; also implicitly learns p(c|lang-only), leaving a CFG-style language sharpener
available later — per Denis's standing rule, only ever as a final sharpener). QUEUED behind
the scratch chain: adherence baselines for gmsig3 vs gmsigs7 ON THE SAME synth3 frames
(prediction: gmsig3 ~0 off-start, gmsigs7 > 0 — the direct test of atrophy-vs-thesis), then
gmsig4 = gmsig3 recipe + COND_DROP 0.25 on unchanged gate_nav3 — if gmsig4 composes on
SEPARATED corridors, text adherence is restored model-side and the clean data stays.

**DROPOUT DESIGN REVISED PER LITERATURE (Denis, 2026-08-24): correlated state+image drop
replaced by INDEPENDENT per-channel Bernoulli — state 0.4 / image 0.15 / language 0.1.**
Grounding: causal confusion (de Haan et al. NeurIPS'19 — more information can hurt; the
informative channel confounds) and the copycat line (Wen et al.'20; residual-prediction
ECCV'22) treat our state-sufficiency exactly, with aggressive dropout of the shortcut channel
the standard mitigation; modality-dropout practice (ModDrop/MUTEX) says INDEPENDENT drops so
every subset pathway trains; image gets a mild drop or the position shortcut migrates into
the pooled pixels (0.15-0.2 m localization, box finding); language dropped 10% CFG-style to
learn the null-language branch = the future language-adherence sharpener's handle (sanctioned
as final-sharpener only). Legacy single-float spec kept for compat. gmsig4 queue updated to
"0.4,0.15,0.1"; adherence baselines (gmsig3 vs gmsigs7, same synth3 frames) still precede it.

**SCRATCH CONTROL ON gate_nav3 (2026-08-24 ~17:45): 36/40 JUDGE — THE HISTORIC SCRATCH
COMPLETION FAILURE WAS SUBSTANTIALLY A DATA ARTIFACT; THE PIN'S EDGE ON THIS DATA IS
PRECISION, NOT CAPABILITY.** Plain pi0 (no pin/head, plain-served): L 9/10, R 10/10, CFL 9/10,
CFR 8/10 judge; clean-join ~30/40 (CFR 4/10 clean); compounds 0/5 both (5/5 clean). vs gmsig3
40/40 judge / 39/40 clean-join / 0-0 compounds. READS: (1) the regenerated data (corridors,
pacing, start variance, tubes) lets a plain baseline nearly saturate n=50 single-task
completion — the old "transits 20/20, completes 0/20" chasm does not survive good data;
(2) the pin+dial's measurable margin here = +4 judge points and a LARGE cleanliness gap
(esp. CFR 9/10 vs 4/10 clean) — precision/consistency, honestly smaller than the historical
framing; (3) compounds fail architecture-independently — language adherence is the
differentiator and lives in the VLM conditioning (supports Denis's model-side direction).
CONSEQUENCE for the north star: the discriminating benchmarks are now the LOW-DATA ladder,
STEERABILITY, and COMPOSITION/adherence — n=50 single-task is saturated. Queue proceeds:
adherence baselines then gmsig4 (systematic dropout).

**ADHERENCE BASELINES CONTRADICT THE CORRIDOR-ATROPHY STORY (2026-08-24 ~18:10) — REFINED
HYPOTHESIS: PROMPT-NEIGHBORHOOD COLLAPSE, NOT SENSITIVITY LOSS.** Same synth3 frames, 6
prompts, adherence = mean pairwise |dc|/cstd by phase: gmsig3 = 0.141/0.170/0.260/0.058,
gmsigs7 = 0.112/0.094/0.119/0.070 — the clean-data arm's language sensitivity is HIGHER,
including mid-flight where the atrophy story predicted collapse (caveat: synth3 frames are
OOD for gmsigs7, depressing its numbers — but that cannot rescue the story, only level it).
Denis's skepticism vindicated by measurement; the earlier root-cause entry stands CORRECTED:
corridor separation did not kill prompt sensitivity. Refined hypothesis: the compound prompts
are NOVEL text (absent from gate_nav3's four task strings) and the head maps them onto their
nearest trained neighbor (the atomic prompt) — so the decision-relevant contrast
dc(compound-left, left) ~= 0 specifically, while left-vs-right contrasts stay healthy and
carry the pairwise average. TEST: pair-resolved adherence probe (per prompt-pair per phase),
queued for GPU-free time after gmsig4's training. FIX MENU under the refined hypothesis:
language-side variety in training (paraphrases/composite instructions) and/or the CFG language
sharpener amplifying the small compound-vs-atomic contrast; the dropout arm (gmsig4, training)
still tests the channel-dominance component.

**PAIR-RESOLVED ADHERENCE (2026-08-25 ~00:30): NO PROMPT-NEIGHBORHOOD COLLAPSE — LANGUAGE
CONTRASTS ARE UNIFORMLY SMALL AT ALL STATES IN BOTH ARMS, AND DROPOUT DID NOT CHANGE THAT;
THE MISSING INGREDIENT IS SERVE-TIME AMPLIFICATION.** gmsig4 sigma: calibrated like gmsig3
(0.844 pooled / 0.924 tail; tail err L 3.14 / R 2.09) — same healthy profile, same gate-bar
artifact (min c-R2 0.32 at CFR). Pair table (|dc|/cstd, start vs switch states): gmsig3
left-vs-right 0.064/0.068, left-vs-cmpL 0.084/0.084; gmsig4 0.087/0.071 and 0.052/0.082 —
(1) the compound prompt is AS DISTINCT from its atomic prompt as the atomic prompts are from
each other: the neighborhood-collapse hypothesis is DEAD; (2) all language contrasts are
~0.05-0.09 cstd (~1.5-2 c-units) EVERYWHERE — including at the start, where that same small
nudge demonstrably suffices to route 10/10 correctly. So the switch failure is not contrast
size per se: at the START both continuations are reachable and the flow amplifies a 2-unit
command nudge into route choice; at the POST-GATE state the atomic-return attractor (state
manifold + vision) plus PARTIAL TRUST at the switch (sigma_serve 0.49-0.76 there, from the
compound clog) swamp the same-size nudge. (3) COND_DROP did not increase language leverage
(0.082 vs 0.084) — the causal-confusion treatment alone is insufficient; but gmsig4's 10%
language-dropout DID train the null-language branch, which enables the directly-indicated
mechanism: the CFG LANGUAGE SHARPENER at serve, c_guided = c + w*(c - c_nolang), amplifying
exactly the contrast that is present-but-small. Proposal: override gmsig4's gate (same case
as gmsig3), fly its six cells, then a compound SCREEN with language-guidance serve at w in
{2,4} — the sanctioned final-sharpener role, and the first mechanism the measurements
actually point at.

**CFG LANGUAGE SHARPENER RECOVERS COMPOSITION (2026-08-25 ~02:30): w=4 GIVES CMPL 3/5 AND
CMPR 3/5 BOTH-GATES+DWELL — ZERO-SHOT COMPOUND COMPLETION ON CLEAN DATA, NO COMPOUND DEMOS,
PURE SERVE-TIME DIAL.** Implementation: mu_guided = mu + (w-1)(mu - mu_nolang) per component
(null-language branch from gmsig4's 10% prompt dropout; pi/sigma/selection/hysteresis/trust
dial unchanged); verified linear (w2 delta 5.98, w4 17.95 c-units vs the raw ~2-unit
contrast). Screen: w=2 0/5+0/5 (insufficient vs the atomic-return attractor); w=4 3/5+3/5 —
CMPR had NEVER exceeded 1/10 for any arm in project history. Denis's model-side thesis
vindicated end-to-end: the VLM carried the signal; adherence needed serve-time amplification,
not data ambiguity. Costs at w=4: clearance (cmpl 0/5, cmpr 2/5 clean — extrapolated commands
fly hot near frames); 5-trial screen tier. NEXT KNOBS: w in {3,5} sweep with per-trial
geometry; phase/disagreement-gated guidance; gmsig4 atomic row incl. a w=4 atomic-regression
check. The guidance stays within the standing rule: a final sharpener on a measured deficit —
and it is the first one the measurements themselves demanded.

**CORRECTION — ROUTE-CLEAN SCORING RULE VETOES THE CFG CMPL RESULT AND THE OLD-DATA CFR/CMPL
ROWS (2026-08-25, from Denis's read of the point cloud).** Denis spotted in the cfg_compounds
viewer that the w=4 CMPL flights fly to the goal FIRST and then thread the center gate from
the wrong side. Event forensic confirms: all three judged CMPL "successes" run left gate ->
goal box (~t130, atomic attractor wins first) -> BACKWARDS pass through the center aperture
(+y, x 2.4-2.8, inside the hoop) -> U-turn behind the gate -> forward (-y) crossing which the
judge latched. `check_directional_transit` already counts these as `wrong_crossings` but
`judge_compound` discarded the field (same bug class as the region-box aperture gap). RULE
UPGRADE in `gate_success.py`: success (atomic AND compound) now additionally requires ZERO
wrong-direction aperture passes over the whole flight. Sanity: all synth3 CFL/CFR demos score
wrong=0 unanimously — the rule never vetoes ground truth. Full re-score of every stored
trajectory:
- **cfg4g4 (w=4 guidance): CMPL 3/5 -> 0/5 — composition NOT recovered on the left compound.
  CMPR 3/5 STANDS (wrong=0; flights 4,5 direct right->center t~66->t~138, flight 2 loiters at
  goal then crosses clean — direction-correct because CMPR's center crossing (+y) points away
  from the goal side, so greedy gate-pull is route-correct there by geometry.** Revised claim:
  the CFG sharpener fixes GATE SELECTION but not ROUTE TOPOLOGY — CMPL's correct crossing
  needs the around-the-far-side detour, and the guided pull aims straight at the hoop.
  Topology is the third oracle ingredient (selection / aim / topology) and remains unlearned.
- **Old-data (gate_nav) rows collapse under the same rule: gmsig CFR 10/10 -> 0/10; gmsigs7
  CFR 6/10 -> 0/10 (all "crazy stuff before the gate" = hoop oscillation, now quantified);
  c10gmsigs7 CMPL 10/10 -> 3/10; gmsigs7 CMPL screen 5/5 -> 1/5; the lone historical CMPR
  successes -> 0.** The old-data center-task and compound claims are WITHDRAWN at the strict
  tier.
- **gmsig3 (new data) is UNTOUCHED: all 40/40 atomic flights route-clean (wrong=0), scr3
  38->37/40.** The flagship row survives; the route-clean rule sharpens rather than weakens
  the new-data story: clean data produced clean routes, old data produced hoop oscillation
  that legacy scoring counted as success.
Scoring rule now: transit judge + route-clean (wrong=0) + clearance + human video. Videos for
the vetoed rows were never reviewed (VIDEO=0 screens) — the cloud page did the veto job here.

**CORRECTIVE SKETCH PROMPTING: CMPL 5/5 ROUTE-CLEAN — FIRST LEFT-COMPOUND COMPLETIONS EVER
(2026-08-25, human-element line, Denis's rung-2 direction).** Mechanism: a coarse polyline
covering ONLY the switch segment (6 waypoints off the CFL demo corridor, 2.2 m) is resampled
at demo speed, its per-step deltas projected through U — the sketch speaks the head's own
c-language — and served at sigma=0 while a per-trial state machine is ACTIVE (nearest-point
activation on the whole polyline, forward-monotonic progress, handback requires ARRIVING at
the sketch end); language swaps to the second atomic at activation; the head + trust dial own
everything outside the segment. `SketchPrompt` in serve_gate_pin_joint.py
(SNMVP_PIN_PROMPT=json), sketchpad UI in viz/build_sketchpad.py. Result on gmsig3 (which is
0/5 unguided and 0/5 route-clean even with CFG w=4): gates latched t~60/t~190, correct -y
center crossing, zero wrong-direction passes, dwell 415-455 — 5/5. Clearance 1/5 clean, but
every graze (0.10-0.14 m) is at the LEFT-GATE crossing t~54-62, BEFORE activation — inherited
from the compound-prompt crossing, not caused by the sketch; the sketch-driven portion is
clearance-clean. First screen was 0/5 from three serve bugs (premature radius-to-first-point
activation across the gate frame; progress cap below the ~50-step replan stride -> window lag
-> overshoot -> OOD handback; trigger flaky vs the ~1.25 m replan sampling), all fixed and
replay-verified — the state machine details are load-bearing. Claim tier: 5-trial screen,
VIDEO=0, single sketch; the sketch is demo-corridor-derived (rule-clean but machine-made) —
Denis's hand-drawn full-route sketches (Sketchpad artifact) are in the air next. The
factorization thesis passed its literal test: a kinematically naive polyline, projected into
the source-noise command subspace, is executed as a feasible flight by the denoising
residual — "coarse from the prompt, residual from the flow".

**HAND-DRAWN FULL-ROUTE SKETCHES COMPLETE BOTH COMPOUNDS (2026-08-25): CMPL 5/5 route-clean +
5/5 CLEARANCE-CLEAN; CMPR (after one waypoint repair) 5/5 route-clean, 3/5 clearance-clean.**
Denis drew both routes in the Sketchpad UI (full route from the start box — the sketch owns
the flight from replan 0 at sigma=0, head takes over only for the goal hover). CMPL: gates
t~48/t~143, min clearance 0.23-0.26 everywhere — the human's mid-aperture left-gate crossing
also cured the graze the corrective sketch inherited from the compound-prompt crossing; first
strict-tier-worthy left compound (video pending). CMPR first attempt 0/5 with a decisive
forensic: flights tracked the polyline to ~7 cm but the polyline itself passed 25 cm east of
the compound-scene right-gate aperture (x-span [0.11,0.47]; waypoint at 0.578) — CLICK
PARALLAX in the sketch UI (oblique-camera ray onto the z-plane), fixed with a top-view
drawing mode + warning. r1 (single waypoint moved to the aperture midline, all else Denis's):
5/5, gates t~57/t~178, dwell 138-416; grazes: one center-west 0.109 (his line passes
x 2.62-2.67, flights drift ~0.15 west), one right-gate 0.171. THE READING: pin execution
fidelity (~7 cm) exceeds human drawing accuracy — sketch success is limited by the UI, not
the flow; and the CMPR cell, never above 1/10 autonomous and 3/5 under CFG, is 5/5 under a
ten-second human sketch. Screen tier: 5 trials, VIDEO=0, single sketch per cell; claim tier
needs >=10 + video + (flywheel next) sketch-reuse stats across start positions. Artifacts:
Sketchpad (dd2624b2), results page. Assisted rows — never mixed with autonomous.

**MINIMAL-SKETCH STUDY (2026-08-25, Denis's question: "start, gate apertures, ending — does
it do everything else on its own?"):** 4-6 point sketches (start + gate midpoints pushed
0.2 m through the plane + end), straight lines between. ANSWER SPLIT: the flow supplies ALL
dynamics (corner rounding, speed, settle) but NO free-space reasoning at sigma=0 — it tracks
the drawn line ~7 cm even into a post. Results: CMPL min4 sigma=0: 5/5 route-clean, 0/5
clearance (straight diagonal shaves the center west post 0.03-0.15). CMPL min4 sigma=0.5:
4/5 route + 4/5 clearance — the trained trust dial buys back the finesse (wider slower
crossing, t~160 vs ~125) at the cost of one route capture. CMPR min4: 0/5 both sigmas — the
gate1->gate2 diagonal pierces the aperture 4 cm from the west post (a LINE error no trust
level fixes); at sigma=0.5 even the right gate is lost: the CFR-prompted prior wants to skip
it and slack surrenders the override. **SIGMA LAW: slack helps where sketch and prior agree
(free-space finesse), destroys the sketch exactly where it must override the prior.**
CMPR min5 (one staging point (2.75,-0.9) so the line pierces mid-aperture) sigma=0: 5/5
route-clean, 1/5 clearance (grazes moved to the narrow right gate — the sharp NE exit turn;
Denis's r1 exit angle cleared it). ECONOMY: 4-5 clicks per compound suffice for route-clean;
clearance needs either the human's exit-angle instincts (r1: 3/5-5/5 clean) or per-waypoint
sigma (proposed: sigma=0 at prior-conflicting gates, loose in corridors — JSON+UI extension,
not yet built). All screen tier, 5 trials, VIDEO=0.

**CORRECTION — RIGHT_AND_CENTER GATE_1 APERTURE WAS A HALF-WIDTH BOX ON THE WEST POST
(2026-08-26, Denis caught it in the results-page overlay).** The compound safety YAML's
gate_1 corners ([0.47,-1.24]/[0.11,-1.39]) covered only s in [-0.10,0.29] of the physical
opening (posts measured from the scene cloud at s=-0.09/+0.82 — matching the atomic
right_gate.yaml post-centre corners exactly). Fixed to the atomic corners; region-box bug
class strikes again, this time in a SAFETY file. Full cmpr re-score under the fixed box:
- **Denis's round-1 hand-drawn sketch (skd): 0/5 -> 5/5 ROUTE-CLEAN.** The flights crossed
  the real gate mid-opening exactly as drawn; the "25 cm click-parallax miss" story is
  RETRACTED (the parallax mechanism is real and top view stays, but it did not cause this).
  The r1 "repair" was chasing the buggy box — it steered flights toward the west post,
  which is where r1/min5's right-gate grazes came from.
- **min4 sigma=0.5 (skm4s): right gate was never abandoned** — flights crossed its east
  half, outside the buggy box. SIGMA LAW REVISED: slack buys clearance finesse and costs
  some route capture (CMPL sigma=0.5 dropout); the "slack surrenders prior-conflicting
  overrides" clause had only the buggy-box evidence and is withdrawn to hypothesis status.
  Both min4 variants fail CMPR at the CENTER crossing for line-geometry reasons (pierce
  point 4 cm from the post) — sigma-independent, fixed by min5's staging point.
- Unchanged: cfg4g4 3/5; unguided/ctl/scr3 etc. 0/5 (gate1-only); old-data seed-7 compounds
  latch both gates under the fixed box but all carry wrong-direction passes -> still 0.
Sketch-line tally after correction: CMPL hand-drawn 5/5+5/5, CMPR hand-drawn (round-1,
unrepaired) 5/5 route-clean 1/5 clearance, CMPR r1 5/5+3/5, CMPL min4 5/5 (sigma-0) /
4/5+4/5 (sigma-0.5), CMPR min5 5/5+1/5.

**ROLLOUT-SEED REPLICATION OF THE SKETCH ROWS (2026-08-26, SNMVP_NOISE_SEED=1 — fresh
residual-noise stream, the only stochasticity in the deterministic sim):** hand-drawn CMPL
5/5 route + 5/5 clearance AGAIN (min 0.225-0.248); hand-drawn CMPR (original round-1 sketch)
5/5 route again; CMPR min5 5/5 route again. Pooled 10-trial tallies: CMPL 10/10+10/10, CMPR
10/10 route (1/10 clearance), min5 10/10 route (1/10). Clearance GRAZES REPLICATE IN PLACE
(same locations/magnitudes both seeds) — systematic sketch geometry, fixable by waypoints,
not rollout lottery. One new outlier: min5 ns1 flight 3 dips to 0.006 m near the right gate
at z 0.82 POST-HANDBACK (dwell 34) — first post-handback wander; video candidate. Training-
seed rep (gmsig3s7, --seed=7) now training; six-cell table to follow.

**TRAINING-SEED REPLICATION LANDS: gmsig3s7 (--seed=7, identical recipe/data) REPRODUCES THE
FLAGSHIP — 40/40 ROUTE-CLEAN JUDGE ON ALL FOUR ATOMIC CELLS (2026-08-26).** Left 10/10
(clearance 10/10, min 0.24-0.39), right 10/10 (10/10 clean, 0.20-0.24), CFL 10/10 (10/10
clean, 0.34-0.39), CFR 10/10 judge (7/10 clean; three grazes 0.11-0.16 at the center west
post z~1.3 during the goal descent — same signature family as seed-42's one strict miss,
within protocol noise). Compounds 0/5+0/5 unguided — also replicating seed-42 (the
composition gap is structural, not lottery). Offline replication chain: per-task readout R2
profile matches (min 0.40 vs 0.32), sigma-probe pooled corr 0.848/tail 0.941 (vs
0.844/0.924), sigma map knots within a few percent. **POOLED TWO-SEED CLAIM: 80/80
route-clean judge on gate_nav3 atomics, 77/80 clearance-clean — the sigma-conditioned GMM x
mh16 arm is seed-robust; two-tier statistics rule satisfied for the atomic claims (>=10
trials x 2 training seeds). Remaining for record-board strict tier: human video review
(left/right reels exist: overlay_armgmsig3s7_*).** Ops note: the chain broke once at
make_sigma_map's stale data_gate_synth default (crashed post-probe; --data-dir now
hardwired in both post scripts) and was resumed cells-only (resume_gmsig3s7_cells.sh).

**SKETCH SEED-PORTABILITY (phase A of seed-rep round 2, 2026-08-26): min4 CMPL sigma=0
replicates EXACTLY on the seed-7 checkpoint (5/5 route, 0/5 clearance, same west-post shave
coordinates); min4 sigma=0.5 replicates the trade direction (3/5 route + 5/5 clearance vs
4/5+4/5). min5 CMPR degrades (5/5 -> 3/5, near-hits 0.004-0.04) — root cause NOT seed
lottery: min5's gate-1 waypoint (0.29,-1.315) was placed at the BUGGY half-box midpoint,
steering flights through the west third of the real opening; seed-42 habits cleared the post
by centimeters, seed-7 drifts ~10 cm west and hits it. The aperture-bug contamination reached
the sketch geometry itself. sketch_cmpr_min5f.json fixes the waypoint to the true aperture
midline +0.2 m through ((0.655,-1.326), computed from post centres); to fly on both
checkpoints when the scratch-s7 chain frees the GPU. Sketches with real margins (hand-drawn,
min4 CMPL) are checkpoint-portable; sketches whose lines run near structure inherit
per-seed flow drift — margin IS the portability budget.

**SCRATCH SEED REP (gate_scratch3s7, --seed=7, plain pi0, same data, 2026-08-26): 37/40
route-clean judge (left 10/10, right 10/10, CFL 10/10, CFR 7/10; clearance 32/40).** Pooled
scratch two-seed: 72/80 judge vs the pin arm's 80/80. Scratch replicates its own profile —
including the consistent CFR weakness (7/10 BOTH seeds) — so the comparison is seed-robust in
both directions: data quality gives scratch the 7-10/10 band per cell; the pin arm's sweep of
every cell on every seed (and its CFR ownership) is the seed-stable margin. Compounds 0/5
both scratch seeds (as with the pin arm unguided). Old-data checkpoints gmm/gmmmh/ctl deleted
for disk (claims withdrawn, retrainable, rollout evidence kept); min5f corrected-sketch cells
launched on both pin checkpoints.

**MIN5F CLOSES THE MINIMAL-SKETCH LOOP (2026-08-26): with the gate waypoint at the TRUE
aperture midline, the 5-point CMPR sketch flies 5/5 route-clean on BOTH training seeds
(seed-42: 2/5 clearance-clean; seed-7: 1/5). The right-gate near-hits are gone entirely —
remaining grazes are all the familiar center-west-post goal-descent signature (0.12-0.18 at
(2.25-2.30,-0.1..-0.15)), the same family as CFR's atomic grazes: the one systematic
imperfection of the gate_nav3 line, present with or without sketches. Seed-7 dwells are
shorter (22-55 vs 243-439; threshold 16) — its hover settles less deeply; screen-tier note.
MINIMAL-SKETCH CONCLUSION: 4 clicks (CMPL) and 5 clicks (CMPR) produce route-clean compound
flight REPLICATED ACROSS TWO TRAINING SEEDS, provided waypoints near structure are placed on
true geometry with margin — margin is the portability budget.**
