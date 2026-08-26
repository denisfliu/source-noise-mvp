# HANDOFF — source-noise action steering: status, results, architectures, resume guide

Single entry point for resuming this project later (by a person or a fresh
Claude). Written 2026-07-20. For method/math detail see `REPRODUCTION_GUIDE.md`;
for the cross-embodiment design see `cross_embodiment_plan.md` / `rung2_plan.md`;
for raw numbers see `experiments/*/README.md` and `findings/`.

Repo: `~/code/source-noise-mvp` on the EC2 box (ssh alias `ec2`). openpi at
`~/code/openpi` (commit 15a9616 + patches). Docs mirror locally to
`~/Desktop/snmvp-docs/` (+ `findings/` for results).

---

## 1. The project, and the one finding that ties it all together

We move a movement command out of a conditioning input and into the **source
noise** of a flow-matching action policy, so the training loss itself penalizes
disobedience (the pin). Built on an ICLR'26 paper (subspace phase-invariant
sources). Two motivations: steerable/interpretable control, and grounding
actions in a learned geometric frame that transfers across scenes/embodiments.

**The central, repeatedly-confirmed finding:**
> Source-noise grounding helps **only when few-demo/low-capacity learning
> from observation alone struggles** — i.e. when the base task is hard or
> perception is the bottleneck. When the observation already determines the
> action cheaply (easy task, ample data, near-ceiling vision), the pin adds
> nothing and its imperfect command source *subtracts* (the "obedience tax").

This showed up as: the LIBERO obedience tax (Phase 1), the toy needing obstacles
before transfer paid off, and the Rung 2 easy-reach null. **The target regime
for all future experiments is therefore: hard task + few demos + novel
embodiment — where scratch struggles.** That is where the method should win, and
where we have not yet run the decisive real-scale test.

---

## 2. Results summary (all honest, incl. negatives)

**Phase 1 — real π0 on LIBERO (`experiments/phase1/`, complete):** a coupling
spectrum. Both conditioning (arm B) and source-noise (arm C) capture the model
(command-less eval ~0%). C binds ~11x tighter under contradiction, fully
steerable, metrically calibrated (mm/unit); B is loosely coupled, unsteerable
against the scene. A (no channel) 90% success; C-with-prior ~50% (a ~40-pt
obedience tax from the ~25%-error prior); B ~76%. No held-out-placement gap for
any arm (the original H1 metric is void at this scale). Steerability is the
clean win.

**Toy mechanism (`experiments/toy/`):** pin executes contradictory commands ~26x
tighter than conditioning; diversity preserved. **CFG-style pin dropout kills the
channel** -> pin must be always-on -> inference always needs a command source.

**toy_frame — learned frame (`experiments/toy_frame/`, all gates passed):**
structure discovered by a coherence criterion (external to the flow loss) beats
a hand-defined invariant; **phase-only pins fail, pins must carry magnitude**
(control structure is partly metric); a scene->invariant prior supplies the pin
with no oracle: +17 pts held-out no-oracle success over baseline.

**Cross-embodiment toy Rung 1 (`experiments/toy_embodiment/`):** freeze a
coherence-learned frame + prior on a set of 2D bodies, adapt ONLY the executor on
a held-out body's few demos. **Transfer works: T > scratch AND T > random-frame
for every held-out body** (incl. a maximally-divergent point robot) — on a task
made hard by obstacles. Success is prior-limited (T-oracle >> T). **G-predict is
negative**: cross-body coherence does NOT predict transfer gain (controlled
sweep, concordance 0.5) — transfer rides on the energy-dominant shared structure
(the endpoint), which is embodiment-invariant by construction.

**Rung 2 real arms (`experiments/rung2/`, first result):** planar EE reaches
across Panda/Sawyer/IIWA/UR5e in robosuite (real cross-arm tracking differences:
demo success 90/64/76/89%). Freeze frame+prior on 3 arms, adapt executor on
held-out UR5e. **On easy free-space reach, transfer does NOT help — scratch beats
it at every n, and T ~= random-frame** (the learned frame does nothing). Exactly
the central finding: free-space reach is trivial for scratch, the only structure
is the endpoint (free from obs), and the cross-arm prior only adds error. -> Must
add obstacles (next experiment).

---

## 3. Architectures

### 3.1 The pin (source-noise invariant), shared by everything
Invariant L(a) = per-dim sum of the action-delta chunk = net displacement;
LINEAR in the action representation (required — only pin linear functionals).
Construction: ε̃ = ε + U(m̂ − Uᵀε), overwrite the pinned subspace of the noise
with the (normalized) command, complement untouched. Because L is linear and the
flow interpolant is x_t = t·ε̃ + (1−t)·a, the invariant is carried at every noise
level and the velocity target has zero invariant-component -> the loss enforces
it. Learned-frame variant pins temporal-Fourier phase (+ magnitude at
magnitude-coherent bins) of learned projections; discovered by cross-body/
cross-demo **coherence** with an energy floor + CV gating.

### 3.2 The three-part factorization (the cross-embodiment architecture)
1. **Front-half (FROZEN, embodiment-shared):** perception -> invariant. At real
   scale = VL trunk + a small readout head (or a standalone CNN+MLP prior, the
   D6 build). Produces the geometric goal in an embodiment-agnostic,
   object/goal-centric, task frame.
2. **The invariant = the API contract**, pinned into the executor's source noise
   (NOT a conditioning input — that is the weaker arm B).
3. **Executor (LEARNED, embodiment-specific):** flow head that realizes the
   pinned invariant as this body's actions. The ONLY part re-learned per body.
Cross-embodiment adaptation: freeze 1-2, re-learn 3 from few demos. Iterate
cheaply by caching the frozen front-half's invariants over the dataset once.
Design rule (validated): all bodies act in a **task-space (EE-delta) action
interface** so the invariant stays linear (pin exact); embodiment difference =
kinematics/reachability, not action parameterization.

### 3.3 Reusable code (the two pipelines)
- **Toy (2D, CPU, autograd), `experiments/toy_frame/` + `toy_embodiment/`:**
  `dataset.py`/`mb_dataset.py` (scenes, bodies), `coherence.py`/
  `coherence_xembod.py` (coherence maps, `align_to_consensus`, selection),
  `pin.py` (pin_noise, extract_phases/mags, preservation_check),
  `flow_embod.py` (executor: `make_loss`/`train_executor`/`rollout`; prior:
  `train_prior`/`prior_predict`/`build_shared_prior`; frame: `freeze_frame`),
  `battery.py` (transfer grid + gates). Key module globals: `fe.H` (horizon),
  `fe.OBS_DIM`, `fe.SET_A`.
- **Robosuite (real arms), `experiments/rung2/`:** `env_check.py` (shared
  EE-action interface across arms), `collect_demos.py` (scripted reaches ->
  per-arm .npz of EE chunks, SHARED targets so demo i is the same scene across
  arms), `rung2_transfer.py` (offline: loads .npz, reuses the toy `flow_embod`
  machinery via `fe.H=32, fe.OBS_DIM=2`, runs the transfer grid). NOTE the
  action-normalization step (ACT_SCALE ~ 1/mean|delta|) — raw ~0.006 m/step
  deltas underfit without it.

---

## 4. Next experiment (ready to build): obstacle reach

Add an obstacle to the robosuite reach so few-demo scratch struggles (the target
regime). Concretely: place a body/region between EE-start and target; script the
demo to arc around it; success = reach target AND no collision. Everything else
(collection -> coherence -> `rung2_transfer.py`) is reused unchanged. This is the
decisive real-scale test of the method: transfer should help here as it did in
the obstacle toy. If it does, escalate to gsplat perception + more arms (Rung 2
proper) then OXE/VLA (Rung 3). If it does not, the method's benefit may not
survive real kinematics even in the hard regime — a key negative to know.

Open design choices carried forward (see `rung2_plan.md` §6): the ultimate target
embodiment; whether the invariant readout is language-conditioned; gsplat's
capability (rendering-only vs physics-controllable).

---

## 5. Resume checklist

**Environment (box):**
- Toy: CPU only. `~/.local/bin/uv run --with autograd --with numpy --python 3.11
  python <script>.py`.
- Robosuite: `MUJOCO_GL=egl uv run --with 'robosuite==1.4.1' --with
  'mujoco==2.3.7' --with numpy --python 3.11 python <script>.py`. **Pin mujoco
  2.3.7** (3.x breaks robosuite 1.4.1: mj_fullM signature). Headless low-dim
  state (no rendering) for the first pass. Both GPUs currently free.
- Real π0 (Phase 1): openpi venv at `~/code/openpi/.venv`; patches in
  `patches/`; arms via env vars `SNMVP_PIN_ALPHA`/`SNMVP_PINNED_DIMS`/
  `SNMVP_COND_STATS`. ONE final checkpoint per run kept at
  `~/code/openpi/checkpoints/pi0_libero/*/14999` (+ reference
  `armA_baseline_s42/29999`); intermediate + duplicate finals pruned
  (1.2 TB -> 146 GB; box ~44% used, 1.1 TB free).

**Operational (SSH to the box is flaky — worked around, not fixed):**
- Any command may exit 255 on a dropped connection; **retry** — it always works
  eventually. Prefer short, single-purpose ssh commands.
- Launch long jobs via a **box-side script** (`run_*.sh` with `nohup ... &
  disown`), not inline `ssh "... &"` — inline backgrounded launches die on drop.
- **Never** put a kill pattern containing both `python` and the script name
  inline in an ssh command — `pkill -f 'python.*foo'` matches the ssh shell
  running it and kills your session. Put kills INSIDE a box-side script whose own
  cmdline lacks the pattern.
- `pgrep -f <script>.py` gives FALSE POSITIVES by matching your own ssh command
  string; treat a lone "RUNNING" skeptically — confirm with a DONE marker in the
  log or the result file's existence.
- Poll long remote jobs with a box-side wait loop (`for i in $(seq 1 40); do
  grep -q DONE log && break; sleep 5; done`) so the sleep runs on the box.

**Pitfalls (learned the hard way):**
normalize actions to O(1); only pin linear functionals; energy-floor the
coherence selection; use mod-π coherence for sign-flipping styles; pins must
carry magnitude for metric structure; keep the coherence criterion external to
the flow loss (else transcription); always-on pin only; match the prior's inputs
to the serving client; validate the prior by episode split; command-less eval of
an always-pinned model is 0% by construction.

---

## 6. Artifact map
- Method/plans: `docs/REPRODUCTION_GUIDE.md` (method + Phase 1 + toy_frame),
  `docs/mvp_plan.md` (original plan, partly superseded), `docs/
  openpi_integration.md`, `docs/cross_embodiment_plan.md`, `docs/rung2_plan.md`,
  `docs/toy_embodiment_plan.md`, decision/reply docs.
- Experiments/code + results: `experiments/{toy,toy_frame,toy_embodiment,rung2}/`
  (each has README.md + results/ or data/); `experiments/phase1/results/`.
- Local mirror of results: `~/Desktop/snmvp-docs/findings/`.
- Source paper: `unpaired_rerendering_subspace.pdf` (in the archive zip).
- Zip snapshot (through 2026-07-09): `~/Desktop/snmvp_archive_2026-07-09.zip`
  (predates cross-embodiment/Rung 2; regenerate if a fresh full archive is
  wanted).
