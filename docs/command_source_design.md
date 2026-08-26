# Where the command source gets its features

Status: **decision recorded 2026-08-11.** Frozen encoder is a TEMPORARY scaffold; joint training is
the target. Written after the feature-source skew bug (see `docs/RESEARCH_LOG.md`, 2026-08-11).

## The problem this solves

`c` is predicted by a command source that consumes a VLM feature. That feature is a function of the
VLM's weights, so it is an artefact **stamped to a checkpoint**. Fine-tuning the VLA moves the VLM as
a side effect — even LoRA: between two of our own checkpoints, 18–22% of the embedding dimensions
moved beyond 3σ of each other. A cache extracted under checkpoint A and consumed under checkpoint B
feeds the command source a representation it never learned against.

That is not hypothetical. The gate language priors were fit on features from
`gate_both_pin_rrr/4999` and served on `gate_pin_zeropad/4999`. Cost: the enumeration-free command
source scored **0/10** closed-loop with an offline c-R² of 0.94, and pairing it back with its own
checkpoint took the left gate to **10/10 clearance-clean** with no retraining.

It hid for months because offline evaluation replays the same cache, so train and test agree with
each other; the disagreement lives only between the cache and the live server, and nothing compared
those two.

## What does and does not depend on the VLM

| artefact | depends on VLM weights? | consequence |
|---|---|---|
| pin basis `U` | **No, in practice** | RRR ≈ PCA of the action chunks (4 of 5 directions within 0.2°), because the chunk is ~91% predictable from features per task. `U` is fixed by action statistics, which do not drift. |
| flow's pin training | **No** | training extracts `c = Uᵀa` from the ground-truth chunk (`_snmvp_extract(actions)` in `patches/openpi_arm_c_training.patch`) — never from the command source. |
| command source | **Yes, entirely** | its input *is* the feature vector. |

So the coupling is one-way: flow → features → command source. There is no circularity, only an
**ordering constraint**:

1. choose `U` from action statistics (no VLM),
2. train the flow with oracle `c = Uᵀa`,
3. *then* extract features from the trained flow and fit the command source.

Nothing can go stale, because nothing consumes features before training finishes. Follow this
ordering regardless of which option below is in force.

## Option A — frozen encoder (TEMPORARY, in force now)

The command source consumes a **frozen** encoder — base π0's VLM, or a standalone PaliGemma/SigLIP —
instead of the fine-tuned flow's.

Why it is worth doing now:
- the cache never goes stale, so no re-extraction per training run;
- the command source is trained **once** and reused across flows, seeds and embodiments;
- no artefact touches the fine-tuned VLM any more (the basis already does not);
- it matches the intended architecture: an upstream module emits the command, the flow sits
  downstream of it.

What we give up: features that are not task-adapted. For this job that may not hurt and may help —
the command source needs task identity plus coarse geometry, and a generic encoder has not been
specialised by a 5k-step LoRA on four gate tasks, which is exactly the specialisation that would hurt
on an unseen instruction. **This is a claim to test, not to assume**: the featfix arm gives the
fine-tuned-feature number and a frozen-encoder arm gives the comparison directly.

**Why it is temporary:** a frozen generic encoder cannot benefit from anything the VLA learns about
this embodiment or these scenes. If the command source ever needs representations shaped by
task experience — and the movement-vocabulary goal suggests it will — it has to read the trained
model. Option B is how we get there without reintroducing the bug.

## Option B — joint training (the target)

Train the command source **inside** the flow training loop, as a parallel head.

The flow's train step already runs the full forward pass, so the post-fusion language-token pooling
is available in-loop at nearly zero extra cost: the extraction that currently takes hours as a
separate pass is free when done during training. And `c = Uᵀa` is already computed each step for the
pin, so the head's target needs no new machinery.

**The structural prize is not speed.** The head is saved *into the flow's checkpoint directory*, so
the command source and the flow it was fitted against ship as one object. The mismatch becomes
impossible by construction rather than merely detected by a stamp.

### Gradient coupling — the one real design choice

- **B1, detached (default).** The head trains on `stop_gradient(features)`. Equivalent to online
  extraction: staleness is solved, and the VLA is untouched, so flow quality cannot regress. Cheap
  and safe; this is what to build first.
- **B2, coupled.** Let the head's loss backprop into the VLM with weight λ, so the representation is
  actively shaped to make `c` predictable. This is scientifically the more interesting variant: it
  turns *"pick the action subspace that happens to be predictable"* (RRR, which on our data
  degenerates to PCA and so selects nothing) into *"make the representation predict the subspace"*.
  It also makes the choice of `U` matter even less. Risk: it changes the VLA's representation, so it
  needs a λ sweep and a hard gate that the flow loss does not regress against λ=0.

### Handling the moving target

While the flow trains, the features drift, so early head updates are fitted to representations that
no longer exist. Options, cheapest first: train the head only over the final fraction of steps; keep
training it after the flow's LR has decayed, when features are nearly stationary; keep an EMA of the
head; or time-stamp buffered features and downweight old ones. Start with "train throughout, then
re-fit the head over the last ~10% of steps" — simple and it makes the final head match the final
weights.

### What must NOT be changed

The flow's pin keeps using **oracle `c` from the ground-truth chunk** during training. Do not feed
the head's *prediction* into the pin: the flow would learn to expect the head's errors and the two
would co-adapt around them. (Deliberately training the flow on predicted `c` — scheduled sampling /
DAgger — is a separate later experiment with its own justification, not part of this.)

### What joint training does NOT fix

Staleness is one of two failure modes measured on 2026-08-11. The other is **covariate shift**: the
command source errs ~0.49 m at states near the demo manifold and drifts 0.26–0.51 m off it in flight,
while the one-hot scaffold stays within 0.05 m and errs 0.06–0.08 m. Joint training does nothing about
that. It needs its own fix — state/render-perturbed training rows, or DAgger from rollouts — and
should not be folded into this work or claimed as a benefit of it.

## Order of work

1. Now: Option A, clearly labelled temporary, plus the ordering rule above and the `feat_ckpt` stamp
   already enforced in `experiments/rung3/pin_basis.py`.
2. Measure: frozen-encoder command source vs the fine-tuned-feature one (featfix), same flow, same
   basis, same APC, ≥10 trials per side. This decides whether task adaptation in the command path is
   worth anything at all — and if it is not, Option A stops being temporary.
3. Build B1 (detached head in the train loop, saved into the checkpoint). Gate: matches the
   post-hoc-fitted prior at equal flow quality.
4. Only then B2 (coupled, λ swept), gated on the flow loss not regressing.
5. Separately, and independent of all of the above: fix the covariate shift.
