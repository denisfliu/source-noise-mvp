# source-noise-mvp

Test harness for the source-noise action steering MVP (see
`docs/mvp_plan.md`): carry a linear chunk invariant (summed action deltas) in
the source noise of a flow-matching action head, so the flow loss directly
enforces invariant adherence. Iterate on openpi π0 (LIBERO baseline has
headroom); confirm on π0.5 (config swap only). π0-FAST is autoregressive and
incompatible.

## Layout

```
src/snmvp/
  source_constructor.py   noise calibration (the core mechanism, ~150 lines)
  invariants.py           invariant extraction + dataset stats
  probes.py               wrong-invariant probe, adherence, diversity metrics
tests/                    carried-invariant property tests (numpy + torch)
docs/
  mvp_plan.md             full staged experiment plan
  openpi_integration.md   where to hook openpi, sanity sequence, Blackwell notes
scripts/
  setup_ec2.sh            setup confined to ~/code (clones openpi, uv env, tests)
```

## Quick start (EC2)

```bash
bash scripts/setup_ec2.sh        # everything stays under ~/code
```

Then follow `docs/openpi_integration.md`: hook `SourceConstructor` into the
openpi PyTorch loss + sampler and run the 4-step sanity sequence before any
real training.

## The mechanism in three lines

```python
noise = torch.randn_like(actions)                       # (B, H, D)
noise = SourceConstructor()(noise, actions.sum(-2))     # pin L(noise) = L(a0)
# now every x_t = t*noise + (1-t)*a0 satisfies L(x_t) = L(a0),
# and the flow target v = noise - a0 satisfies L(v) = 0
```

## Experiment arms (Phase 1)

| Arm | Invariant path | `SourceConstructor` |
|---|---|---|
| A | none | `alpha=0.0` |
| B | conditioning token | `alpha=0.0` + input token |
| C | source noise | `alpha=1.0` |
| D | both | `alpha=1.0` + input token |

Primary gate: arm C beats arm B on held-out LIBERO-Spatial placements AND
wrong-invariant follow rate >= ~80% for C (see `probes.follow_rate`).
