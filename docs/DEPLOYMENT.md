# Deploying a pinned policy (real-robot inference)

The inference half of the method has no sim, renderer, or experiment-script dependency:
`src/snmvp/deploy.py` (library) + `scripts/package_policy.py` (bundler). Everything an
inference host needs lives in one immutable directory.

## Build a bundle

    python scripts/package_policy.py \
        --ckpt <checkpoint dir containing params/> --config <openpi config name> \
        --norm <norm stats dir> --pin-u <U .npy> --prior <prior .pt> \
        --out /path/to/bundle_v1 --note "what this is"

It copies `params/`, `norm_stats/`, `pin_U.npy`, converts the torch prior to `prior.npz`
(numpy — no torch at inference), and writes `manifest.json` with a sha256 per artifact plus
config, action horizon/dim, and the source paths + git commit.

It refuses to write unless both checks pass:
1. the numpy prior matches the torch module (relative error < 1e-5 on inputs drawn from the
   prior's own input distribution);
2. the packaged bundle loads and produces a finite action chunk.

## Use it

    from snmvp.deploy import PinnedPolicy
    pol = PinnedPolicy.from_bundle("/path/to/bundle_v1")     # verifies every sha256
    res = pol.act({"observation/image": rgb, "observation/wrist_image": wrist,
                   "observation/state": state, "prompt": instruction})
    chunk = res["actions"]                       # (action_horizon, action_dim)
    res["snmvp_command"]                         # the c that was pinned
    res["snmvp_command_displacement"]            # what c asks for, in metres — loggable

## Steering

    pol.nudge("z", 0.30)      # every command now asks for +0.30 m more net climb
    pol.clear_nudge()

Verified end to end on the record bundle: commanding +0.30 m produced a +0.291 m net chunk
displacement with lateral axes undisturbed (+0.002, -0.003 m). Axes with zero action std in
the training data (yaw, gripper on the drone set) raise rather than silently doing nothing.
Closed loop, a held nudge is opposed by the command source's restoring field, so realized
offset is smaller than commanded — measure the command-response curve per policy.

## Command sources

A bundle ships one of:
- **one-hot scaffold** (`tasks` in prior.npz): exact string match; raises on unseen wording.
  Not for deployment beyond bring-up.
- **language encoder** (`Em`/`P` in prior.npz): PCA-projected post-fusion language-token
  embedding of the live instruction; no task list. `SNMVP_LATCH_N` averages the first n
  calls of an episode then holds — default 0 (live), since latching assumes single-stage
  instructions and breaks multi-stage ones.

## Checklist before a real run

- [ ] `PinnedPolicy.from_bundle(..., verify=True)` passes on the target machine.
- [ ] Action space and units match the robot's controller (`manifest.axes`, action dim).
- [ ] Chunk execution policy decided (how many steps executed per inference call).
- [ ] Command displacement logged per step — it is the interpretable trace of intent.
- [ ] Safety limits applied downstream of `act()`; the policy does not clamp.
