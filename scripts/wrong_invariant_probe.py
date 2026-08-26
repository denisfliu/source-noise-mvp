#!/usr/bin/env python3
"""Wrong-invariant probe (task 4 of the MVP plan).

Loads a PyTorch training checkpoint, draws observation/action samples from the
same (episode-restricted) training pipeline the overfit run saw, and measures
error-to-command in NORMALIZED action units for three command conditions:

    oracle   command = L(a0) of the sample's own demo chunk (consistency check)
    wrong    command = another sample's invariant (contradicts the scene)
    negated  command = -L(a0) (strong contradiction)

plus an unpinned-noise control (what the model does with plain Gaussian noise).

Everything happens in the model's native normalized/padded action space via
model.sample_actions(noise=make_calibrated_noise(...)) — no un/re-normalization
round trip, so the metric is exactly the quantity the training pin carried.

Run from the openpi checkout (its venv has snmvp installed):

    cd ~/code/openpi
    UV_NO_SYNC=1 SNMVP_OVERFIT_EPISODES=10 uv run python \
        ~/code/source-noise-mvp/scripts/wrong_invariant_probe.py \
        --checkpoint checkpoints/pi0_libero/armC_overfit/399 \
        --out ~/code/source-noise-mvp/experiments/phase1/results/overfit_probe.json
"""

import argparse
import dataclasses
import json
import os
import pathlib

import numpy as np
import torch

N_EPISODES = int(os.environ.get("SNMVP_OVERFIT_EPISODES", "10"))

import lerobot.common.datasets.lerobot_dataset as _lds  # noqa: E402

_orig_dataset = _lds.LeRobotDataset


def _episode_subset_dataset(repo_id, *args, **kwargs):
    kwargs.setdefault("episodes", list(range(N_EPISODES)))
    ds = _orig_dataset(repo_id, *args, **kwargs)
    _lds.LeRobotDataset = _orig_dataset
    return ds


_lds.LeRobotDataset = _episode_subset_dataset

import jax  # noqa: E402  (used only for tree_map on torch tensors, as in train_pytorch)
import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data  # noqa: E402

from snmvp.openpi_adapter import make_calibrated_noise  # noqa: E402

REAL_DIMS = 7  # LIBERO: 6 EE deltas + gripper (leading dims of the padded 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="dir containing model.safetensors")
    ap.add_argument("--config", default="pi0_libero")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--noise-draws", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=10, help="Euler steps (openpi default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad-command", action="store_true",
                    help="pin all padded dims (command zeros beyond the real 7) to "
                         "match the training-side pin, which covered all 32 dims")
    ap.add_argument("--arm", default="C", choices=["B", "C"],
                    help="C: command via pinned source noise; B: command via the "
                         "conditioning-state injection (plain noise), mirroring "
                         "the arm B training patch")
    ap.add_argument("--cond-stats", default=None,
                    help="invariant_stats.json (required for --arm B)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(cfg, batch_size=1, num_workers=0)

    model = cfg.model.load_pytorch(cfg, os.path.join(args.checkpoint, "model.safetensors"))
    model = model.to(device)
    model.eval()

    loader = _data.create_data_loader(
        cfg, shuffle=True, num_batches=args.num_samples, framework="pytorch"
    )
    H, D = cfg.model.action_horizon, cfg.model.action_dim

    samples = []
    for obs, actions in loader:
        actions = actions.to(torch.float32)
        m_true = actions.sum(-2)[0, :REAL_DIMS].cpu().numpy().astype(np.float32)
        samples.append((obs, m_true))
        if len(samples) >= args.num_samples:
            break

    inv_stack = np.stack([m for _, m in samples])
    inv_scale = float(np.linalg.norm(inv_stack.std(0)))  # dataset scale reference

    def to_dev(obs):
        return jax.tree.map(lambda x: x.to(device) if hasattr(x, "to") else x, obs)

    def widen(cmd):
        if not args.pad_command:
            return cmd
        full = np.zeros(D, dtype=np.float32)
        full[:REAL_DIMS] = cmd
        return full

    cond_stats = None
    if args.arm == "B":
        from snmvp.conditioning import inject_invariant_state, load_invariant_stats
        assert args.cond_stats, "--arm B requires --cond-stats"
        cond_stats = load_invariant_stats(args.cond_stats)

    def rollout(obs_d, noise, command=None):
        if args.arm == "B" and command is not None:
            # arm B path: command through the state token, plain noise
            import dataclasses as _dc
            state = obs_d.state.clone()
            inject_invariant_state(
                state, state.new_tensor(command[:REAL_DIMS])[None], *cond_stats)
            obs_d = _dc.replace(obs_d, state=state)
            noise = rng.normal(size=noise.shape)  # fresh plain noise
        noise_t = torch.from_numpy(noise.astype(np.float32))[None].to(device)
        pred = model.sample_actions(device, obs_d, noise=noise_t, num_steps=args.num_steps)
        chunk = pred[0].float().cpu().numpy()  # (H, D) normalized
        return chunk

    conditions = {"oracle": [], "wrong": [], "negated": [], "unpinned_control": []}
    per_sample = []
    with torch.no_grad():
        for i, (obs, m_true) in enumerate(samples):
            obs_d = to_dev(obs)
            m_wrong = samples[(i + len(samples) // 2) % len(samples)][1]
            row = {"oracle_invariant": m_true.tolist()}
            for tag, cmd in [("oracle", m_true), ("wrong", m_wrong), ("negated", -m_true)]:
                errs, realized_draws, spreads = [], [], []
                for _ in range(args.noise_draws):
                    if args.arm == "B":
                        chunk = rollout(obs_d, rng.normal(size=(H, D)), command=cmd)
                    else:
                        chunk = rollout(obs_d, make_calibrated_noise(widen(cmd), H, D, rng))
                    realized = chunk.sum(0)[:REAL_DIMS]
                    errs.append(float(np.linalg.norm(realized - cmd)))
                    realized_draws.append(realized)
                # diversity of the residual (within-chunk variation across draws)
                stack = np.stack(realized_draws)
                row[tag] = {
                    "command": cmd.tolist(),
                    "err_to_command_l2": errs,
                    "realized_mean": stack.mean(0).tolist(),
                    "realized_std_across_draws": float(stack.std(0).mean()),
                }
                conditions[tag].extend(errs)
                if tag == "wrong":
                    d_cmd = float(np.linalg.norm(stack.mean(0) - m_wrong))
                    d_scene = float(np.linalg.norm(stack.mean(0) - m_true))
                    row["wrong_follows_command"] = bool(d_cmd < d_scene)
                    row["wrong_d_cmd"] = d_cmd
                    row["wrong_d_scene"] = d_scene
            # unpinned control: plain Gaussian noise, error vs the scene oracle
            errs = []
            for _ in range(args.noise_draws):
                chunk = rollout(obs_d, rng.normal(size=(H, D)))
                errs.append(float(np.linalg.norm(chunk.sum(0)[:REAL_DIMS] - m_true)))
            row["unpinned_control_err_to_oracle"] = errs
            conditions["unpinned_control"].extend(errs)
            per_sample.append(row)
            print(f"sample {i}: oracle_err={np.mean(row['oracle']['err_to_command_l2']):.4f} "
                  f"wrong_err={np.mean(row['wrong']['err_to_command_l2']):.4f} "
                  f"follows_cmd={row.get('wrong_follows_command')}", flush=True)

    def agg(v):
        a = np.asarray(v)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "n": len(v)}

    follow_rate = float(np.mean([r["wrong_follows_command"] for r in per_sample]))
    out = {
        "experiment": "wrong_invariant_probe",
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "n_episodes_overfit": N_EPISODES,
        "num_samples": len(samples),
        "noise_draws": args.noise_draws,
        "euler_steps": args.num_steps,
        "seed": args.seed,
        "invariant_dataset_scale_l2": inv_scale,
        "summary_err_to_command_l2": {k: agg(v) for k, v in conditions.items()},
        "wrong_invariant_follow_rate": follow_rate,
        "per_sample": per_sample,
    }
    p = pathlib.Path(os.path.expanduser(args.out))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)
    print("SUMMARY:",
          {k: round(agg(v)["mean"], 4) for k, v in conditions.items()},
          "follow_rate:", follow_rate, "dataset_scale:", round(inv_scale, 3))


if __name__ == "__main__":
    main()
