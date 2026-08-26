#!/usr/bin/env python3
"""Compute dataset statistics of the chunk invariant L(a0) in normalized units.

Iterates the same training data pipeline the model sees (post q01/q99
normalization, post delta transform) and writes {"mean": [k], "std": [k]} for
the leading k real action dims. Needed by arm B (conditioning injection,
SNMVP_COND_STATS) and useful as the PinStats source for the zscored-pin
ablation of arm C.

Run from the openpi checkout:

    cd ~/code/openpi
    UV_NO_SYNC=1 uv run python \
        ~/code/source-noise-mvp/scripts/compute_invariant_stats.py \
        --config pi0_libero --num-batches 300 \
        --out ~/code/source-noise-mvp/experiments/phase1/invariant_stats.json

Set SNMVP_OVERFIT_EPISODES to restrict episodes (same knob as
run_overfit_train.py) when computing stats for an overfit-subset run.
"""

import argparse
import dataclasses
import json
import os
import pathlib

import numpy as np

if os.environ.get("SNMVP_OVERFIT_EPISODES"):
    N_EPISODES = int(os.environ["SNMVP_OVERFIT_EPISODES"])
    import lerobot.common.datasets.lerobot_dataset as _lds

    _orig_dataset = _lds.LeRobotDataset

    def _episode_subset_dataset(repo_id, *args, **kwargs):
        kwargs.setdefault("episodes", list(range(N_EPISODES)))
        ds = _orig_dataset(repo_id, *args, **kwargs)
        _lds.LeRobotDataset = _orig_dataset
        return ds

    _lds.LeRobotDataset = _episode_subset_dataset

import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data  # noqa: E402

REAL_DIMS = 7  # LIBERO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi0_libero")
    ap.add_argument("--num-batches", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dims", type=int, default=REAL_DIMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(cfg, batch_size=args.batch_size, num_workers=2)
    loader = _data.create_data_loader(
        cfg, shuffle=True, num_batches=args.num_batches, framework="pytorch"
    )

    invs = []
    for i, (_, actions) in enumerate(loader):
        a = actions.to("cpu").float().numpy()  # (B, H, D) normalized
        invs.append(a.sum(-2)[:, : args.dims])
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{args.num_batches} batches", flush=True)
    invs = np.concatenate(invs)

    out = {
        "config": args.config,
        "num_chunks": int(invs.shape[0]),
        "dims": args.dims,
        "episodes_restricted_to": os.environ.get("SNMVP_OVERFIT_EPISODES"),
        "mean": invs.mean(0).tolist(),
        "std": invs.std(0).tolist(),
    }
    p = pathlib.Path(os.path.expanduser(args.out))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)
    print("mean:", np.round(invs.mean(0), 2), "std:", np.round(invs.std(0), 2))


if __name__ == "__main__":
    main()
