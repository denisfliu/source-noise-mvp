#!/usr/bin/env python3
"""Overfit-probe training wrapper (task 4 of the MVP plan).

Runs openpi's scripts/train_pytorch.py unmodified, but restricts the LeRobot
dataset to the first SNMVP_OVERFIT_EPISODES episodes (default 10) by
monkeypatching lerobot's LeRobotDataset before the training script imports it.
No additional openpi patch needed; the arm C noise pin remains controlled by
SNMVP_PIN_ALPHA via patches/openpi_arm_c_training.patch as usual.

Usage (from the openpi checkout, inside its venv):

    cd ~/code/openpi
    UV_NO_SYNC=1 SNMVP_PIN_ALPHA=1.0 uv run python \
        ~/code/source-noise-mvp/scripts/run_overfit_train.py \
        pi0_libero --exp_name armC_overfit --num-train-steps 400 ...

All CLI args after the script name are passed through to train_pytorch.py.
Env knobs:
    SNMVP_OVERFIT_EPISODES  number of leading episodes to keep (default 10)
    OPENPI_DIR              openpi checkout (default ~/code/openpi)
"""

import os
import pathlib
import runpy
import sys

N_EPISODES = int(os.environ.get("SNMVP_OVERFIT_EPISODES", "10"))
OPENPI_DIR = pathlib.Path(os.environ.get("OPENPI_DIR", pathlib.Path.home() / "code" / "openpi"))

import lerobot.common.datasets.lerobot_dataset as _lds  # noqa: E402

_orig_dataset = _lds.LeRobotDataset


def _episode_subset_dataset(repo_id, *args, **kwargs):
    kwargs.setdefault("episodes", list(range(N_EPISODES)))
    ds = _orig_dataset(repo_id, *args, **kwargs)
    # Restore immediately: leaving a function in the class's module slot breaks
    # pickling of dataset instances in DataLoader worker processes.
    _lds.LeRobotDataset = _orig_dataset
    print(f"[snmvp overfit] LeRobotDataset restricted to episodes 0..{N_EPISODES - 1} "
          f"({ds.num_frames} frames)", flush=True)
    return ds


_lds.LeRobotDataset = _episode_subset_dataset

train_script = OPENPI_DIR / "scripts" / "train_pytorch.py"
sys.argv = [str(train_script), *sys.argv[1:]]
runpy.run_path(str(train_script), run_name="__main__")
