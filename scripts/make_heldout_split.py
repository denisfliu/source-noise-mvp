#!/usr/bin/env python3
"""Generate and FREEZE the held-out placement split for LIBERO-Spatial.

The benchmark's canonical init states (task_suite.get_task_init_states) are
the states the original demos — and hence the pi0 LIBERO training data — were
collected from, so they are NOT held-out. This script samples fresh object
placements per task by resetting the env under a dedicated frozen seed and
records the full MuJoCo sim states. These states are used by Phase 1 eval as
the held-out placement set (plan: freeze BEFORE any Phase 1 training).

Run in the LIBERO client venv:

    PYTHONPATH=~/code/openpi/third_party/libero \
    LIBERO_CONFIG_PATH=~/code/libero-config \
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
    ~/code/openpi/examples/libero/.venv/bin/python \
        ~/code/source-noise-mvp/scripts/make_heldout_split.py \
        --out ~/code/source-noise-mvp/experiments/phase1/heldout_init_states
"""

import argparse
import json
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

HELDOUT_SEED = 424242  # frozen; never change after Phase 1 training starts
N_STATES_PER_TASK = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--n-states", type=int, default=N_STATES_PER_TASK)
    ap.add_argument("--seed", type=int, default=HELDOUT_SEED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    manifest = {"task_suite": args.task_suite, "seed": args.seed,
                "n_states_per_task": args.n_states, "date_utc": "2026-07-04",
                "note": "fresh BDDL placement samples; canonical init states are "
                        "in-distribution for the demos and are NOT held-out",
                "tasks": []}

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        canonical = suite.get_task_init_states(task_id)
        states = []
        for k in range(args.n_states):
            env.seed(args.seed + 1000 * task_id + k)
            env.reset()
            states.append(np.asarray(env.get_sim_state(), dtype=np.float64))
        env.close()
        states = np.stack(states)
        # sanity: fresh placements must differ from every canonical state
        d = np.abs(states[:, None, : canonical.shape[1]] - canonical[None]).max(-1)
        min_dist = float(d.min())
        fname = f"task{task_id:02d}.npy"
        np.save(out / fname, states)
        manifest["tasks"].append({"task_id": task_id, "language": task.language,
                                  "file": fname, "state_dim": int(states.shape[1]),
                                  "min_linf_dist_to_canonical": min_dist})
        print(f"task {task_id:2d}: {states.shape} min_linf_dist_to_canonical={min_dist:.4f} "
              f"| {task.language}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out / "manifest.json")
    print("HELDOUT_FINAL=ok")


if __name__ == "__main__":
    main()
