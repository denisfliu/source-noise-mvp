"""Convert the gate_scenes_all_no_3pov v3.0 dataset into a v2.0 LeRobot dataset our pi0 pipeline can
load ('local/gate_nav'). Decodes the embedded image bytes and swaps BGR->RGB (the doc flags legacy
BGR fisheye). Preserves the language (which-gate task). Writes synth/real episode-index splits using
the frame-count classifier with episode 53 forced to REAL -> 100 real / 200 synthetic.

Recovered from the box's loose home scripts (box-code-backup-2026-08-13/ec2/_loose_home) and adapted
for the local 4090 machine (2026-08-19): SRC defaults to the falsify copy of the dataset; the
lerobot home constant fell back to the older LEROBOT_HOME name in this venv's lerobot.

  SRC=<v3 dataset dir> REPO=local/gate_nav LIMIT=<n eps, smoke only> \
      ~/code/openpi/.venv/bin/python gate_v3_to_lerobot.py
"""
import csv
import glob
import io
import json
import os
import shutil

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ImportError:  # older lerobot (this machine's openpi venv)
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME as HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

SRC = os.path.expanduser(os.environ.get(
    "SRC", "~/code/falsify/data/no_3pov_v3/gate_scenes_all_no_3pov"))
REPO = os.environ.get("REPO", "local/gate_nav")
LIMIT = int(os.environ.get("LIMIT", "0"))          # >0 = smoke test on the first N episodes
RD = os.path.dirname(os.path.abspath(__file__))
FORCE_REAL = {53}


def dec(cell):
    im = np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    return np.ascontiguousarray(im[:, :, ::-1])   # BGR -> RGB


def main():
    tasks = pq.read_table(f"{SRC}/meta/tasks.parquet").to_pydict()
    tmap = dict(zip(tasks["task_index"], tasks["__index_level_0__"]))
    station = {int(r["episode_index"]): r["station_id"]
               for r in csv.DictReader(open(f"{SRC}/meta/custom_metadata.csv"))}

    out = HF_LEROBOT_HOME / REPO
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=REPO, robot_type="drone", fps=10,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10, image_writer_processes=5)

    synth, real = [], []
    files = sorted(glob.glob(f"{SRC}/data/chunk-000/episode-*.parquet"))
    if LIMIT:
        files = files[:LIMIT]
    for new_ei, p in enumerate(files):
        ei = int(os.path.basename(p).split("-")[1].split(".")[0])
        n = pq.read_metadata(p).num_rows
        st = station.get(ei)
        is_synth = ((st == "center_gate" and n == 301) or
                    (st in ("left_gate", "right_gate") and n in (241, 301)))
        if ei in FORCE_REAL:
            is_synth = False
        (synth if is_synth else real).append(new_ei)

        tb = pq.read_table(p).to_pydict()
        imgs = tb["observation.images.image"]; wrists = tb["observation.images.wrist_image"]
        states = tb["observation.state"]; acts = tb["action"]; tix = tb["task_index"]
        for i in range(n):
            ds.add_frame({
                "image": dec(imgs[i]), "wrist_image": dec(wrists[i]),
                "state": np.asarray(states[i], np.float32), "actions": np.asarray(acts[i], np.float32),
            })
        # this venv's older lerobot takes the language string at save_episode, not per-frame
        ds.save_episode(task=tmap[int(tix[0])])
        if (new_ei + 1) % 25 == 0:
            print(f"...{new_ei + 1}/{len(files)} episodes", flush=True)

    if not LIMIT:
        json.dump(synth, open(f"{RD}/gate_synth_eps.json", "w"))
        json.dump(real, open(f"{RD}/gate_real_eps.json", "w"))
    print(f"GATE_CONVERT_DONE repo={REPO} synth={len(synth)} real={len(real)} out={out}")


if __name__ == "__main__":
    main()
