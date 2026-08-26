"""Build local/gate_nav3: the regenerated training set (2026-08-23).

Composition, preserving gate_nav's episode-order convention exactly (real 0-99, then synth
CFL 100-149 / CFR 150-199 / L 200-249 / R 250-299) so every episode-indexed tool keeps working:

  - REAL 0-99: carried over from the v3 source unchanged (same BGR->RGB decode as
    gate_v3_to_lerobot.py; frame-count classifier + episode-53 exception).
  - SYNTH 100-299: the regen2 renders (fixed course family: CFR cross_west return + funnels +
    return berths + capped correctives; real-matched start distribution; every kept trajectory
    passed the posthoc judge AND gate_clearance at plan time). Staging parquets are ALREADY
    RGB (2026-06-12 exporter convention) — no channel swap.

Norm stats are NOT recomputed: assets reuse gate_nav's stats so U bases and c units stay
comparable across the data change (box ladder/aug precedent). Splits written as
gate_{synth,real}_eps3.json.

  ~/code/openpi/.venv/bin/python build_gate_nav3.py
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
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME as HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

SRC = os.path.expanduser("~/code/falsify/data/no_3pov_v3/gate_scenes_all_no_3pov")
REGEN = os.path.expanduser("~/code/falsify/runs/regen2/staging")
REPO = os.environ.get("REPO", "local/gate_nav3")
RD = os.path.dirname(os.path.abspath(__file__))
FORCE_REAL = {53}
PROMPTS = {
    "through_center_gate_from_left": "go through the center gate from the left and hover over the stuffed animal",
    "through_center_gate_from_right": "go through the center gate from the right and hover over the stuffed animal",
    "through_left_gate": "go through the gate on the left and hover over the stuffed animal",
    "through_right_gate": "go through the gate on the right and hover over the stuffed animal",
}
SYNTH_ORDER = ["through_center_gate_from_left", "through_center_gate_from_right",
               "through_left_gate", "through_right_gate"]


def dec_bgr(cell):
    im = np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    return np.ascontiguousarray(im[:, :, ::-1])


def dec_rgb(cell):
    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


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

    new_ei = 0
    real = []
    # --- real episodes from the v3 source, in file order (matches gate_nav real 0-99) ---
    for p in sorted(glob.glob(f"{SRC}/data/chunk-000/episode-*.parquet")):
        ei = int(os.path.basename(p).split("-")[1].split(".")[0])
        n = pq.read_metadata(p).num_rows
        st = station.get(ei)
        is_synth = ((st == "center_gate" and n == 301) or
                    (st in ("left_gate", "right_gate") and n in (241, 301)))
        if ei in FORCE_REAL:
            is_synth = False
        if is_synth:
            continue
        tb = pq.read_table(p).to_pydict()
        for i in range(n):
            ds.add_frame({
                "image": dec_bgr(tb["observation.images.image"][i]),
                "wrist_image": dec_bgr(tb["observation.images.wrist_image"][i]),
                "state": np.asarray(tb["observation.state"][i], np.float32),
                "actions": np.asarray(tb["action"][i], np.float32),
            })
        ds.save_episode(task=tmap[int(tb["task_index"][0])])
        real.append(new_ei)
        new_ei += 1
        if new_ei % 25 == 0:
            print(f"...real {new_ei}", flush=True)
    assert len(real) == 100, f"expected 100 real, got {len(real)}"

    # --- new synth from regen2 staging, fixed task order ---
    synth = []
    for course in SYNTH_ORDER:
        prompt = PROMPTS[course]
        eps = sorted(glob.glob(f"{REGEN}/{course}/episode_*/episode_*.parquet"))
        assert len(eps) == 50, f"{course}: expected 50 staged episodes, got {len(eps)}"
        for p in eps:
            tb = pq.read_table(p).to_pydict()
            n = len(tb["state"])
            for i in range(n):
                ds.add_frame({
                    "image": dec_rgb(tb["image"][i]),
                    "wrist_image": dec_rgb(tb["wrist_image"][i]),
                    "state": np.asarray(tb["state"][i], np.float32),
                    "actions": np.asarray(tb["actions"][i], np.float32),
                })
            ds.save_episode(task=prompt)
            synth.append(new_ei)
            new_ei += 1
        print(f"...synth {course} done ({new_ei} total)", flush=True)

    json.dump(synth, open(f"{RD}/gate_synth_eps3.json", "w"))
    json.dump(real, open(f"{RD}/gate_real_eps3.json", "w"))
    print(f"GATE_NAV3_DONE repo={REPO} real={len(real)} synth={len(synth)} out={out}")


if __name__ == "__main__":
    main()
