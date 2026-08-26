"""Build the augmented LeRobot dataset local/gate_nav_aug (A3 of the augmentation plan).

Sources: data_gate_synth (200 synth eps) + data_gate_real (100 real eps, langs from
meta.json) — same composition as local/gate_nav — augmented per gate_traj_algebra:
original + reverse + crop_to_gate + crop_from_gate + 1 hover per episode
(forward-dominant ~2:3 original:augmented by episode count; hover eps are short).

Schema mirrors local/gate_nav (v2.1, fps 10, image/wrist_image/state/actions,
use_videos=False). Norm stats are NOT recomputed — the aug run reuses gate_nav
stats (G0 verified augmented actions/c in-support) so U coords, clamps, and all
c-map caches stay comparable. Copy them:
  cp -r ~/code/openpi/assets/pi0_gate/local/gate_nav \
        ~/code/openpi/assets/pi0_gate_aug/local/gate_nav_aug
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import gate_traj_algebra as ta
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO = "local/gate_nav_aug"
ROOT = os.path.expanduser(f"~/.cache/huggingface/lerobot/{REPO}")
REAL = os.path.join(gc.RD, "data_gate_real")

def load_real():
    meta = json.load(open(os.path.join(REAL, "meta.json")))
    eps = []
    for k in sorted(meta):
        d = np.load(os.path.join(REAL, k + ".npz"), allow_pickle=True)
        eps.append({"image": d["image"], "wrist": d["wrist"],
                    "state": d["state"].astype(np.float32),
                    "action": d["action"].astype(np.float32), "lang": meta[k]["lang"]})
    return eps

def main():
    if os.path.exists(ROOT):
        print("removing stale", ROOT); shutil.rmtree(ROOT)
    eps = gc.load_eps(with_images=True) + load_real()
    print("source episodes:", len(eps), flush=True)
    aug = ta.augment(eps)
    n_by = {}
    for e in aug:
        n_by[e["lang"]] = n_by.get(e["lang"], 0) + 1
    print("augmented episodes:", len(aug), {k: v for k, v in sorted(n_by.items())}, flush=True)

    # NB shapes must be TUPLES: lerobot 0.1.0 validate_frame compares tuple(actual)!=shape
    feats = {
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    ds = LeRobotDataset.create(REPO, fps=10, robot_type="drone", features=feats,
                               use_videos=False, image_writer_processes=4, image_writer_threads=8)
    for i, e in enumerate(aug):
        n = min(len(e["action"]), len(e["state"]) - 1)
        act = np.concatenate([e["action"][:n], np.zeros((1, e["action"].shape[1]), np.float32)], 0)
        for t in range(n + 1):
            ds.add_frame({"image": np.ascontiguousarray(e["image"][t]),
                          "wrist_image": np.ascontiguousarray(e["wrist"][t]),
                          "state": e["state"][t].astype(np.float32),
                          "actions": act[t].astype(np.float32),
                          "task": e["lang"]})
        ds.save_episode()
        if i % 50 == 0:
            print("episode %d/%d" % (i, len(aug)), flush=True)
    print("BUILD_GATE_NAV_AUG_DONE", flush=True)

if __name__ == "__main__":
    main()
