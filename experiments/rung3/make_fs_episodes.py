"""Write the episode-index list for one LIBERO task (for few-shot adaptation via SNMVP_EPISODES).
Env: SNMVP_FS_TASK (global task_index), SNMVP_FS_OUT (output json path)."""
import json
import os

import numpy as np
import lerobot.common.datasets.lerobot_dataset as L

g = int(os.environ["SNMVP_FS_TASK"])
out = os.environ["SNMVP_FS_OUT"]
ds = L.LeRobotDataset("physical-intelligence/libero")
frm = ds.episode_data_index["from"].tolist()
tix = np.asarray(ds.hf_dataset.with_format("numpy")["task_index"])
eps = [e for e in range(len(frm)) if int(tix[frm[e]]) == g]
json.dump(eps, open(out, "w"))
print(f"FS_EP task={g} n={len(eps)} -> {out}")
