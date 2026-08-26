"""Extract the real gate episodes from the converted local/gate_nav LeRobot dataset into raw npz
(image, wrist, state, action + per-episode task/lang) so eval_offline_lang.py can score each trained
pattern on held-out real. Episode index list from gate_real_eps.json (env EPS overrides)."""
import json
import os

import numpy as np
import lerobot.common.datasets.lerobot_dataset as L

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
OUT = os.environ.get("OUT", os.path.join(RD, "data_gate_real"))
EPS = json.load(open(os.environ.get("EPS", os.path.join(RD, "gate_real_eps.json"))))


def hwc(t):
    a = t.numpy()
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return a


def main():
    os.makedirs(OUT, exist_ok=True)
    # local_files_only: this machine has no HF token; hub probes of local/* 401 (see LOCAL_CONTINUATION.md)
    ds = L.LeRobotDataset(os.environ.get("REPO", "local/gate_nav"), local_files_only=True)
    frm, to = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    meta = {}
    for n, e in enumerate(sorted(EPS)):
        imgs, wr, sts, acts, lang, tk = [], [], [], [], None, None
        for i in range(frm[e], to[e]):
            f = ds[i]
            imgs.append(hwc(f["image"])); wr.append(hwc(f["wrist_image"]))
            sts.append(np.asarray(f["state"], np.float32)); acts.append(np.asarray(f["actions"], np.float32))
            if lang is None:
                # local lerobot 0.1.0 frames carry no "task" string; resolve via the meta task table
                tk = int(f["task_index"]); lang = f["task"] if "task" in f else ds.meta.tasks[tk]
        np.savez_compressed(os.path.join(OUT, f"ep_{n:04d}.npz"),
                            image=np.stack(imgs), wrist=np.stack(wr),
                            state=np.stack(sts), action=np.stack(acts))
        meta[f"ep_{n:04d}"] = {"task": tk, "lang": lang, "T": len(acts), "src_ep": int(e)}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print(f"GATE_RAW_DONE {OUT} n={len(meta)} tasks={sorted(set(m['task'] for m in meta.values()))}")


if __name__ == "__main__":
    main()
