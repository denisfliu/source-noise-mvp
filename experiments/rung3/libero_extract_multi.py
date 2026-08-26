"""Extract a MULTI-task LIBERO raw set for the both-cases eval: the goal suite (10-19, language-
driven motion) and the object suite (20-29, state-driven target). ~10 episodes/task with images,
each labeled by task_index + language, into data_libero_multi/. Feeds the (state+language)->c prior
and the language-based offline eval (RRR vs PCA, does adding language help on goal tasks)."""
import json
import os

import numpy as np
import lerobot.common.datasets.lerobot_dataset as L

OUT = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3/data_libero_multi")
TASKS = list(range(10, 30))     # goal + object
PER = 10


def hwc(t):
    return (t.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = L.LeRobotDataset("physical-intelligence/libero")
    frm, to = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    got = {t: 0 for t in TASKS}
    meta = {}
    n = 0
    for e in range(len(frm)):
        ti = int(ds[frm[e]]["task_index"])
        if ti not in got or got[ti] >= PER:
            continue
        got[ti] += 1
        imgs, wr, sts, acts, lang = [], [], [], [], None
        for i in range(frm[e], to[e]):
            f = ds[i]
            imgs.append(hwc(f["image"])); wr.append(hwc(f["wrist_image"]))
            sts.append(f["state"].numpy()); acts.append(f["actions"].numpy())
            if lang is None:
                lang = f["task"]
        np.savez_compressed(os.path.join(OUT, f"ep_{n:04d}.npz"), image=np.stack(imgs),
                            wrist=np.stack(wr), state=np.stack(sts).astype(np.float32),
                            action=np.stack(acts).astype(np.float32))
        meta[f"ep_{n:04d}"] = {"task": ti, "lang": lang, "T": len(acts)}
        n += 1
        if all(got[t] >= PER for t in TASKS):
            break
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print(f"LIBERO_MULTI_DONE episodes={n} tasks={sorted(set(m['task'] for m in meta.values()))}")


if __name__ == "__main__":
    main()
