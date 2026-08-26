"""Control-set extractor: dump ~40 single-task LIBERO episodes into the same raw npz schema
as the Bridge data (image + wrist + state + action, plus meta lang), so eval_offline_action.py
can be validated in-distribution. On LIBERO the pin is known to work, so the evaluator should
show low pass-through error, high oracle subspace R^2, and a high state->c prior R^2 for a
single task -- confirming the evaluator before trusting the Bridge numbers."""
import json
import os

import numpy as np
import lerobot.common.datasets.lerobot_dataset as L

OUT = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3/data_libero_raw")
N = 40


def hwc_u8(t):
    return (t.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = L.LeRobotDataset("physical-intelligence/libero")
    frm = ds.episode_data_index["from"].tolist()
    to = ds.episode_data_index["to"].tolist()
    tgt = int(ds[frm[0]]["task_index"])
    picked = [e for e in range(len(frm)) if int(ds[frm[e]]["task_index"]) == tgt][:N]
    meta = {}
    for n, e in enumerate(picked):
        imgs, wr, sts, acts, lang = [], [], [], [], None
        for i in range(frm[e], to[e]):
            f = ds[i]
            imgs.append(hwc_u8(f["image"]))
            wr.append(hwc_u8(f["wrist_image"]))
            sts.append(f["state"].numpy())
            acts.append(f["actions"].numpy())
            if lang is None:
                lang = f["task"]
        np.savez_compressed(os.path.join(OUT, f"ep_{n:04d}.npz"),
                            image=np.stack(imgs), wrist=np.stack(wr),
                            state=np.stack(sts).astype(np.float32), action=np.stack(acts).astype(np.float32))
        meta[f"ep_{n:04d}"] = {"lang": lang, "T": len(acts)}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print(f"LIBERO_RAW_DONE tasks_index={tgt} episodes={len(picked)} lang={meta['ep_0000']['lang']!r}")


if __name__ == "__main__":
    main()
