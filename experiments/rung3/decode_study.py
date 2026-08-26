"""Decode study: for a variety of ground-truth LIBERO action chunks, split each into the part the
RRR pin CAPTURES (the projection P a = U UT a, what the prior sets and the source noise carries) and
the RESIDUAL (I - U UT) a (what the action expert must generate). We work in the model's normalized
action space, where the split is exact and linear: a = Pa + (I-P)a, so the end-effector paths add
(full = pinned + residual). Reports the explained fraction per chunk for RRR vs PCA and writes paths
for the artifact (top-down x-y of the EE position deltas, integrated)."""
import json
import os

import numpy as np
import lerobot.common.datasets.lerobot_dataset as L
import openpi.shared.normalize as N

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32
ns = N.load(os.path.join(RD, "norm_shared_libero"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
U_rrr = np.load(os.path.join(RD, "pin_U_rrr_k5_shared.npy")).astype(np.float64)
U_pca = np.load(os.path.join(RD, "pin_U_pca_k5_shared.npy")).astype(np.float64)
TASKS = [0, 3, 8, 10, 11, 12, 15, 16, 20, 24, 30, 33]     # diverse: libero_10/90, goal, object, spatial


def chunkX(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def path_xy(chunk7):                                       # integrate position deltas -> 2-D path
    p = np.concatenate([[[0, 0]], np.cumsum(chunk7[:, :2], 0)], 0)
    return p.tolist()


def main():
    ds = L.LeRobotDataset("physical-intelligence/libero")
    frm, to = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    hf = ds.hf_dataset.with_format("numpy")
    acts = np.asarray(hf["actions"], dtype=np.float32)
    tix = np.asarray(hf["task_index"])
    first = {}
    for e in range(len(frm)):
        ti = int(tix[frm[e]])
        if ti in TASKS and ti not in first:
            first[ti] = (frm[e], to[e], ds[frm[e]]["task"])

    out = {"chunks": [], "mean_explained": {}}
    exp_rrr, exp_pca = [], []
    for ti in TASKS:
        a, b, lang = first[ti]
        X = chunkX(acts[a:b])                              # (1600,)
        rec = {"task": ti, "lang": lang}
        full7 = X.reshape(H, AD)[:, :7]
        rec["full"] = path_xy(full7)
        for name, U in [("rrr", U_rrr), ("pca", U_pca)]:
            pin = (U @ (U.T @ X))                           # projection P a
            frac = float((pin ** 2).sum() / ((X ** 2).sum() + 1e-9))
            pin7 = pin.reshape(H, AD)[:, :7]
            res7 = full7 - pin7
            rec[f"pin_{name}"] = path_xy(pin7)
            rec[f"res_{name}"] = path_xy(res7)
            rec[f"explained_{name}"] = round(frac, 3)
            (exp_rrr if name == "rrr" else exp_pca).append(frac)
        out["chunks"].append(rec)
    out["mean_explained"] = {"rrr": round(float(np.mean(exp_rrr)), 3),
                             "pca": round(float(np.mean(exp_pca)), 3)}
    json.dump(out, open(os.path.join(RD, "decode_study.json"), "w"))
    print("mean explained  RRR", out["mean_explained"]["rrr"], " PCA", out["mean_explained"]["pca"])
    for c in out["chunks"]:
        print(f"  task {c['task']:2d} explained rrr={c['explained_rrr']:.2f} pca={c['explained_pca']:.2f}  {c['lang'][:52]}")
    print("DECODE_STUDY_DONE")


if __name__ == "__main__":
    main()
