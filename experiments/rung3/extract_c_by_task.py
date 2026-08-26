"""E1: where does the pin coordinate c live? Over three single-scene LIBERO suites (goal 10-19,
object 20-29, spatial 30-39), compute c = U^T a for every action chunk (no-delta shared space, so
c is faithful) and decompose Var(c) via the law of total variance into BETWEEN-instruction
(what language sets = E[c|task] variance) and WITHIN-instruction (what state must set = mean
Var(c|task)). between/total is the ceiling a language predictor can reach; within/total is left to
state. Saves per-sample c + per-task mean c for the decoder and the (state,language)->c predictor."""
import json
import os

import numpy as np
import lerobot.common.datasets.lerobot_dataset as L
import openpi.shared.normalize as N

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32
U = np.load(os.path.join(RD, "pin_U_pca_k5_shared.npy")).astype(np.float32)
ns = N.load(os.path.join(RD, "norm_shared_libero"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)


def seg_to_c(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    segn = (seg - amean[:r]) / (astd[:r] + 1e-6)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = segn
    return ch.reshape(-1) @ U


def main():
    ds = L.LeRobotDataset("physical-intelligence/libero")
    frm, to = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    hf = ds.hf_dataset.with_format("numpy")
    acts = np.asarray(hf["actions"], dtype=np.float32)
    tidx = np.asarray(hf["task_index"])
    targets = set(range(10, 40))
    C, TI = [], []
    for e in range(len(frm)):
        a, b = frm[e], to[e]
        ti = int(tidx[a])
        if ti not in targets:
            continue
        ep = acts[a:b]
        for t in range(len(ep)):
            C.append(seg_to_c(ep[t:t + H]))
            TI.append(ti)
    C, TI = np.asarray(C), np.asarray(TI)

    def decomp(mask):
        c, ti = C[mask], TI[mask]
        n = len(c)
        total = c.var(0).sum()
        within = sum((ti == k).sum() / n * c[ti == k].var(0).sum() for k in np.unique(ti))
        return total, total - within, within  # total, between, within

    print(f"samples={len(C)} K={C.shape[1]}")
    for name, rng in [("goal", range(10, 20)), ("object", range(20, 30)),
                      ("spatial", range(30, 40)), ("all", range(10, 40))]:
        mask = np.isin(TI, list(rng))
        tot, btw, wtn = decomp(mask)
        print(f"{name:>8}: total={tot:.3f}  between/language={btw/tot*100:5.1f}%  within/state={wtn/tot*100:5.1f}%")

    taskmean = {int(k): C[TI == k].mean(0).tolist() for k in np.unique(TI)}
    np.savez(os.path.join(RD, "c_by_task.npz"), C=C, TI=TI)
    json.dump(taskmean, open(os.path.join(RD, "c_task_means.json"), "w"))
    print("E1_DONE saved c_by_task.npz, c_task_means.json")


if __name__ == "__main__":
    main()
