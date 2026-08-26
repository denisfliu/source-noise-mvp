"""Is the residual sim-vs-real difference a coordinate-FRAME artifact or a genuine ROUTE difference?
Build each episode's start-relative pose SHAPE (resample pose(progress) to G points, subtract the start
pose) so any constant frame offset cancels; also try per-domain mean-centering + per-domain std scaling
(removes offset AND scale/gain). Compare the mean shape sim vs real per gate against the left-vs-right
shape separation. If the shapes match after alignment, the flights are the same and the gap was frame
calibration (fixable, zero-shot should then work); if they still differ, the twin flies a different
route."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = "go through the gate on the right and hover over the stuffed animal"
DIMS = [0, 1, 2, 3]
GP = 12  # resample points


def load(raw, lang):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    out = []
    for k in sorted(meta):
        if meta[k]["lang"] == lang:
            d = np.load(os.path.join(raw, k + ".npz"))
            out.append(d["state"].astype(np.float32)[:, DIMS])
    return out


def resample(pose):
    T = len(pose); pr = np.arange(T) / max(T - 1, 1)
    grid = np.linspace(0, 1, GP)
    return np.stack([np.interp(grid, pr, pose[:, i]) for i in range(pose.shape[1])], 1)  # [GP,4]


def shapes(eps):
    return np.stack([resample(p) for p in eps])  # [N,GP,4]


def main():
    RR, SS = os.path.join(RD, "data_gate_real"), os.path.join(RD, "data_gate_synth")
    dom = {}
    for name, raw in (("real", RR), ("sim", SS)):
        for g, lang in ((0, LEFT), (1, RIGHT)):
            dom[(name, g)] = shapes(load(raw, lang))

    def report(transform, label):
        print(f"\n=== {label} ===")
        # per (domain,gate) mean shape after transform
        M = {k: transform(v, k).mean(0) for k, v in dom.items()}  # [GP,4]
        for g, tag in ((0, "LEFT"), (1, "RIGHT")):
            gap = np.linalg.norm(M[("real", g)] - M[("sim", g)]) / np.sqrt(GP)
            print(f"  [{tag}] sim-vs-real mean-shape gap = {gap:.3f}")
        sep_r = np.linalg.norm(M[("real", 0)] - M[("real", 1)]) / np.sqrt(GP)
        sep_s = np.linalg.norm(M[("sim", 0)] - M[("sim", 1)]) / np.sqrt(GP)
        avg_gap = 0.5 * (np.linalg.norm(M[("real", 0)] - M[("sim", 0)]) + np.linalg.norm(M[("real", 1)] - M[("sim", 1)])) / np.sqrt(GP)
        print(f"  left-right shape sep: real={sep_r:.3f} sim={sep_s:.3f} | mean domain gap={avg_gap:.3f} | ratio gap/sep_real={avg_gap/(sep_r+1e-9):.3f}")

    # raw (absolute pose)
    report(lambda v, k: v, "RAW absolute pose")
    # start-relative (subtract each episode's start pose) -> cancels constant frame offset
    report(lambda v, k: v - v[:, :1, :], "START-RELATIVE (removes constant frame offset)")
    # per-domain affine: subtract domain-mean and divide by domain-std over all its poses (removes offset+scale)
    dstat = {}
    for name in ("real", "sim"):
        allp = np.concatenate([dom[(name, 0)].reshape(-1, 4), dom[(name, 1)].reshape(-1, 4)])
        dstat[name] = (allp.mean(0), allp.std(0) + 1e-6)

    def affine(v, k):
        m, s = dstat[k[0]]
        return (v - m) / s
    report(affine, "PER-DOMAIN AFFINE (removes offset + per-axis scale/gain)")
    print("SHAPE_DONE")


if __name__ == "__main__":
    main()
