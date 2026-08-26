"""Compute pi0 norm stats (state + actions) for the extracted Bridge data, in the exact
openpi format (openpi.shared.normalize), so the frozen sim flow can be served with Bridge
normalization via create_trained_policy(norm_stats=...). Stats are per-channel mean/std
(and quantiles) over all per-step Bridge state/action vectors from data_bridge_raw/."""
import glob
import os

import numpy as np
import openpi.shared.normalize as normalize

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
RAW = os.path.join(RD, "data_bridge_raw")
OUT = os.path.join(RD, "bridge_norm")


def main():
    os.makedirs(OUT, exist_ok=True)
    rs_s, rs_a = normalize.RunningStats(), normalize.RunningStats()
    files = sorted(glob.glob(os.path.join(RAW, "ep_*.npz")))
    n = 0
    for f in files:
        d = np.load(f)
        rs_s.update(d["state"].astype(np.float32))
        rs_a.update(d["action"].astype(np.float32))
        n += 1
    ns = {"state": rs_s.get_statistics(), "actions": rs_a.get_statistics()}
    normalize.save(OUT, ns)
    np.set_printoptions(precision=3, suppress=True)
    print("episodes", n)
    print("state mean", ns["state"].mean, "std", ns["state"].std)
    print("actions mean", ns["actions"].mean, "std", ns["actions"].std)
    print("BRIDGE_NORM_DONE", OUT)


if __name__ == "__main__":
    main()
