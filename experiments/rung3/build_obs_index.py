"""Pose index over an episode corpus for the real-in-the-loop emulator (2026-08-28).
Saves pos/yaw/ep/t arrays; images are loaded lazily at retrieval time.

  python3 build_obs_index.py --data-dir data_gate_real --out /home/dfliu/ctxrun/obsidx_real.npz
"""
import argparse
import glob
import os

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=2)
a = ap.parse_args()
pos, yaw, ep, tt = [], [], [], []
for f in sorted(glob.glob(f"{RD}/{a.data_dir}/ep_*.npz")):
    e = int(os.path.basename(f)[3:7])
    st = np.load(f, allow_pickle=True)["state"].astype(np.float32)
    for t in range(0, len(st), a.stride):
        pos.append(st[t, :3]); yaw.append(st[t, 3]); ep.append(e); tt.append(t)
np.savez(a.out, pos=np.stack(pos), yaw=np.array(yaw, np.float32),
         ep=np.array(ep, np.int32), t=np.array(tt, np.int32),
         data_dir=os.path.abspath(f"{RD}/{a.data_dir}"))
print(f"{a.out}: {len(ep)} frames from {a.data_dir}")
