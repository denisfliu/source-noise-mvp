"""Matched-pair swap table for S3 cross-supervised training (2026-08-28): for every REAL
frame (gate_nav3 episodes 0-99) with a synth state match (pos+yaw gate), store the matched
synth 50-step action chunk. The loader transform swaps it in with probability p; the
in-graph pin c and head target follow the swapped chunk automatically.

  python3 build_xswap_table.py --out /home/dfliu/ctxrun/xswap_table.npz
"""
import argparse
import glob
import os

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
H = 50
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--max-dist", type=float, default=0.35)
ap.add_argument("--max-dyaw", type=float, default=0.6)
a = ap.parse_args()

synth_states, synth_ref = [], []
for f in sorted(glob.glob(f"{RD}/data_gate_synth3/ep_*.npz")):
    e = int(os.path.basename(f)[3:7])
    d = np.load(f, allow_pickle=True)
    st = d["state"].astype(np.float32)
    for t in range(0, len(st) - H - 1, 3):
        synth_states.append(st[t])
        synth_ref.append((e, t))
S = np.stack(synth_states)
print(f"synth index {len(S)}")

eps, frs, chunks, dists = [], [], [], []
synth_cache = {}
for f in sorted(glob.glob(f"{RD}/data_gate_real/ep_*.npz")):
    e = int(os.path.basename(f)[3:7])
    d = np.load(f, allow_pickle=True)
    st = d["state"].astype(np.float32)
    for t in range(len(st)):
        dp = np.linalg.norm(S[:, :3] - st[t, :3], axis=1)
        dy = np.abs(np.angle(np.exp(1j * (S[:, 3] - st[t, 3]))))
        score = dp + 0.3 * dy
        j = int(np.argmin(score))
        if dp[j] > a.max_dist or dy[j] > a.max_dyaw:
            continue
        es, ts = synth_ref[j]
        if es not in synth_cache:
            if len(synth_cache) > 6:
                synth_cache.pop(next(iter(synth_cache)))
            synth_cache[es] = np.load(f"{RD}/data_gate_synth3/ep_{es:04d}.npz",
                                      allow_pickle=True)["action"].astype(np.float32)
        ac = synth_cache[es]
        ch = np.zeros((H, 7), np.float16)
        m = min(H, len(ac) - ts)
        ch[:m] = ac[ts:ts + m].astype(np.float16)
        eps.append(e); frs.append(t); chunks.append(ch); dists.append(dp[j])
    print(f"ep{e:03d}: cumulative matched {len(eps)}", flush=True)

np.savez_compressed(a.out, ep=np.array(eps, np.int32), frame=np.array(frs, np.int32),
                    chunk=np.stack(chunks), dist=np.array(dists, np.float32))
print(f"saved {a.out}: {len(eps)} matched real frames "
      f"(match dist median {np.median(dists):.2f})")
