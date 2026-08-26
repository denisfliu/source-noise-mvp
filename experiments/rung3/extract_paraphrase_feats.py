"""Extract fused features for the TRAINING paraphrase set (GPU): a subsample of
TRAIN-episode rendered frames x every training paraphrase. Output feeds the
paraphrase-augmented selector retrain. env: SHARD_K/SHARD_N, FRAMES_PER_SCENE (30).
Writes RUN/Xparashard_K.npy + RUN/parameta.npz (frame keys + prompt labels)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
from train_paraphrases import TRAIN_PARAPHRASES

RUN = os.path.expanduser("~/ctxrun")
SHARD_N = int(os.environ.get("SHARD_N", "1")); SHARD_K = int(os.environ.get("SHARD_K", "0"))
FPS_ = int(os.environ.get("FRAMES_PER_SCENE", "30"))
TASKS = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]

rf = np.load(f"{RUN}/rendered_frames.npz")
fwd224, wrist224 = rf["fwd224"], rf["wrist224"]  # materialize once (npz trap)
row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rf["si"], rf["fidx"]))}
src = gc.load_eps(with_images=False)
rng = np.random.default_rng(0)
idx = rng.permutation(len(src)); trep = [int(i) for i in idx[:160]]

# balanced train-episode frames across the two splat scenes
frames = {"left": [], "right": []}
for si in trep:
    scene = "right" if src[si]["lang"] == gc.PROMPT_R else "left"
    if len(frames[scene]) >= FPS_:
        continue
    for t in (12, 96):
        if (si, t) in row and len(frames[scene]) < FPS_:
            frames[scene].append((si, t))
probe = frames["left"] + frames["right"]
print("training frames:", len(probe), flush=True)

rows = []   # (frame_key, prompt, task_idx)
for task, plist in TRAIN_PARAPHRASES.items():
    k = TASKS.index(task)
    for p in plist:
        for f in probe:
            rows.append((f, p, k))
print("paraphrase rows:", len(rows), flush=True)

per = (len(rows) + SHARD_N - 1) // SHARD_N
lo = SHARD_K * per; hi = min(len(rows), lo + per)
policy = gc.make_policy()
obs = [{"observation/image": fwd224[row[f]], "observation/wrist_image": wrist224[row[f]],
        "observation/state": src[f[0]]["state"][f[1]], "prompt": p} for f, p, _ in rows[lo:hi]]
X = gc.feats(policy, obs, log_every=20)
np.save(f"{RUN}/Xparashard_{SHARD_K}.npy", X)
if SHARD_K == 0:
    np.savez(f"{RUN}/parameta.npz",
             si=np.array([f[0] for f, _, _ in rows], np.int32),
             fidx=np.array([f[1] for f, _, _ in rows], np.int32),
             task=np.array([k for _, _, k in rows], np.int32))
print("PARASHARD_%d_DONE %d-%d %s" % (SHARD_K, lo, hi, X.shape), flush=True)
