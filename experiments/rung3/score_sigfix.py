import sys, glob, numpy as np
sys.path.insert(0, "/home/dfliu/code/source-noise-mvp/experiments/rung3")
import gate_success as GS
import re
tag = sys.argv[1]
clr = {}
for l in open(sys.argv[2] if len(sys.argv) > 2 else f"/home/dfliu/ctxrun/arm_{tag}_scores.txt"):
    m = re.match(r"(traj_\S+)\.npy\s+min-clearance ([\d.]+)", l)
    if m: clr[m.group(1)] = float(m.group(2))
for side in ["left", "right"]:
    ok = miss = touch = n = 0
    for f in sorted(glob.glob(f"/home/dfliu/ctxrun/traj_arm{tag}_{side}_[0-9]*.npy")):
        stem = f.split("/")[-1][:-4]; j = GS.judge(np.load(f)[:, :3].astype(float), side); c = clr[stem] >= 0.18
        n += 1; miss += not j["transit"]; touch += not c; ok += j["transit"] and j["wrong_dir_crossings"] == 0 and c
    print(f"{tag} {side}: {ok}/{n} success, missed {miss}, touched {touch}")
