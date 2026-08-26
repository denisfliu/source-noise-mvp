"""c-along-rollout analysis (CPU): the interpretability panel for the gate pin.

For each instrumented rollout (clog_<map>_<side>.jsonl from serve_gate_pin_vlmc
with SNMVP_C_LOG): decode each chunk's commanded c into intended net displacement
(meters), align with the actually-visited pose, and measure how command quality
evolves with DRIFT = distance from the nearest same-side demo state. Answers:
does the sinking command exist from chunk 0 (static bias) or grow with drift
(compounding OOD)? And what did the successful prefusion flight's command curve
look like instead?
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc

ns, amean, astd = gc.load_norm()
eps = gc.load_eps(with_images=False)
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
RUN = os.path.expanduser("~/ctxrun")

demo = {"left": np.concatenate([e["state"][:, :3] for e in eps if e["lang"] == gc.PROMPT_L]),
        "right": np.concatenate([e["state"][:, :3] for e in eps if e["lang"] == gc.PROMPT_R])}
# demo commanded net displacement per progress bin (what a healthy command looks like)
def demo_cmd(side):
    rows = []
    for e in eps:
        if (e["lang"] == gc.PROMPT_L) != (side == "left"):
            continue
        for t in range(0, len(e["action"]), 16):
            y = gc.segY(e["action"][t:], amean, astd)
            rows.append((t / max(1, len(e["action"])), (y @ U @ U.T).reshape(gc.H, gc.AD)[:, :4].sum(0) * astd[:4]))
    return rows

def decode(c):
    return (U @ np.asarray(c)).reshape(gc.H, gc.AD)[:, :4].sum(0) * astd[:4]

for f in sorted(glob.glob(f"{RUN}/clog_*_*.jsonl")):
    name = os.path.basename(f)[5:-6]
    side = "left" if name.endswith("left") else "right"
    rows = [json.loads(l) for l in open(f)]
    if not rows:
        print(name, "EMPTY"); continue
    P = np.array([r["state"][:3] for r in rows])
    D = np.array([np.linalg.norm(demo[side] - p, axis=1).min() for p in P])
    V = np.stack([decode(r["c"]) for r in rows])
    print("\n=== %s  (%d chunks) ===" % (name, len(rows)), flush=True)
    print(" chunk |    pos x     y     z | drift[m] | commanded dx    dy    dz")
    for i in range(0, len(rows), 5):
        print(" %5d | %+7.2f %+5.2f %+5.2f |   %5.2f  |    %+5.2f  %+5.2f  %+5.2f"
              % (i, *P[i], D[i], *V[i, :3]))
    # summary: command-z in the near-demo regime vs drifted regime
    near, far = D < 0.25, D >= 0.5
    print(" cmd-z: chunks 0-2 %+0.2f | near-demo(<0.25m, n=%d) %+0.2f | drifted(>=0.5m, n=%d) %+0.2f"
          % (V[:3, 2].mean(), near.sum(), V[near, 2].mean() if near.any() else float("nan"),
             far.sum(), V[far, 2].mean() if far.any() else float("nan")), flush=True)
    if len(rows) > 4:
        cz = np.corrcoef(D, V[:, 2])[0, 1]
        cm = np.corrcoef(D, np.linalg.norm(V[:, :3], axis=1))[0, 1]
        print(" corr(drift, cmd-z) %+0.2f   corr(drift, |cmd|) %+0.2f" % (cz, cm), flush=True)
ref = demo_cmd("left")
zs = [v[2] for _, v in ref]
print("\nreference healthy cmd-z (left demos, all phases): mean %+0.3f  p10 %+0.2f p90 %+0.2f"
      % (np.mean(zs), np.percentile(zs, 10), np.percentile(zs, 90)), flush=True)
print("ANALYZE_DONE", flush=True)
