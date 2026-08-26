"""Quick de-risk before the G-predict battery: does the point_phase(theta) knob
actually spread cross-body coherence c(B,setA) on the S_A bins, and do the
bodies still clear the obstacle (ceiling)? ~30s CPU."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embodiments as emb
import mb_dataset as ds
import flow_embod as fe
import coherence_xembod as cx

THETAS = np.linspace(0.0, np.pi, 90)
LADDER = ["point_phase0", "point_phase15", "point_phase30", "point_phase45"]

bodies = emb.make_bodies()
scenes, obs, angles, chunks = ds.make_dataset(bodies, 120, 8, np.random.default_rng(7))
S_A, _ = fe.freeze_frame(chunks, angles)
print("S_A (%d pins):" % len(S_A), [(p["axis"], p["omega"], p["mode"], p.get("mag")) for p in S_A])
sel = [{"axis_deg": (0.0 if p["axis"][0] else 90.0), "omega": p["omega"], "mode": p["mode"]}
       for p in S_A]

print(f"{'body':14} {'pooled_c':>10} {'align':>8} {'ceiling':>8}")
for B in LADDER:
    cs = cx.pairwise_c(chunks, angles, fe.SET_A + [B], THETAS, sel)
    c = float(np.mean([v for k, v in cs.items() if B in k]))
    al = cx.align_to_consensus(chunks, angles, fe.SET_A, B, S_A)
    ceil = float(np.mean([ds.success(scenes[s], chunks[B][s, d])
                          for s in range(len(scenes)) for d in range(8)]))
    print(f"{B:14} {c:10.3f} {al:8.3f} {ceil:8.3f}")
print("CHECK_DONE=ok")
