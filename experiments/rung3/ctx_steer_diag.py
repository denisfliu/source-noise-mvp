"""Balanced instruction-steer diagnostic for a phi->c map (contextualized VLM-c).

On a side-BALANCED sample of held-out frames, measures whether the instruction
alone moves c, and whether it moves c along the BEHAVIORAL left/right axis
(the direction separating demo chunk-coords of left vs right episodes) —
i.e. is the language signal present AND pointed the right way. Reusable for
any instruction minimal pair / any ridge map (env WC).

env: WC (ridge npz, default /tmp/vlmc_ridge_ctx.npz), NPS (frames per side, 24),
     CKPT (policy checkpoint override).
"""
import os
import numpy as np
import gate_ctx_common as gc

WC = os.environ.get("WC", "/tmp/vlmc_ridge_ctx.npz")
NPS = int(os.environ.get("NPS", "24"))

ns, amean, astd = gc.load_norm()
eps = gc.load_eps(with_images=True)
recs = gc.make_recs(eps, amean, astd)
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))

# behavioral L/R axis from demo chunk coords (train episodes only)
C = np.stack([r["Y"] for r in recs]).astype(np.float32) @ U
tr = np.array([r["sp"] == "tr" for r in recs])
side_L = np.array([eps[r["ei"]]["lang"] == gc.PROMPT_L for r in recs])
bL, bR = C[tr & side_L].mean(0), C[tr & ~side_L].mean(0)
b = bL - bR; bhat = b / (np.linalg.norm(b) + 1e-9)
print("behavioral axis |c_L-c_R| (train demos) = %.3f  dir %s" %
      (np.linalg.norm(b), np.round(bhat, 2).tolist()), flush=True)

# balanced held-out sample
te = [i for i, r in enumerate(recs) if r["sp"] == "te"]
subL = [i for i in te if side_L[i]][:NPS]
subR = [i for i in te if not side_L[i]][:NPS]
print("balanced sample: %d left frames, %d right frames" % (len(subL), len(subR)), flush=True)
sub = subL + subR

policy = gc.make_policy(os.environ.get("CKPT"))
m = gc.load_ridge(WC)
def cof(prompt=None):
    obs = [gc.mkobs(eps[recs[i]["ei"]], recs[i]["t"], prompt) for i in sub]
    return gc.apply_ridge(m, gc.feats(policy, obs))

cl, cr, ct = cof(gc.PROMPT_L), cof(gc.PROMPT_R), cof(None)
d = cl - cr; dm = d.mean(0)
cons = float(np.mean(d @ dm / (np.linalg.norm(d, axis=1) * np.linalg.norm(dm) + 1e-9)))
print("PROMPT-SWAP ||c_L-c_R|| mean=%.3f  per-dim |d| %s  direction-consistency %.2f" %
      (np.linalg.norm(d, axis=1).mean(), np.round(np.abs(d).mean(0), 2).tolist(), cons), flush=True)
print("ALIGNMENT cos(prompt-swap axis, behavioral axis) = %+.3f" %
      float(dm @ bhat / (np.linalg.norm(dm) + 1e-9)), flush=True)
print("MAGNITUDE |prompt-swap| / |behavioral| = %.2f" %
      (np.linalg.norm(dm) / (np.linalg.norm(b) + 1e-9)), flush=True)

# does c under the TRUE prompt separate sides along bhat?
nL = len(subL)
pL, pR = ct[:nL] @ bhat, ct[nL:] @ bhat
pooled = np.sqrt(0.5 * (pL.std() ** 2 + pR.std() ** 2)) + 1e-9
print("TRUE-PROMPT separation along axis: left %.2f+-%.2f  right %.2f+-%.2f  d'=%.2f "
      "(demo means: left %.2f right %.2f)" %
      (pL.mean(), pL.std(), pR.mean(), pR.std(), (pL.mean() - pR.mean()) / pooled,
       bL @ bhat, bR @ bhat), flush=True)

# ridge-target ceiling: same projections for the DEMO coords of these frames
qL, qR = C[subL] @ bhat, C[subR] @ bhat
print("DEMO-TARGET separation on same frames: left %.2f+-%.2f right %.2f+-%.2f" %
      (qL.mean(), qL.std(), qR.mean(), qR.std()), flush=True)
print("STEER_DIAG_DONE", flush=True)
