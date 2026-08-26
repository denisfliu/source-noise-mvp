"""Contextualized feature extraction + ridge build over the TRAJECTORY-ALGEBRA
augmented synth set (for the aug c-map that serves the retrained flow).

Key properties:
- Split is grouped by SOURCE episode (same rng(0) perm over the 200 synth eps as
  every other consumer) — augmented variants reuse the same stored frames, so an
  ungrouped split would leak test images into train.
- The aug set contains REAL within-scene language contrast by construction:
  a forward rec at frame t and the reverse rec at the mirrored index share the
  IDENTICAL stored image with different prompts and different targets. Build
  mode exploits these pairs to report fwd/back grounding alignment offline
  (no extra forward passes, no synthetic counterfactual targets).

env MODE extract|build; SHARD_K/SHARD_N (extract); STRIDE (12); LAM (100);
    RUN (~/ctxrun). Outputs RUN/Xaugshard_K.npy; build -> rung3/vlmc_ridge_aug.npz.
env OBS=stored (default) | rendered: 'rendered' swaps every obs image for the
    gsplat re-render of the same pose through the SERVING chain (render_aug_frames.py
    output, RUN/rendered_frames.npz) — the domain-matching fix: fit the c-map on
    what the policy actually sees in flight. Shards Xrendshard_K, map vlmc_ridge_rend.npz.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import gate_traj_algebra as ta

RUN = os.environ.get("RUN", os.path.expanduser("~/ctxrun"))
MODE = os.environ.get("MODE", "extract")
OBS = os.environ.get("OBS", "stored")
PROMPTS = os.environ.get("PROMPTS", "true")  # 'swaplr': ORIG rows only, L<->R prompt swapped
TAG = ("Xaugshard" if OBS == "stored" else "Xrendshard") if PROMPTS == "true" else "Xrendlrshard"
MAPOUT = "vlmc_ridge_aug.npz" if OBS == "stored" else "vlmc_ridge_rend.npz"
SHARD_N = int(os.environ.get("SHARD_N", "1")); SHARD_K = int(os.environ.get("SHARD_K", "0"))
STRIDE = int(os.environ.get("STRIDE", "12")); LAM = float(os.environ.get("LAM", "100"))

ns, amean, astd = gc.load_norm()
src = gc.load_eps(with_images=(MODE == "extract" and OBS == "stored"))
rng = np.random.default_rng(0)
idx = rng.permutation(len(src)); ntr = int(0.8 * len(src)); trep = set(idx[:ntr].tolist())

groups = []  # (source_idx, variant, ep)
for si, e in enumerate(src):
    groups.append((si, "orig", e))
    groups.append((si, "reverse", ta.reverse(e)))
    for nm, f in (("crop_to", ta.crop_to_gate), ("crop_from", ta.crop_from_gate)):
        a = f(e)
        if a is not None:
            groups.append((si, nm, a))
    groups.append((si, "hover", ta.hover(e, len(e["action"]) // 2)))

recs = []
for si, variant, e in groups:
    n = min(len(e["action"]), len(e["state"]) - 1)
    for t in range(0, n, STRIDE):
        recs.append({"si": si, "variant": variant, "ep": e, "t": t,
                     "fidx": int(e["fidx"][t]) if "fidx" in e else t,
                     "sp": "tr" if si in trep else "te",
                     "Y": gc.segY(e["action"][t:], amean, astd)})
if PROMPTS == "swaplr":
    assert OBS == "rendered", "swaplr only implemented for OBS=rendered (stored branch ignores _prompt; TAG would lie)"
    recs = [r for r in recs if r["variant"] == "orig"]
print("aug recs", len(recs), "MODE", MODE, "shard", SHARD_K, "/", SHARD_N, "prompts", PROMPTS, flush=True)

if MODE == "extract":
    if OBS == "rendered":
        rf = np.load(f"{RUN}/rendered_frames.npz")
        row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rf["si"], rf["fidx"]))}
        # materialize once: NpzFile re-decompresses the full array on EVERY access,
        # and a row view pins that fresh ~1.2GB parent alive -> ~2.3GB leaked per record
        fwd224, wrist224 = rf["fwd224"], rf["wrist224"]
        # swaplr: each task swaps with its mirror-side counterpart (gate<->gate,
        # center<->center) — never across families (a center row must not get a
        # gate prompt; that would manufacture false counterfactuals)
        SWAP = {gc.PROMPT_L: gc.PROMPT_R, gc.PROMPT_R: gc.PROMPT_L,
                gc.PROMPT_CFL: gc.PROMPT_CFR, gc.PROMPT_CFR: gc.PROMPT_CFL}
        def _prompt(r):
            if PROMPTS == "swaplr":
                return SWAP[r["ep"]["lang"]]
            return r["ep"]["lang"]
        def obs_of(r):
            i = row[(r["si"], r["fidx"])]
            return {"observation/image": fwd224[i], "observation/wrist_image": wrist224[i],
                    "observation/state": r["ep"]["state"][r["t"]], "prompt": _prompt(r)}
    else:
        def obs_of(r):
            return gc.mkobs(r["ep"], r["t"])
    policy = gc.make_policy()
    per = (len(recs) + SHARD_N - 1) // SHARD_N
    lo = SHARD_K * per; hi = min(len(recs), lo + per)
    X = gc.feats(policy, [obs_of(r) for r in recs[lo:hi]], log_every=20)
    np.save(f"{RUN}/{TAG}_{SHARD_K}.npy", X)
    print("AUGSHARD_%d_DONE %d-%d %s obs=%s" % (SHARD_K, lo, hi, X.shape, OBS), flush=True)
    raise SystemExit

X = np.concatenate([np.load(f"{RUN}/{TAG}_{k}.npy") for k in range(SHARD_N)], 0)
assert len(X) == len(recs), (len(X), len(recs))
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
Y = np.stack([r["Y"] for r in recs]).astype(np.float32); C = Y @ U
tr = np.array([r["sp"] == "tr" for r in recs]); te = ~tr

def r2(p, y):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))

m = gc.fit_ridge(X, C, tr, LAM)
pt = gc.apply_ridge(m, X[te])
print("lam=%g held c-R2 (all aug rows) %+.3f  maxW %.3f" % (LAM, r2(pt, C[te]), np.abs(m["W"]).max()), flush=True)
vt = np.array([r["variant"] for r in recs])
for v in ("orig", "reverse", "crop_to", "crop_from", "hover"):
    mk = te & (vt == v)
    if mk.any():
        print("  held R2 %-9s %+.3f (n=%d)" % (v, r2(gc.apply_ridge(m, X[mk]), C[mk]), mk.sum()), flush=True)

# fwd/back grounding from IDENTICAL-FRAME pairs (orig frame t vs reverse frame n-t)
bykey = {}
for i, r in enumerate(recs):
    n = min(len(r["ep"]["action"]), len(r["ep"]["state"]) - 1)
    if r["variant"] == "orig":
        bykey[(r["si"], r["t"])] = i
pairs = []
for i, r in enumerate(recs):
    if r["variant"] != "reverse" or r["sp"] != "te":
        continue
    n = min(len(r["ep"]["action"]), len(r["ep"]["state"]) - 1)
    j = bykey.get((r["si"], n - r["t"]))
    if j is not None:
        pairs.append((j, i))
if pairs:
    cF = gc.apply_ridge(m, X[[j for j, _ in pairs]])
    cB = gc.apply_ridge(m, X[[i for _, i in pairs]])
    cFt = C[[j for j, _ in pairs]]; cBt = C[[i for _, i in pairs]]
    b = (cFt - cBt).mean(0); bhat = b / (np.linalg.norm(b) + 1e-9)
    d = (cF - cB).mean(0)
    print("FWD/BACK grounding (%d identical-frame held pairs): |target axis| %.2f  "
          "cos(pred axis, target axis) %+.3f  |pred|/|target| %.2f"
          % (len(pairs), np.linalg.norm(b), float(d @ bhat / (np.linalg.norm(d) + 1e-9)),
             np.linalg.norm(d) / (np.linalg.norm(b) + 1e-9)), flush=True)
out = os.path.join(gc.RD, MAPOUT)
np.savez(out, **m); np.savez(os.path.join(RUN, MAPOUT), **m)
print("saved", out, flush=True)
print("AUG_BUILD_DONE", flush=True)
