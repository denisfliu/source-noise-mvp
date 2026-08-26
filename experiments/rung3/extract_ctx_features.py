"""Contextualized (post-fusion) pi0 prefix feature extraction + ridge phi->c build, SHARDED.

Same U (pin_U_gate_rrr_k5) so the trained RRR flow stays valid; only re-grounds c.
Replaces the /tmp/extract_ctx2.py one-off (whose build-mode steer diagnostic
crashed; the balanced diagnostic now lives in ctx_steer_diag.py).

env MODE:
  'extract' — this shard (SHARD_K of SHARD_N) -> RUN/Xshard_K.npy   (GPU)
  'build'   — concat shards, sweep lambdas, save ridge map at LAM   (CPU ok)
env: RUN (default /home/ubuntu/ctxrun), SHARD_N/SHARD_K, LAM (default 100),
     OUT (default /tmp/vlmc_ridge_ctx.npz; also mirrored into RUN/),
     PROMPTS ('true' default; 'swap' extracts with the OPPOSITE prompt per
     episode -> RUN/Xswapshard_K.npy — counterfactual features for grounding
     the language direction; see ctx_steer_diag alignment finding 2026-08-04).
"""
import os
import numpy as np
import gate_ctx_common as gc

RUN = os.environ.get("RUN", "/home/ubuntu/ctxrun")
MODE = os.environ.get("MODE", "extract")
SHARD_N = int(os.environ.get("SHARD_N", "2")); SHARD_K = int(os.environ.get("SHARD_K", "0"))
LAM = float(os.environ.get("LAM", "100")); OUT = os.environ.get("OUT", "/tmp/vlmc_ridge_ctx.npz")

ns, amean, astd = gc.load_norm()
eps = gc.load_eps(with_images=(MODE == "extract"))
recs = gc.make_recs(eps, amean, astd)
print("recs", len(recs), "MODE", MODE, "shard", SHARD_K, "/", SHARD_N, flush=True)

PROMPTS = os.environ.get("PROMPTS", "true")

def prompt_for(ep):
    if PROMPTS == "swap":
        return gc.PROMPT_R if ep["lang"] == gc.PROMPT_L else gc.PROMPT_L
    return ep["lang"]

if MODE == "extract":
    policy = gc.make_policy()
    per = (len(recs) + SHARD_N - 1) // SHARD_N
    lo = SHARD_K * per; hi = min(len(recs), lo + per)
    obs = [gc.mkobs(eps[r["ei"]], r["t"], prompt_for(eps[r["ei"]])) for r in recs[lo:hi]]
    X = gc.feats(policy, obs, log_every=20)
    tag = "Xswapshard" if PROMPTS == "swap" else "Xshard"
    np.save(f"{RUN}/{tag}_{SHARD_K}.npy", X)
    print("SHARD_%d_DONE %d-%d %s prompts=%s" % (SHARD_K, lo, hi, X.shape, PROMPTS), flush=True)
    raise SystemExit

# MODE build (pure numpy; run with JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1)
X = np.concatenate([np.load(f"{RUN}/Xshard_{k}.npy") for k in range(SHARD_N)], 0)
print("merged", X.shape, flush=True)
assert len(X) == len(recs), (len(X), len(recs))
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
Y = np.stack([r["Y"] for r in recs]).astype(np.float32)
sp = np.array([r["sp"] for r in recs]); tr, te = sp == "tr", sp == "te"
C = Y @ U

def r2(p, y):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))

for lam in [10., 100., 1000.]:
    m = gc.fit_ridge(X, C, tr, lam)
    print("lam=%-6g held c-R2=%+.3f maxW=%.3f" %
          (lam, r2(gc.apply_ridge(m, X[te]), C[te]), np.abs(m["W"]).max()), flush=True)
m = gc.fit_ridge(X, C, tr, LAM)
for path in (OUT, os.path.join(RUN, os.path.basename(OUT))):
    np.savez(path, **m)
print("saved LAM=%g -> %s (+ mirror in %s)" % (LAM, OUT, RUN), flush=True)
print("CTX_BUILD_DONE", flush=True)
