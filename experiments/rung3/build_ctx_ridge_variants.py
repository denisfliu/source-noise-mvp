"""Build the two principled phi->c map variants from cached true+swapped-prompt
contextualized features (CPU-only; run with JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1).

Motivation (2026-08-04 ctx_steer_diag finding): prompt and scene are perfectly
confounded in data_gate_synth, so a ridge fit on true prompts leaves the
language direction of feature space UNIDENTIFIED — it maps to c-noise
(cos(prompt-swap axis, behavioral axis) ~ 0.01). Two fixes, neither scene-specific:

1. ABLATE (control): estimate the prompt-variation subspace from paired
   true/swap features, project it out, refit -> a map that is scene-driven by
   construction (language-INSENSITIVE). Isolates language-noise vs render-gap
   as the closed-loop divergence cause. Projection folds into W (same npz
   schema, no serving change).
2. CFGROUND (grounding): augment training with swapped-prompt features whose
   targets are PHASE-MATCHED counterfactuals (opposite side's mean c at the
   same episode-progress) -> the language direction is forced onto the
   behavioral axis.

Outputs: vlmc_ridge_ctx_ablate.npz, vlmc_ridge_ctx_cfground.npz (+RUN mirrors)
and offline metrics for all maps: held R2, prompt-swap alignment with the
behavioral axis, swap-shift magnitude.

env: RUN (/home/ubuntu/ctxrun), LAM (100), KABL (ablate subspace dim, 8),
     NPHASE (20).
"""
import os
import numpy as np
import gate_ctx_common as gc

RUN = os.environ.get("RUN", "/home/ubuntu/ctxrun")
LAM = float(os.environ.get("LAM", "100"))
KABL = int(os.environ.get("KABL", "8"))
NPHASE = int(os.environ.get("NPHASE", "20"))

ns, amean, astd = gc.load_norm()
eps = gc.load_eps(with_images=False)
recs = gc.make_recs(eps, amean, astd)
X = np.concatenate([np.load(f"{RUN}/Xshard_{k}.npy") for k in range(2)], 0)
Xs_ = np.concatenate([np.load(f"{RUN}/Xswapshard_{k}.npy") for k in range(2)], 0)
assert len(X) == len(recs) == len(Xs_)
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
Y = np.stack([r["Y"] for r in recs]).astype(np.float32)
C = Y @ U
tr = np.array([r["sp"] == "tr" for r in recs])
te = ~tr
side_L = np.array([eps[r["ei"]]["lang"] == gc.PROMPT_L for r in recs])
prog = np.array([r["t"] / max(1, len(eps[r["ei"]]["action"])) for r in recs])

bL, bR = C[tr & side_L].mean(0), C[tr & ~side_L].mean(0)
b = bL - bR; bhat = b / (np.linalg.norm(b) + 1e-9)
print("behavioral axis |b|=%.3f" % np.linalg.norm(b), flush=True)


def r2(p, y):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def report(name, m):
    cT = gc.apply_ridge(m, X); cS = gc.apply_ridge(m, Xs_)
    d = cS - cT                       # c-shift induced by swapping the prompt
    # sign-align: for left episodes swap means "say right" -> behavioral shift should be -b
    dal = np.where(side_L[:, None], -d, d)    # aligned so positive projection = correct grounding
    proj = dal[te] @ bhat
    print("%-10s heldR2=%+.3f  |swap-shift|=%.2f  grounding: proj-on-b %.2f+-%.2f "
          "(target |b|=%.2f)  cos(mean-shift,b)=%+.2f" %
          (name, r2(cT[te], C[te]), np.linalg.norm(d[te], axis=1).mean(),
           proj.mean(), proj.std(), np.linalg.norm(b),
           float(dal[te].mean(0) @ bhat / (np.linalg.norm(dal[te].mean(0)) + 1e-9))), flush=True)


# baseline (true-prompt ridge, same as extract_ctx_features build)
m0 = gc.fit_ridge(X, C, tr, LAM)
report("baseline", m0)

# 1) ABLATE: project the prompt-variation subspace out of standardized features
mu = X[tr].mean(0); sg = X[tr].std(0) + 1e-6
D = ((Xs_ - X) / sg)[tr]                      # standardized prompt deltas (train)
Dc = D - D.mean(0)
_, S, Vt = np.linalg.svd(Dc, full_matrices=False)
ev = (S ** 2) / (S ** 2).sum()
print("ablate: prompt-delta PCA explained var (top 10):", np.round(ev[:10], 3).tolist(), flush=True)
V = Vt[:KABL].T                               # (2048, KABL)
P = np.eye(X.shape[1], dtype=np.float32) - (V @ V.T).astype(np.float32)
Xp = mu + ((X - mu) / sg @ P) * sg            # ablated features in raw space (so fit_ridge re-standardizes consistently)
ma = gc.fit_ridge(Xp, C, tr, LAM)
# fold ablation into W so serving needs no change: c = ((x-mu')/sg')@P@W
ma["W"] = (P @ ma["W"]).astype(np.float32)
report("ablate", ma)

# 2) CFGROUND: augment with swapped features + phase-matched counterfactual targets
bins = np.clip((prog * NPHASE).astype(int), 0, NPHASE - 1)
cf = np.zeros_like(C)
for side in (True, False):
    src = tr & (side_L != side)               # opposite side, train only
    for k in range(NPHASE):
        mk = src & (bins == k)
        tgt = C[mk].mean(0) if mk.sum() else C[src].mean(0)
        cf[(side_L == side) & (bins == k)] = tgt
Xaug = np.concatenate([X, Xs_], 0)
Caug = np.concatenate([C, cf], 0)
traug = np.concatenate([tr, tr], 0)
mc = gc.fit_ridge(Xaug, Caug, traug, LAM)
report("cfground", mc)

for name, m in (("ablate", ma), ("cfground", mc)):
    out = os.path.join(gc.RD, f"vlmc_ridge_ctx_{name}.npz")
    np.savez(out, **m); np.savez(os.path.join(RUN, f"vlmc_ridge_ctx_{name}.npz"), **m)
    print("saved", out, flush=True)
print("VARIANTS_BUILD_DONE", flush=True)
