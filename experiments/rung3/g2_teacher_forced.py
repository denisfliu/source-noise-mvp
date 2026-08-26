"""G2 gate: teacher-forced whole-trajectory evaluation of the augmented retrain.

For held-out (frozen-split) synth episodes, at sampled chunk starts: build the
obs from STORED frames, pin the noise with the chunk's TRUE c, and compare the
policy's emitted chunk against the true continuation — per continuation type
(forward / reverse / hover) and per model (aug retrain vs original RRR flow).

B1 lesson baked in: pinned-coordinate err is confounded by passthrough; the
metrics here are whole-trajectory — chunk ADE in meters over the first EVAL_T
steps, plus net-displacement error, plus (for hover) absolute drift.

Pre-registered bars (report only, never auto-pass):
  G2.1 aug reverse ADE <= 2x aug forward ADE
  G2.2 aug hover net-drift < 0.05 m
  G2.3 forward non-regression: aug fwd ADE <= 1.3x baseline fwd ADE

env: AUG_CKPT (default checkpoints/pi0_gate_aug/gate_aug_pin_rrr/4999),
     BASE_CKPT (default .../pi0_gate/gate_both_pin_rrr/4999), N_EP (12), N_T (6).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import gate_traj_algebra as ta

AUG_CKPT = os.environ.get("AUG_CKPT", os.path.expanduser(
    "~/code/openpi/checkpoints/pi0_gate_aug/gate_aug_pin_rrr/4999"))
BASE_CKPT = os.environ.get("BASE_CKPT", os.path.expanduser(
    "~/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999"))
N_EP = int(os.environ.get("N_EP", "12")); N_T = int(os.environ.get("N_T", "6"))
EVAL_T = 25

ns, amean, astd = gc.load_norm()
eps = gc.load_eps(with_images=True)
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
rng = np.random.default_rng(0)
idx = rng.permutation(len(eps)); ntr = int(0.8 * len(eps))
test_src = [eps[i] for i in idx[ntr:][:N_EP]]

def variants(src):
    out = {"forward": src, "reverse": ta.reverse(src)}
    h = ta.hover(src, len(src["action"]) // 2)
    out["hover"] = h
    return out

def true_chunk_raw(e, t):
    seg = e["action"][t:t + gc.H, :4]
    if len(seg) < gc.H:
        seg = np.concatenate([seg, np.zeros((gc.H - len(seg), 4), np.float32)], 0)
    return seg

def eval_model(name, ckpt):
    policy = gc.make_policy(ckpt)
    res = {}
    for e in test_src:
        for vname, ve in variants(e).items():
            n = min(len(ve["action"]), len(ve["state"]) - 1)
            ts = np.linspace(0, max(0, n - gc.H), N_T).astype(int) if vname != "hover" else [0, 5]
            for t in ts:
                obs = gc.mkobs(ve, t)
                c = gc.segY(ve["action"][t:], amean, astd) @ U
                g = np.random.default_rng(int(t) + 1).standard_normal((gc.H, gc.AD)).astype(np.float32).reshape(-1)
                noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(gc.H, gc.AD).astype(np.float32)
                pred = np.asarray(policy.infer(obs, noise=noise)["actions"])[:, :4]
                true = true_chunk_raw(ve, t)
                m = min(EVAL_T, max(1, n - t))
                pp, tp = np.cumsum(pred[:m, :3], 0), np.cumsum(true[:m, :3], 0)
                ade = float(np.linalg.norm(pp - tp, axis=1).mean())
                nde = float(np.linalg.norm(pred[:m, :3].sum(0) - true[:m, :3].sum(0)))
                drift = float(np.linalg.norm(pred[:m, :3].sum(0)))
                res.setdefault(vname, []).append((ade, nde, drift))
    print("== %s (%s)" % (name, ckpt), flush=True)
    out = {}
    for vname, rows in res.items():
        a = np.array(rows)
        out[vname] = a[:, 0].mean()
        print("  %-8s n=%3d  ADE %.3f+-%.3f m   net-disp err %.3f m   |net-disp| %.3f m"
              % (vname, len(rows), a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 2].mean()), flush=True)
    return out, {v: np.array(r) for v, r in res.items()}

aug, aug_raw = eval_model("AUG", AUG_CKPT)
base, base_raw = eval_model("BASELINE", BASE_CKPT)
hover_drift = aug_raw["hover"][:, 2].mean()
print("G2.1 aug reverse ADE <= 2x aug fwd: %.3f vs %.3f  %s"
      % (aug["reverse"], 2 * aug["forward"], "PASS" if aug["reverse"] <= 2 * aug["forward"] else "FAIL"), flush=True)
print("G2.2 aug hover |net-disp| < 0.05 m: %.3f  %s"
      % (hover_drift, "PASS" if hover_drift < 0.05 else "FAIL"), flush=True)
print("G2.3 fwd non-regression aug <= 1.3x base: %.3f vs %.3f  %s"
      % (aug["forward"], 1.3 * base["forward"], "PASS" if aug["forward"] <= 1.3 * base["forward"] else "FAIL"), flush=True)
print("  (baseline on reverse/hover, for the record: rev ADE %.3f, hover drift %.3f m)"
      % (base["reverse"], base_raw["hover"][:, 2].mean()), flush=True)
print("G2_DONE", flush=True)
