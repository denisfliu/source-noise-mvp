"""Fused-serve check (2026-09-11): (a) equivalence — head outputs and flow actions from the fused path
(one prefix pass, KV cache reused) vs the two-pass path, same noise and sigma, on real and synth frames;
(b) latency of each path."""
import os, sys, time
import numpy as np
RD = "/home/dfliu/code/source-noise-mvp/experiments/rung3"; sys.path.insert(0, RD)
for k, v in dict(SNMVP_HEAD="1", SNMVP_ZERO_PAD_ACTIONS="1", SNMVP_PIN_U=f"{RD}/pin_U_mh16.npy", SNMVP_HEAD_DETACH="0",
                 SNMVP_HEAD_LAM="0.3", SNMVP_HEAD_GMM="1", SNMVP_PIN_NOISE="1.5", SNMVP_PIN_NOISE_RAND="1", SNMVP_PIN_NOISE_COND="1").items():
    os.environ.setdefault(k, v)
import jax
from openpi.training import config as _cfg
from openpi.policies import policy_config as _pc
from openpi.shared import normalize as _nz
import joint_head
from PIL import Image
cfg = _cfg.get_config("pi0_gate")
ns = dict(_nz.load("/home/dfliu/hf_bundle/gate-drone-pi0/assets/gate_nav"))
policy = _pc.create_trained_policy(cfg, "/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999", norm_stats=ns)
U = np.load(f"{RD}/pin_U_mh16.npy").astype(np.float32); H, AD = 50, 32
astd = np.asarray(ns["actions"].std, np.float32)[:7] if hasattr(ns["actions"], "std") else None
r224 = lambda im: np.asarray(Image.fromarray(np.asarray(im)).resize((224, 224), Image.BICUBIC), np.uint8)
P = {"left": "go through the gate on the left and hover over the stuffed animal", "right": "go through the gate on the right and hover over the stuffed animal",
     "cfl": "go through the center gate from the left and hover over the stuffed animal", "cfr": "go through the center gate from the right and hover over the stuffed animal"}
frames = []
for dom, eps, tasks in [("real", [3, 27, 61, 88], ["left", "left", "right", "right"]), ("synth", [5, 60, 120, 170], ["cfl", "cfr", "left", "right"])]:
    for e, task in zip(eps, tasks):
        d = np.load(f"{RD}/data_gate_{dom}{'' if dom == 'real' else '3'}/ep_{e:04d}.npz", allow_pickle=True)
        for t in (20, 90):
            frames.append((f"{dom} ep{e} t{t}", {"observation/image": r224(d["image"][t]), "observation/wrist_image": r224(d["wrist"][t]),
                                                  "observation/state": d["state"].astype(np.float32)[t], "prompt": P[task]}))
rng = np.random.default_rng(0)
rows = []
for name, obs in frames:
    c1, w1, mu1, sig1 = joint_head.head_c(policy, [obs], return_gmm=True)
    c2, w2, mu2, sig2, cache = joint_head.head_c(policy, [obs], return_gmm=True, return_cache=True)
    g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
    c = mu1[0, int(w1[0].argmax())]
    noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
    a1 = np.asarray(policy.infer(obs, noise=noise, snmvp_sigma=0.3)["actions"], np.float32)
    a2 = np.asarray(policy.infer(obs, noise=noise, snmvp_sigma=0.3, cache=cache)["actions"], np.float32)
    # compare in normalized units (what the flow produces) via the norm stats used by the server
    d_act = np.abs(a1 - a2)[:, :7]
    rows.append((name, int(w1[0].argmax()), int(w2[0].argmax()), np.abs(w1 - w2).max(), np.abs(mu1 - mu2).max(), np.abs(sig1 - sig2).max(), d_act.max(), d_act.mean(), np.abs(a1[:, :7]).mean()))
print(f"{'frame':16s} comp(2-pass/fused) |dpi|max |dmu|max |dsig|max |dact|max(m/step) |dact|mean |act|mean")
for r in rows:
    print(f"{r[0]:16s} {r[1]}/{r[2]}  {r[3]:.2e}  {r[4]:.2e}  {r[5]:.2e}  {r[6]:.2e}  {r[7]:.2e}  {r[8]:.3f}")
# projected-command check: the executed chunk's coarse part vs c, both paths (in c units)
def T(f, n=20):
    for _ in range(3): f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e3 * np.median(ts)
name, obs = frames[0]
c1, w1, mu1, sig1 = joint_head.head_c(policy, [obs], return_gmm=True); c = mu1[0, int(w1[0].argmax())]
g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1); noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
def two_pass():
    joint_head.head_c(policy, [obs], return_gmm=True); policy.infer(obs, noise=noise, snmvp_sigma=0.3)
def fused():
    *_, cache = joint_head.head_c(policy, [obs], return_gmm=True, return_cache=True); policy.infer(obs, noise=noise, snmvp_sigma=0.3, cache=cache)
print(f"latency per replan: two-pass {T(two_pass):.1f} ms | fused {T(fused):.1f} ms")
print(f"  fused head+prefix {T(lambda: joint_head.head_c(policy, [obs], return_gmm=True, return_cache=True)):.1f} ms, cached denoise {T(lambda: policy.infer(obs, noise=noise, snmvp_sigma=0.3, cache=joint_head.head_c(policy, [obs], return_gmm=True, return_cache=True)[-1])) - T(lambda: joint_head.head_c(policy, [obs], return_gmm=True, return_cache=True)):.1f} ms")
