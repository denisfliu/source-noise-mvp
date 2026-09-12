"""Latency profile of the hardware pin server's per-replan path (2026-09-11). Loads the xswap checkpoint
exactly as serve_gate_pin_joint.py does (same env) and times each stage on one real frame:
input transform (CPU), head forward (prefix + MDN), flow at 10 vs 1 Euler steps (prefix + steps),
noise construction, CLOG save, and the msgpack request size."""
import os, sys, time, json
import numpy as np
RD = "/home/dfliu/code/source-noise-mvp/experiments/rung3"; sys.path.insert(0, RD)
os.environ.setdefault("SNMVP_HEAD", "1"); os.environ.setdefault("SNMVP_ZERO_PAD_ACTIONS", "1")
os.environ.setdefault("SNMVP_PIN_U", f"{RD}/pin_U_mh16.npy"); os.environ.setdefault("SNMVP_HEAD_DETACH", "0")
os.environ.setdefault("SNMVP_HEAD_LAM", "0.3"); os.environ.setdefault("SNMVP_HEAD_GMM", "1")
os.environ.setdefault("SNMVP_PIN_NOISE", "1.5"); os.environ.setdefault("SNMVP_PIN_NOISE_RAND", "1"); os.environ.setdefault("SNMVP_PIN_NOISE_COND", "1")
import jax, jax.numpy as jnp
from openpi.training import config as _cfg
from openpi.policies import policy_config as _pc
from openpi.shared import normalize as _nz
import joint_head
from openpi_client import msgpack_numpy
cfg = _cfg.get_config("pi0_gate")
raw_ns = _nz.load("/home/dfliu/hf_bundle/gate-drone-pi0/assets/gate_nav")
ns = {k: v for k, v in raw_ns.items()}
policy = _pc.create_trained_policy(cfg, "/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999", norm_stats=ns)
U = np.load(f"{RD}/pin_U_mh16.npy").astype(np.float32); H, AD = 50, 32
d = np.load(f"{RD}/data_gate_real/ep_0010.npz", allow_pickle=True)
from PIL import Image
r224 = lambda im: np.asarray(Image.fromarray(np.asarray(im)).resize((224, 224), Image.BICUBIC), np.uint8)
obs = {"observation/image": r224(d["image"][40]), "observation/wrist_image": r224(d["wrist"][40]),
       "observation/state": d["state"].astype(np.float32)[40], "prompt": "go through the gate on the right and hover over the stuffed animal"}
print("request payload (msgpack):", len(msgpack_numpy.packb(obs)) / 1e3, "kB")
def T(f, n=20, warm=2):
    for _ in range(warm): f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e3 * np.median(ts), 1e3 * np.percentile(ts, 90)
t0 = time.perf_counter(); joint_head.head_c(policy, [obs], return_gmm=True); print(f"head first call (compile): {time.perf_counter()-t0:.1f} s")
g = np.random.standard_normal((H, AD)).astype(np.float32).reshape(-1)
c = joint_head.head_c(policy, [obs])[0]
noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
t0 = time.perf_counter(); policy.infer(obs, noise=noise, snmvp_sigma=0.3); print(f"flow first call (compile): {time.perf_counter()-t0:.1f} s")
print("input_transform (CPU)      : %.1f ms (p90 %.1f)" % T(lambda: policy._input_transform(dict(obs))))
print("head_c (prefix + MDN)      : %.1f ms (p90 %.1f)" % T(lambda: joint_head.head_c(policy, [obs], return_gmm=True)))
print("policy.infer, 10 steps     : %.1f ms (p90 %.1f)" % T(lambda: policy.infer(obs, noise=noise, snmvp_sigma=0.3)))
# steps-only cost: the policy's own jitted sampler with num_steps passed as a traced int (one compile, reused)
from openpi.models import model as _model
inp = policy._input_transform(dict(obs)); b = jax.tree.map(lambda x: jnp.asarray(x)[None], inp)
o = _model.Observation.from_dict(b); rng = jax.random.key(0); nz = jnp.asarray(noise)[None]; sg = jnp.asarray([0.3], jnp.float32)
def run(n):
    return jax.block_until_ready(policy._sample_actions(rng, o, num_steps=jnp.asarray(n, jnp.int32), noise=nz, snmvp_sigma=sg))
t0 = time.perf_counter(); run(1); print(f"traced-num_steps sampler compile: {time.perf_counter()-t0:.1f} s")
m1 = T(lambda: run(1))[0]; m2 = T(lambda: run(2))[0]; m10 = T(lambda: run(10))[0]
print("sampler 1 / 2 / 10 steps   : %.1f / %.1f / %.1f ms -> per Euler step ~%.1f ms, prefix+overhead ~%.1f ms" % (m1, m2, m10, (m10 - m1) / 9, m1 - (m10 - m1) / 9))
print("noise construction (numpy) : %.2f ms" % T(lambda: (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32))[0])
log = [np.zeros(27, np.float32) for _ in range(120)]
print("CLOG np.save (120 rows)    : %.2f ms" % T(lambda: np.save("/tmp/claude-1002/-home-dfliu-code-source-noise-mvp/e0e9cb98-7501-4afd-a834-5584b07b276e/scratchpad/clog_prof.npy", np.stack(log)))[0])
print("msgpack encode reply       : %.2f ms" % T(lambda: msgpack_numpy.packb({"actions": np.zeros((50, 32), np.float32), "state": obs["observation/state"]}))[0])
print("msgpack decode request     : %.2f ms" % T(lambda: msgpack_numpy.unpackb(msgpack_numpy.packb(obs)))[0])
