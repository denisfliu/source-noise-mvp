"""Render-gap probe, stage 2 (openpi env, GPU). Compares features + predicted c on
STORED demo frames vs the stage-1 gsplat RE-RENDERS at identical poses, under:
  - ctx feature (post-fusion) with the lam100 map and the cfground map
  - pre-fusion feature (embed_prefix mean-pool) with the prefusion map
Key output: the render-induced Δc decoded into an implied commanded net
displacement (meters, raw units) — the direct test of the vertical-bias
hypothesis (all ctx maps commanded descent closed-loop; prefusion did not).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax.numpy as jnp
import jax
import gate_ctx_common as gc
from openpi.models import model as _model

FR = np.load(os.path.expanduser("~/ctxrun/rendergap_frames.npz"), allow_pickle=True)
ns, amean, astd = gc.load_norm()
eps = gc.load_eps(with_images=True)
byname = {f"ep_{i:04d}.npz": e for i, e in enumerate(eps)}
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
policy = gc.make_policy()


def prefusion_pool(raws):
    tds = [policy._input_transform(dict(r)) for r in raws]
    b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
    o = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
    tok, mask, _ = policy._model.embed_prefix(o)
    m = mask[..., None].astype(jnp.float32)
    return np.asarray((tok.astype(jnp.float32) * m).sum(1) / jnp.clip(m.sum(1), 1e-6)).astype(np.float32)


def batched(fn, obs, bs=8):
    return np.concatenate([fn(obs[i:i + bs]) for i in range(0, len(obs), bs)], 0)


stored_obs, rendered_obs = [], []
for k in range(len(FR["t"])):
    ep = byname[str(FR["ep_file"][k])]; t = int(FR["t"][k])
    stored_obs.append(gc.mkobs(ep, t))
    rendered_obs.append({"observation/image": FR["fwd224"][k], "observation/wrist_image": FR["wrist224"][k],
                         "observation/state": ep["state"][t], "prompt": ep["lang"]})

feats = {
    "ctx": (lambda o: gc.ctx_pool(policy, o)),
    "prefusion": prefusion_pool,
}
maps = {
    "ctx_lam100": ("ctx", gc.load_ridge(os.path.join(gc.RD, "vlmc_ridge_ctx_lam100.npz"))),
    "cfground": ("ctx", gc.load_ridge(os.path.join(gc.RD, "vlmc_ridge_ctx_cfground.npz"))),
    "prefusion": ("prefusion", gc.load_ridge(os.path.join(gc.RD, "vlmc_ridge_prefusion.npz"))),
}

phi = {}
for fname, fn in feats.items():
    phi[fname] = {"stored": batched(fn, stored_obs), "rendered": batched(fn, rendered_obs)}
    d = phi[fname]["rendered"] - phi[fname]["stored"]
    spread = phi[fname]["stored"].std(0).mean()
    print("FEATURE %-10s ||render shift|| %.2f  (per-dim rms %.4f vs in-dist per-dim std %.4f)"
          % (fname, np.linalg.norm(d, axis=1).mean(), d.std(), spread), flush=True)


def net_disp_raw(dc):
    """Decode a c-shift into implied commanded net displacement (raw units, dims x,y,z,yaw)."""
    chunk = (U @ dc).reshape(gc.H, gc.AD)[:, :4]
    return chunk.sum(0) * astd[:4]


for mname, (fname, m) in maps.items():
    cs = gc.apply_ridge(m, phi[fname]["stored"], clamp=True)
    cr = gc.apply_ridge(m, phi[fname]["rendered"], clamp=True)
    dc = cr - cs
    nd = np.stack([net_disp_raw(d) for d in dc])
    print("MAP %-10s |dc| %.2f+-%.2f   implied net-displacement bias [m]: "
          "x %+.3f+-%.3f  y %+.3f+-%.3f  z %+.3f+-%.3f  yaw %+.3f"
          % (mname, np.linalg.norm(dc, axis=1).mean(), np.linalg.norm(dc, axis=1).std(),
             nd[:, 0].mean(), nd[:, 0].std(), nd[:, 1].mean(), nd[:, 1].std(),
             nd[:, 2].mean(), nd[:, 2].std(), nd[:, 3].mean()), flush=True)
    # reference scale: demo chunk net displacement magnitude at these frames
    if mname == "ctx_lam100":
        nds = []
        for k in range(len(FR["t"])):
            ep = byname[str(FR["ep_file"][k])]; t = int(FR["t"][k])
            y = gc.segY(ep["action"][t:], amean, astd)
            nds.append((y.reshape(gc.H, gc.AD)[:, :4].sum(0) * astd[:4]))
        nds = np.stack(nds)
        print("  (reference: demo net displacement at these frames: |x| %.2f |y| %.2f |z| %.2f m)"
              % tuple(np.abs(nds[:, :3]).mean(0)), flush=True)
print("STAGE2_DONE", flush=True)
