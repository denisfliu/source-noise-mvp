"""x_t through denoise time (2026-09-23): step the flow ourselves and record every Euler state for a plain
Gaussian source vs coarse-pinned sources on ONE observation, then draw each chunk's xy path at several t.

  <PINENV> python xt_trace.py --ckpt <flow>/4999 --norm <assets>/gate_nav --pin-u pin_U_mh16.npy \
      --frame ~/gate_flights/agent_log/agentmann_pin_02/0024 --out xt_trace.npz
Sources: plain (z = eps, served like SNMVP_PIN_OFF at sigma 1.5), head (the checkpoint's own c), and named
primitive moves converted with AgentMove exactly as the agent server does. Records x_t at t = 1.0 .. 0.0."""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp, einops
from PIL import Image
RD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, RD)
import serve_gate_pin_joint as S            # noqa: E402  (module-level imports: _cfg, _nz, _pad, _pc)
import joint_head                            # noqa: E402
from agent_prompt import AgentMove, H, AD    # noqa: E402
import openpi.models.model as _model         # noqa: E402
from openpi.models.pi0 import make_attn_mask # noqa: E402

def load_frame(prefix, prompt):
    j = json.load(open(prefix + ".json")); pose = np.asarray(j["pose"], np.float32).reshape(-1)
    st = np.zeros(7, np.float32); st[:min(7, len(pose))] = pose[:7]
    return {"observation/image": np.asarray(Image.open(prefix + "_front.png").convert("RGB"), np.uint8),
            "observation/wrist_image": np.asarray(Image.open(prefix + "_down.png").convert("RGB"), np.uint8),
            "observation/state": st, "prompt": prompt or j.get("prompt", "")}, pose[:4]

def trace(policy, obs, noise, sigma, num_steps=10):
    """List of x_t (H, AD) at t = 1, 1-dt, ..., 0, stepping exactly as pi0._snmvp_denoise does."""
    m = policy._model
    inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    o = _model.preprocess_observation(None, _model.Observation.from_dict(inputs), train=False)
    _, prefix_mask, kv_cache = m.snmvp_prefix(o, preprocessed=True)
    sig = jnp.full((1,), float(sigma), jnp.float32)
    x = jnp.asarray(noise, jnp.float32)[None]; t = 1.0; dt = -1.0 / num_steps; out = [np.asarray(x[0])]
    while t >= -dt / 2:
        st, sm, sam, cond = m.embed_suffix(o, x, jnp.broadcast_to(jnp.asarray(t, jnp.float32), (1,)), snmvp_sigma=sig)
        sa = make_attn_mask(sm, sam); pa = einops.repeat(prefix_mask, "b p -> b s p", s=st.shape[1])
        full = jnp.concatenate([pa, sa], axis=-1); pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        (_, so), _ = m.PaliGemma.llm([None, st], mask=full, positions=pos, kv_cache=kv_cache, adarms_cond=[None, cond])
        v = m.action_out_proj(so[:, -m.action_horizon:]); x = x + dt * v; t += dt; out.append(np.asarray(x[0]))
    return np.stack(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True); ap.add_argument("--frame", required=True); ap.add_argument("--prompt", default="")
    ap.add_argument("--out", default="xt_trace.npz"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = S._cfg.get_config(a.config); raw = S._nz.load(a.norm); ns = S._pad(raw, cfg.model.action_dim)
    policy = S._pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    U = np.load(a.pin_u).astype(np.float32); amean = np.asarray(raw["actions"].mean[:7], np.float32); astd = np.asarray(raw["actions"].std[:7], np.float32)
    mover = AgentMove(amean, astd, U)
    obs, pose = load_frame(a.frame, a.prompt)
    g = np.random.default_rng(a.seed).standard_normal(H * AD).astype(np.float32)
    def pinned(c): return (g - (g @ U) @ U.T + c @ U.T).reshape(H, AD).astype(np.float32)
    c_head = np.asarray(joint_head.head_c(policy, [obs]), np.float32).reshape(-1)
    moves = {"forward 1 m": {"forward": 1.0}, "turn left 45 + forward 1 m": {"forward": 1.0, "yaw_deg": 45.0}, "left 1 m": {"left": 1.0}}
    srcs = {"plain noise": (g.reshape(H, AD), 1.5), "head's own command": (pinned(c_head), 0.0)}
    for k, mv in moves.items():
        c, m, _ = mover.command(dict(mv), pose); srcs[k] = (pinned(c), 0.0)
    res = {}
    for name, (z, sig) in srcs.items():
        X = trace(policy, obs, z, sig); res[name] = X
        cproj = X.reshape(len(X), -1) @ U
        print(f"{name:28s} |U^T x_t - U^T x_1| max over t: {np.abs(cproj - cproj[0]).max():.3f}   net xy (m) at t=0: "
              f"{(X[-1][:, :2] * astd[:2] + amean[:2]).sum(0).round(2)}", flush=True)
    np.savez(a.out, names=np.array(list(res)), X=np.stack([res[k] for k in res]), amean=amean, astd=astd, U=U, pose=pose, frame=a.frame)
    print("saved", a.out)

if __name__ == "__main__":
    main()
