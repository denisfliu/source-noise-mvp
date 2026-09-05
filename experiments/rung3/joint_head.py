"""Read the command head out of a jointly-trained flow checkpoint and run it at inference.

The head is an nnx submodule of the flow (`snmvp_q/k/v/head_in/head_out`), so it ships inside the
checkpoint and cannot be paired with the wrong weights — that is the point of joint training. This
replays the exact readout used in `Pi0.compute_loss`: a 4-query attention pool over the post-fusion
prefix tokens, then a 2-layer MLP to c.

Rather than re-implementing the arithmetic, it calls the model's OWN submodules, so serving cannot
drift from training. `prefix_out` is taken from a prefix-only forward, which is identical to the
prefix half of the joint prefix+suffix forward used in training because the attention mask forbids
prefix->suffix attention (the same property `gate_ctx_common.lang_pool` relies on).

IMPORTANT: the head submodules only exist if SNMVP_HEAD=1 and SNMVP_PIN_U are set BEFORE openpi is
imported, since module construction is env-gated. Import this before openpi, or set the env yourself.

  python joint_head.py --ckpt <flow ckpt> --check      # validate plumbing against oracle c
"""
import argparse
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
H = 50
STRIDE = 6
# demo dir for oracle/check rows; gmsig2+ arms train on gate_nav2 -> data_gate_synth2
DATA_DIR = os.environ.get("SNMVP_DATA_DIR", "data_gate_synth")
TASK_EPS = {"center_from_left": range(0, 50), "center_from_right": range(50, 100),
            "left": range(100, 150), "right": range(150, 200)}
PROMPTS = {"center_from_left": "go through the center gate from the left and hover over the stuffed animal",
           "center_from_right": "go through the center gate from the right and hover over the stuffed animal",
           "left": "go through the gate on the left and hover over the stuffed animal",
           "right": "go through the gate on the right and hover over the stuffed animal"}


def enable_head(pin_u_path, with_state=None):
    """Must run before openpi is imported: the head modules are constructed env-gated.

    with_state must match how the checkpoint was TRAINED (it changes the head's input width), so it
    defaults to whatever SNMVP_HEAD_STATE already says rather than silently picking a value."""
    os.environ["SNMVP_HEAD"] = "1"
    os.environ.setdefault("SNMVP_PIN_U", pin_u_path)
    if with_state is not None:
        os.environ["SNMVP_HEAD_STATE"] = "1" if with_state else "0"


_GEN_RNG = __import__("numpy").random.default_rng(0)


_GMM_FWD = {}


def _gmm_forward(m, o, state, lang_zero):
    """Pure-JAX GMM head forward on a preprocessed Observation batch (module method form so it can
    be frozen + jitted with nnx_utils.module_jit; 2026-09-04 hardware latency fix: 2.4 s -> ~0.1 s)."""
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    tok, mask, ar = m.embed_prefix(o)
    (prefix_out, _), _ = m.PaliGemma.llm([tok, None], mask=make_attn_mask(mask, ar),
                                         positions=jnp.cumsum(mask, axis=1) - 1)
    pm = mask.astype(jnp.float32)
    keys, vals = m.snmvp_k(prefix_out), m.snmvp_v(prefix_out)
    sc = jnp.einsum("qd,btd->bqt", m.snmvp_q.value, keys) / jnp.sqrt(256.0)
    sc = jnp.where(pm[:, None, :] > 0, sc, -1e9)
    ctx = jnp.einsum("bqt,btd->bqd", jax.nn.softmax(sc, axis=-1), vals).reshape(prefix_out.shape[0], -1)
    if os.environ.get("SNMVP_HEAD_STATE") == "1":          # must mirror training exactly
        ctx = jnp.concatenate([ctx, state], axis=-1)
    nl = o.tokenized_prompt.shape[1]
    lm = o.tokenized_prompt_mask.astype(jnp.float32)[:, :, None]
    lang = (prefix_out[:, -nl:] * lm).sum(1) / jnp.clip(lm.sum(1), 1e-6)
    st_b = jax.nn.silu(m.snmvp_state_proj(state))
    lg_b = jax.nn.silu(m.snmvp_lang_proj(lang))
    if lang_zero:
        lg_b = jnp.zeros_like(lg_b)
    im_b = jax.nn.silu(m.snmvp_gen_ctx(ctx))
    cond = jnp.concatenate([st_b, lg_b, im_b], axis=-1)
    h = jax.nn.silu(m.snmvp_gmm_h2(jax.nn.silu(m.snmvp_gmm_h1(cond))))
    return m.snmvp_gmm_out(h)


def _gmm_fwd_jitted(m):
    if id(m) not in _GMM_FWD:
        import types
        from openpi.shared import nnx_utils
        bound = types.MethodType(_gmm_forward, m)
        _GMM_FWD[id(m)] = nnx_utils.module_jit(bound, static_argnames=("lang_zero",))
    return _GMM_FWD[id(m)]


def head_c(policy, raws, return_gmm=False):
    """Commanded c for a batch of raw observation dicts, from the checkpoint's own head.

    return_gmm=True (MDN checkpoints only) additionally returns (pi, mu) so the caller can do
    component selection itself (the server's pi-hysteresis latch) and log pi per replan."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    m = policy._model
    for attr in ("snmvp_q", "snmvp_k", "snmvp_v", "snmvp_head_in", "snmvp_head_out"):
        if not hasattr(m, attr):
            raise SystemExit(f"[joint_head] this checkpoint has no {attr}: it was not trained with "
                             f"SNMVP_HEAD=1, or SNMVP_HEAD/SNMVP_PIN_U were unset before openpi import")
    tds = [policy._input_transform(dict(r)) for r in raws]
    b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
    o = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
    if hasattr(m, "snmvp_gmm_out"):
        # MDN head served through the compiled forward (see _gmm_forward); numerics identical to the
        # eager path below, which is kept for the non-GMM heads.
        state = jnp.stack([jnp.asarray(t["state"]) for t in tds], 0)
        fwd = _gmm_fwd_jitted(m)
        out = np.asarray(fwd(o, state, lang_zero=False), np.float32)
        K = m.snmvp_head_out.kernel.value.shape[1]
        M = out.shape[1] // (1 + 2 * K)
        logit = out[:, :M]
        mu = out[:, M:M * (1 + K)].reshape(-1, M, K)
        _w = float(os.environ.get("SNMVP_GMM_LANG_CFG", "1"))
        if _w != 1.0:
            out0 = np.asarray(fwd(o, state, lang_zero=True), np.float32)
            mu0 = out0[:, M:M * (1 + K)].reshape(-1, M, K)
            mu = mu + (_w - 1.0) * (mu - mu0)
        sig = np.exp(np.clip(out[:, M * (1 + K):].reshape(-1, M, K), -5.0, 2.0))
        w = np.exp(logit - logit.max(-1, keepdims=True))
        w = w / w.sum(-1, keepdims=True)
        if os.environ.get("SNMVP_GMM_MODE", "argmax") == "mean":
            c = (w[..., None] * mu).sum(1)
        else:
            c = mu[np.arange(len(mu)), w.argmax(-1)]
        return (c, w, mu, sig) if return_gmm else c
    tok, mask, ar = m.embed_prefix(o)
    # llm returns ((prefix_out, suffix_out), kv_cache); with no suffix the second output is None,
    # so the outer tuple must be unpacked too (same form gate_ctx_common.lang_pool uses)
    (prefix_out, _), _ = m.PaliGemma.llm([tok, None], mask=make_attn_mask(mask, ar),
                                         positions=jnp.cumsum(mask, axis=1) - 1)
    pm = mask.astype(jnp.float32)
    keys, vals = m.snmvp_k(prefix_out), m.snmvp_v(prefix_out)
    sc = jnp.einsum("qd,btd->bqt", m.snmvp_q.value, keys) / jnp.sqrt(256.0)
    sc = jnp.where(pm[:, None, :] > 0, sc, -1e9)
    ctx = jnp.einsum("bqt,btd->bqd", jax.nn.softmax(sc, axis=-1), vals)
    ctx = ctx.reshape(prefix_out.shape[0], -1)
    if os.environ.get("SNMVP_HEAD_STATE") == "1":          # must mirror training exactly
        ctx = jnp.concatenate([ctx, jnp.stack([jnp.asarray(t["state"]) for t in tds], 0)], axis=-1)
    if hasattr(m, "snmvp_gmm_out"):
        # MDN head (toy_cmdhead 2026-08-19): explicit (pi, mu, sigma) from the FiLM information
        # diet. Default serve = argmax-mode MEAN — commits to one component, zero sampling jitter.
        # SNMVP_GMM_MODE=mean gives the mixture mean Sum pi_k mu_k instead (the readout gate uses
        # it: tests the learned distribution's CENTER, the analogue of the CFM 8-sample mean —
        # NOT for serving, it is exactly the mode-average the head exists to avoid).
        nl = o.tokenized_prompt.shape[1]
        lm = o.tokenized_prompt_mask.astype(jnp.float32)[:, :, None]
        lang = (prefix_out[:, -nl:] * lm).sum(1) / jnp.clip(lm.sum(1), 1e-6)
        state = jnp.stack([jnp.asarray(t["state"]) for t in tds], 0)
        st_b = jax.nn.silu(m.snmvp_state_proj(state))
        lg_b = jax.nn.silu(m.snmvp_lang_proj(lang))
        im_b = jax.nn.silu(m.snmvp_gen_ctx(ctx))
        def _mdn_out(lb):
            cond = jnp.concatenate([st_b, lb, im_b], axis=-1)
            h = jax.nn.silu(m.snmvp_gmm_h2(jax.nn.silu(m.snmvp_gmm_h1(cond))))
            return np.asarray(m.snmvp_gmm_out(h), np.float32)
        out = _mdn_out(lg_b)
        K = m.snmvp_head_out.kernel.value.shape[1]
        M = out.shape[1] // (1 + 2 * K)
        logit = out[:, :M]
        mu = out[:, M:M * (1 + K)].reshape(-1, M, K)
        # CFG language sharpener (2026-08-25, the sanctioned final-sharpener): amplify the
        # measured-but-small language contrast by extrapolating the component MEANS away from
        # the null-language conditional (trained via the 10% language dropout). pi/sigma,
        # component selection, hysteresis, and the trust dial stay full-conditional.
        _w = float(os.environ.get("SNMVP_GMM_LANG_CFG", "1"))
        if _w != 1.0:
            out0 = _mdn_out(jnp.zeros_like(lg_b))
            mu0 = out0[:, M:M * (1 + K)].reshape(-1, M, K)
            mu = mu + (_w - 1.0) * (mu - mu0)
        sig = np.exp(np.clip(out[:, M * (1 + K):].reshape(-1, M, K), -5.0, 2.0))
        w = np.exp(logit - logit.max(-1, keepdims=True))
        w = w / w.sum(-1, keepdims=True)
        if os.environ.get("SNMVP_GMM_MODE", "argmax") == "mean":
            c = (w[..., None] * mu).sum(1)
        else:
            c = mu[np.arange(len(mu)), w.argmax(-1)]
        return (c, w, mu, sig) if return_gmm else c
    if hasattr(m, "snmvp_gf_out"):
        # FiLM generative head: conditioning (state + language-token mean + image pool) enters
        # only through per-layer scale/shift of the CFM trunk. Mirrors the training path exactly.
        n_steps = int(os.environ.get("SNMVP_GEN_STEPS", "10"))
        n_samp = int(os.environ.get("SNMVP_GEN_SAMPLES", "1"))
        K = m.snmvp_gf_out.kernel.value.shape[1]
        b = ctx.shape[0]
        nl = o.tokenized_prompt.shape[1]
        lm = o.tokenized_prompt_mask.astype(jnp.float32)[:, :, None]
        lang = (prefix_out[:, -nl:] * lm).sum(1) / jnp.clip(lm.sum(1), 1e-6)
        state = jnp.stack([jnp.asarray(t["state"]) for t in tds], 0)
        cond = jnp.concatenate([
            jax.nn.silu(m.snmvp_state_proj(state)),
            jax.nn.silu(m.snmvp_lang_proj(lang)),
            jax.nn.silu(m.snmvp_gen_ctx(ctx))], axis=-1)
        f = jax.nn.silu(m.snmvp_film_h(cond))
        g1, b1 = jnp.split(m.snmvp_film1(f), 2, axis=-1)
        g2, b2 = jnp.split(m.snmvp_film2(f), 2, axis=-1)
        fr = jnp.asarray(2.0 ** np.arange(16), jnp.float32)
        out = []
        for s in range(n_samp):
            c = jnp.asarray(_GEN_RNG.standard_normal((b, K)).astype(np.float32))
            t = 1.0
            for _ in range(n_steps):
                tg = jnp.full((b, 1), t, jnp.float32)
                temb = jnp.concatenate([jnp.sin(tg * fr), jnp.cos(tg * fr)], axis=-1)
                h = jax.nn.silu((1 + g1) * m.snmvp_gf_in(jnp.concatenate([c, temb], axis=-1)) + b1)
                h = jax.nn.silu((1 + g2) * m.snmvp_gf_h2(h) + b2)
                c = c + (-1.0 / n_steps) * m.snmvp_gf_out(h)
                t -= 1.0 / n_steps
            out.append(np.asarray(c, np.float32))
        return np.mean(out, 0)
    if hasattr(m, "snmvp_gen_out"):
        # generative head: SAMPLE c ~ p(c|o) by Euler over the trained K-dim CFM. Independent
        # draw per call — no commitment mechanism, the state is the memory (Denis, 2026-08-13).
        # gen_samples>1 returns the SAMPLE MEAN (diagnostics only — the mean is exactly the
        # invalid average the head exists to avoid; serving uses one draw).
        n_steps = int(os.environ.get("SNMVP_GEN_STEPS", "10"))
        n_samp = int(os.environ.get("SNMVP_GEN_SAMPLES", "1"))
        K = m.snmvp_gen_out.kernel.value.shape[1]
        b = ctx.shape[0]
        cp = jax.nn.silu(m.snmvp_gen_ctx(ctx))
        fr = jnp.asarray(2.0 ** np.arange(16), jnp.float32)
        out = []
        for s in range(n_samp):
            c = jnp.asarray(_GEN_RNG.standard_normal((b, K)).astype(np.float32))
            dt = -1.0 / n_steps
            t = 1.0
            for _ in range(n_steps):
                tg = jnp.full((b, 1), t, jnp.float32)
                temb = jnp.concatenate([jnp.sin(tg * fr), jnp.cos(tg * fr)], axis=-1)
                h = jax.nn.silu(m.snmvp_gen_h1(jnp.concatenate([c, cp, temb], axis=-1)))
                v = m.snmvp_gen_out(jax.nn.silu(m.snmvp_gen_h2(h)))
                c = c + dt * v
                t += dt
            out.append(np.asarray(c, np.float32))
        return np.mean(out, 0)
    return np.asarray(m.snmvp_head_out(jax.nn.silu(m.snmvp_head_in(ctx))), np.float32)


def _oracle_c(U, norm_dir, ep, t):
    """c = U^T a from the ground-truth chunk — the target the head was trained against."""
    import openpi.training.config as C
    import openpi.transforms as T
    from openpi.shared.normalize import NormStats, load as load_ns
    AD = C.get_config("pi0_gate").model.action_dim

    def pads(d, dim):
        o = {}
        for k, s in d.items():
            n = len(s.mean)
            if n >= dim:
                o[k] = s
                continue
            p = dim - n
            ext = lambda a, f: None if a is None else np.concatenate(
                [np.asarray(a, np.float32), np.full(p, f, np.float32)])
            o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
        return o

    nrm = T.Normalize(pads(load_ns(norm_dir), AD), use_quantiles=False)
    d = np.load(f"{RD}/{DATA_DIR}/ep_{ep:04d}.npz", allow_pickle=True)
    ac = d["action"].astype(np.float32)
    ch = np.zeros((H, AD), np.float32)
    k = min(H, len(ac) - t)
    ch[:k, :7] = ac[t:t + k]
    return (nrm({"actions": ch})["actions"].reshape(-1)) @ U


def check(a):
    """Gate before flying: does the served head reproduce the oracle c it was trained on?

    If the plumbing is wrong (wrong readout, wrong mask, wrong param) this collapses, so a high
    per-task R2 is evidence the inference path matches the training path. Reported PER TASK, because
    pooled R2 is inflated by between-task variance."""
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.training.config as C
    # generative head: gate on the 8-sample mean (tests the learned distribution's center and the
    # plumbing; serving itself uses single draws). MDN analogue: gate on the mixture mean.
    os.environ.setdefault("SNMVP_GEN_SAMPLES", "8")
    os.environ.setdefault("SNMVP_GMM_MODE", "mean")
    # norm stats passed EXPLICITLY from --norm: checkpoints trained on gate_nav2/3 store their
    # assets under their own repo_id, so the config-keyed lookup 404s (bit gmsig2, 2026-08-23)
    import openpi.shared.normalize as _nz
    policy = PC.create_trained_policy(C.get_config(a.config), a.ckpt,
                                      norm_stats=_nz.load(a.norm))
    U = np.load(a.pin_u).astype(np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(0)
    print(f"{'task':18s} {'n':>4s} {'per-task c-R2':>14s} {'mean |c| err':>13s}")
    for task, eps in TASK_EPS.items():
        P, T_ = [], []
        for e in rng.choice(list(eps), a.n, replace=False):
            d = np.load(f"{RD}/{DATA_DIR}/ep_{int(e):04d}.npz", allow_pickle=True)
            st = d["state"].astype(np.float32)
            t = int(rng.integers(0, max(1, len(st) - H - 1)))
            raw = {"observation/image": r224(d["image"][t]),
                   "observation/wrist_image": r224(d["wrist"][t]),
                   "observation/state": st[t], "prompt": PROMPTS[task]}
            P.append(head_c(policy, [raw])[0])
            T_.append(_oracle_c(U, a.norm, int(e), t))
        P, T_ = np.array(P), np.array(T_)
        r2 = 1 - ((T_ - P) ** 2).sum() / (((T_ - T_.mean(0)) ** 2).sum() + 1e-9)
        print(f"{task:18s} {len(P):4d} {r2:+14.4f} {np.abs(P - T_).mean():13.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_gate_rrr_k5.npy")
    ap.add_argument("--norm", default="/home/dfliu/hf_bundle/gate-drone-pi0/assets/gate_nav")
    ap.add_argument("--n", type=int, default=8, help="episodes sampled per task")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--head-state", action="store_true",
                    help="set if the checkpoint was trained with SNMVP_HEAD_STATE=1")
    a = ap.parse_args()
    enable_head(a.pin_u, with_state=a.head_state or None)
    if a.check:
        check(a)
    else:
        ap.error("nothing to do; pass --check")


if __name__ == "__main__":
    main()
