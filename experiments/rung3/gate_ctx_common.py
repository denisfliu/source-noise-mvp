"""Shared definitions for the gate contextualized-VLM-c pipeline.

Single source of truth for: dataset load + episode L/R labeling, the frozen
80/20 episode split (rng seed 0 — every consumer MUST use this), chunk-target
construction (segY), the contextualized (post-fusion) prefix feature, and the
standardized ridge phi->c map. Used by extract_ctx_features.py,
ctx_steer_diag.py, serve_gate_pin_vlmc.py.

Feature definition (the 2026-08-03 fix for the pre-fusion language wash-out):
phi = masked mean-pool over the PaliGemma LLM prefix OUTPUT (post-fusion,
language has attended over the image), NOT over embed_prefix's pre-fusion
concat. See RESEARCH_LOG 2026-08-03/04 and FINDINGS_INDEX "Instruction
encoding".
"""
import os, glob
import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(RD, "data_gate_synth")
HFB = os.path.expanduser("~/hf_bundle/gate-drone-pi0")
H, AD, K = 50, 32, 5
STRIDE, BS = 8, 8
GATE = np.array([0.86, 0.69, 1.5])
NRM0 = np.array([0.7488, 0.6628, 0.0]); NRM0 /= np.linalg.norm(NRM0)
APER = 0.45
PROMPT_L = "go through the gate on the left and hover over the stuffed animal"
PROMPT_R = "go through the gate on the right and hover over the stuffed animal"
PROMPT_CFL = "go through the center gate from the left and hover over the stuffed animal"
PROMPT_CFR = "go through the center gate from the right and hover over the stuffed animal"
ALL_PROMPTS = (PROMPT_L, PROMPT_R, PROMPT_CFL, PROMPT_CFR)
# c clamp = training-set c range (computed once over the full synth set)
CLO = np.array([-19.379, -13.518, -11.837, -10.703, -13.748], np.float32)
CHI = np.array([19.099, 20.367, 14.758, 13.735, 14.071], np.float32)


def is_left(st):
    """DEPRECATED FOR LABELING (2026-08-05): this geometric check folds center-from-left
    episodes into 'left' and center-from-right into 'right' — the task-label contamination
    that broke the right gate (RESEARCH_LOG 2026-08-05). Labels come from data_gate_synth
    meta.json via load_eps. Retained only as a geometry helper; never assign prompts with it."""
    P = st[:, :3]; s = (P - GATE) @ NRM0
    cr = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
    for i in cr:
        t = s[i] / (s[i] - s[i + 1] + 1e-9)
        xp = P[i] + t * (P[i + 1] - P[i]); d = xp - GATE
        if np.linalg.norm(d - (d @ NRM0) * NRM0) <= APER:
            return True
    return False


def load_norm():
    import openpi.shared.normalize as _nz
    ns = _nz.load(f"{HFB}/assets/gate_nav")
    return ns, np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)


def load_eps(with_images=True):
    """Labels come from the AUTHORITATIVE task map (data_gate_synth/meta.json), never
    from geometry — see is_left's deprecation note. Fails loudly on unknown prompts."""
    import json
    meta = json.load(open(os.path.join(DD, "meta.json")))
    eps = []
    for f in sorted(glob.glob(f"{DD}/ep_*.npz")):
        name = os.path.basename(f)[:-4]
        lang = meta[name]["lang"]
        if lang not in ALL_PROMPTS:
            raise ValueError(f"{name}: unknown task prompt in meta.json: {lang!r}")
        d = np.load(f, allow_pickle=True)
        ep = {"state": d["state"].astype(np.float32),
              "action": d["action"].astype(np.float32), "lang": lang}
        if with_images:
            ep["image"] = d["image"]; ep["wrist"] = d["wrist"]
        eps.append(ep)
    return eps


def make_recs(eps, amean, astd):
    """Frame records with the FROZEN episode split. Order defines shard/X row order."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(eps)); ntr = int(0.8 * len(eps))
    trep = set(idx[:ntr].tolist())
    recs = []
    for ei, ep in enumerate(eps):
        for t in range(0, len(ep["action"]), STRIDE):
            recs.append({"ei": ei, "t": t, "sp": "tr" if ei in trep else "te",
                         "Y": segY(ep["action"][t:], amean, astd)})
    return recs


def segY(seg, amean, astd):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def pad_norm_stats(ns, dim):
    from openpi.transforms import NormStats
    out = {}
    for k, s in ns.items():
        n = np.asarray(s.mean).shape[-1]
        if n >= dim:
            out[k] = s; continue
        p = dim - n
        ext = lambda a, f: None if a is None else np.concatenate(
            [np.asarray(a, np.float32), np.full(p, f, np.float32)])
        out[k] = NormStats(mean=ext(s.mean, 0.), std=ext(s.std, 1.),
                           q01=ext(s.q01, 0.), q99=ext(s.q99, 1.))
    return out


def make_policy(ckpt=None, config="pi0_gate"):
    import openpi.training.config as _cfg
    import openpi.policies.policy_config as _pc
    cfg = _cfg.get_config(config)
    ns, _, _ = load_norm()
    return _pc.create_trained_policy(
        cfg, ckpt or f"{HFB}/checkpoints/gate_both_pin",
        norm_stats=pad_norm_stats(ns, cfg.model.action_dim))


def r224(im):
    from PIL import Image
    return np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC)).astype(np.uint8)


def mkobs(ep, t, lang=None):
    return {"observation/image": r224(ep["image"][t]),
            "observation/wrist_image": r224(ep["wrist"][t]),
            "observation/state": ep["state"][t],
            "prompt": lang if lang is not None else ep["lang"]}


def ctx_pool(policy, raws):
    """Contextualized (post-fusion) prefix feature, masked mean-pool. (B, 2048)."""
    import jax, jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    tds = [policy._input_transform(dict(r)) for r in raws]
    b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
    o = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
    tok, mask, ar = policy._model.embed_prefix(o)
    attn = make_attn_mask(mask, ar); pos = jnp.cumsum(mask, axis=1) - 1
    (pout, _), _ = policy._model.PaliGemma.llm([tok, None], mask=attn, positions=pos)
    m = mask[..., None].astype(jnp.float32); pf = pout.astype(jnp.float32)
    return np.asarray((pf * m).sum(1) / jnp.clip(m.sum(1), 1e-6)).astype(np.float32)


def lang_pool(policy, raws):
    """Contextualized feature pooled over LANGUAGE token positions only (post-fusion).
    The whole-sequence mean-pool dilutes ~10 language tokens against ~500 image
    tokens — fine for geometry, fatal for task semantics (gate-b paraphrase FAIL,
    2026-08-05). Language tokens are the FINAL n_txt prefix positions (pi0
    embed_prefix layout), masked by tokenized_prompt_mask."""
    import jax, jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    tds = [policy._input_transform(dict(r)) for r in raws]
    b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
    o = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
    tok, mask, ar = policy._model.embed_prefix(o)
    attn = make_attn_mask(mask, ar); pos = jnp.cumsum(mask, axis=1) - 1
    (pout, _), _ = policy._model.PaliGemma.llm([tok, None], mask=attn, positions=pos)
    n_txt = o.tokenized_prompt.shape[1]
    tm = o.tokenized_prompt_mask[..., None].astype(jnp.float32)
    pf = pout[:, -n_txt:, :].astype(jnp.float32)
    return np.asarray((pf * tm).sum(1) / jnp.clip(tm.sum(1), 1e-6)).astype(np.float32)


def prefusion_pool(policy, raws):
    """PRE-fusion prefix feature (embed_prefix mean-pool) — the legacy feature the
    first VLM-c maps were built on; kept for faithful A/B serving and probes."""
    import jax, jax.numpy as jnp
    from openpi.models import model as _model
    tds = [policy._input_transform(dict(r)) for r in raws]
    b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
    o = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
    tok, mask, _ = policy._model.embed_prefix(o)
    m = mask[..., None].astype(jnp.float32)
    return np.asarray((tok.astype(jnp.float32) * m).sum(1) / jnp.clip(m.sum(1), 1e-6)).astype(np.float32)


def feats(policy, obs_list, log_every=None):
    Xs = []
    for i in range(0, len(obs_list), BS):
        Xs.append(ctx_pool(policy, obs_list[i:i + BS]))
        if log_every and i % (BS * log_every) == 0:
            print("  ctx feat %d/%d" % (i, len(obs_list)), flush=True)
    return np.concatenate(Xs, 0)


def load_ridge(path):
    """Loads either schema: ridge (mu,sg,W,c0,...) or GELU MLP (mu,sg,W1..b3,...)."""
    r = np.load(path)
    keys = ("mu", "sg", "W", "c0") if "W" in r.files else ("mu", "sg", "W1", "b1", "W2", "b2", "W3", "b3")
    return {k: r[k].astype(np.float32) for k in keys + ("clo", "chi")}


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def apply_ridge(m, X, clamp=False):
    x = (X - m["mu"]) / m["sg"]
    if "W" in m:
        c = x @ m["W"] + m["c0"]
    else:
        c = _gelu(_gelu(x @ m["W1"] + m["b1"]) @ m["W2"] + m["b2"]) @ m["W3"] + m["b3"]
    return np.clip(c, m["clo"], m["chi"]) if clamp else c


def fit_ridge(X, C, tr, lam):
    """Standardized ridge fit on the train mask; returns map dict (unclamped stats)."""
    mu = X[tr].mean(0); sg = X[tr].std(0) + 1e-6
    Xs = (X - mu) / sg; c0 = C[tr].mean(0); Cc = C - c0
    A = Xs[tr].T @ Xs[tr]; B = Xs[tr].T @ Cc[tr]
    W = np.linalg.solve(A + lam * np.eye(X.shape[1], dtype=np.float32), B).astype(np.float32)
    return {"mu": mu.astype(np.float32), "sg": sg.astype(np.float32), "W": W,
            "c0": c0.astype(np.float32), "clo": CLO, "chi": CHI}
