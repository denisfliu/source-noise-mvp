"""Does the MDN head KNOW when it is guessing? sigma(o) vs actual command error, by phase.

Motivation (2026-08-20): the basis axis is characterized — expressive bases execute the tail
command faithfully, including faithfully-wrong (gmmmh right/center thrash). The binding problem
is tail COMMAND CONTENT. If the trained sigma(o) is honestly wide exactly where mu* is wrong,
the head carries its own uncertainty signal, and a sigma-gated serve (relax the pin toward plain
denoising when sigma blows up) becomes a principled mechanism rather than a regime patch.

For an MDN checkpoint: forward demo frames, take the argmax-pi component's (mu*, sigma*), compare
mu* against the oracle c = U^T a of the true future chunk. Report per task x phase:
  |err|  mean ||mu* - c_oracle||  (c-space)
  sig    mean ||sigma*||          (c-space)
  corr   Spearman rank corr(||sigma*||, ||err||) within the cell rows pooled per task, and
         pooled overall / tail-only. Mirrors the training/serve mixture forward exactly
         (joint_head.head_c GMM branch); any drift here is a bug, compare against it.

  SNMVP_HEAD_GMM=1 ... python sigma_phase_probe.py --ckpt <ck> --pin-u <U>
"""
import argparse
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import joint_head

H = 50
TASK_EPS = joint_head.TASK_EPS
PROMPTS = joint_head.PROMPTS
BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def gmm_params(policy, raws):
    """(pi, mu, logsig) for raw obs dicts — the head_c GMM branch, with sigma exposed."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    m = policy._model
    tds = [policy._input_transform(dict(r)) for r in raws]
    b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
    o = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
    tok, mask, ar = m.embed_prefix(o)
    (prefix_out, _), _ = m.PaliGemma.llm([tok, None], mask=make_attn_mask(mask, ar),
                                         positions=jnp.cumsum(mask, axis=1) - 1)
    pm = mask.astype(jnp.float32)
    keys, vals = m.snmvp_k(prefix_out), m.snmvp_v(prefix_out)
    sc = jnp.einsum("qd,btd->bqt", m.snmvp_q.value, keys) / jnp.sqrt(256.0)
    sc = jnp.where(pm[:, None, :] > 0, sc, -1e9)
    ctx = jnp.einsum("bqt,btd->bqd", jax.nn.softmax(sc, axis=-1), vals)
    ctx = ctx.reshape(prefix_out.shape[0], -1)
    nl = o.tokenized_prompt.shape[1]
    lm = o.tokenized_prompt_mask.astype(jnp.float32)[:, :, None]
    lang = (prefix_out[:, -nl:] * lm).sum(1) / jnp.clip(lm.sum(1), 1e-6)
    state = jnp.stack([jnp.asarray(t["state"]) for t in tds], 0)
    import jax.nn as jnn
    cond = jnp.concatenate([jnn.silu(m.snmvp_state_proj(state)),
                            jnn.silu(m.snmvp_lang_proj(lang)),
                            jnn.silu(m.snmvp_gen_ctx(ctx))], axis=-1)
    h = jnn.silu(m.snmvp_gmm_h2(jnn.silu(m.snmvp_gmm_h1(cond))))
    out = np.asarray(m.snmvp_gmm_out(h), np.float32)
    K = m.snmvp_head_out.kernel.value.shape[1]
    M = out.shape[1] // (1 + 2 * K)
    logit = out[:, :M]
    mu = out[:, M:M * (1 + K)].reshape(-1, M, K)
    logsig = np.clip(out[:, M * (1 + K):].reshape(-1, M, K), -5.0, 2.0)
    w = np.exp(logit - logit.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    return w, mu, logsig


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--eps-per-task", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=12)
    ap.add_argument("--save", default="", help="npz to save raw (frac,is_stop,err,sig) rows per task")
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.training.config as C
    import openpi.shared.normalize as _nz
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt,
                                      norm_stats=_nz.load(a.norm))
    U = np.load(a.pin_u).astype(np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(0)

    rows = {}  # task -> list of (frac, is_stop, |err|, |sig|)
    for task, eps in TASK_EPS.items():
        rows[task] = []
        for e in rng.choice(list(eps), a.eps_per_task, replace=False):
            d = np.load(f"{RD}/{joint_head.DATA_DIR}/ep_{int(e):04d}.npz", allow_pickle=True)
            st = d["state"].astype(np.float32)
            T = len(st)
            raws, metas = [], []
            for t in range(0, T - 2, a.frame_stride):
                raws.append({"observation/image": r224(d["image"][t]),
                             "observation/wrist_image": r224(d["wrist"][t]),
                             "observation/state": st[t], "prompt": PROMPTS[task]})
                metas.append((t / T, t > T - H, int(e), t))
            for i in range(0, len(raws), 16):
                w, mu, logsig = gmm_params(policy, raws[i:i + 16])
                for k in range(len(w)):
                    j = int(w[k].argmax())
                    frac, is_stop, ee, tt = metas[i + k]
                    c_or = joint_head._oracle_c(U, a.norm, ee, tt)
                    err = float(np.linalg.norm(mu[k, j] - c_or))
                    sig = float(np.linalg.norm(np.exp(logsig[k, j])))
                    rows[task].append((frac, is_stop, err, sig))
        print(f"[{task}] {len(rows[task])} rows", flush=True)

    print(f"\n{'task':18s} " + "".join(f"{f'[{lo:.2f},{hi:.2f})':>16s}" for lo, hi in BINS)
          + f"{'stop':>16s}{'corr(sig,err)':>14s}")
    all_err, all_sig, tail_err, tail_sig = [], [], [], []
    for task, rr in rows.items():
        rr = np.array(rr, np.float64)
        cells = []
        for lo, hi in BINS:
            m = (rr[:, 0] >= lo) & (rr[:, 0] < hi)
            cells.append(f"{rr[m, 2].mean():5.2f}/{rr[m, 3].mean():4.2f}" if m.sum() else "  -  ")
        m = rr[:, 1] > 0
        stopc = f"{rr[m, 2].mean():5.2f}/{rr[m, 3].mean():4.2f}" if m.sum() else "  -  "
        cor = spearman(rr[:, 3], rr[:, 2])
        print(f"{task:18s} " + "".join(f"{c:>16s}" for c in cells) + f"{stopc:>16s}{cor:>14.3f}")
        all_err += rr[:, 2].tolist(); all_sig += rr[:, 3].tolist()
        tm = rr[:, 0] > 0.7
        tail_err += rr[tm, 2].tolist(); tail_sig += rr[tm, 3].tolist()
    if a.save:
        np.savez_compressed(a.save, **{t: np.array(r, np.float64) for t, r in rows.items()})
        print(f"saved rows -> {a.save}")
    print(f"\ncells are  mean|err| / mean|sigma|  (c-space L2)")
    print(f"pooled corr(sig, err) = {spearman(np.array(all_sig), np.array(all_err)):.3f}   "
          f"tail-only (frac>0.7) = {spearman(np.array(tail_sig), np.array(tail_err)):.3f}")


if __name__ == "__main__":
    main()
