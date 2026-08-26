"""Feature-separation-vs-phase probe (Denis, 2026-08-13: analyze WHY posterior calibration fails
before building anything).

Hypothesis: the sampler's mode weights at a conditioning point follow the RATIO of between-task
feature separation to model smoothing. If a checkpoint's pooled head features map left-task and
right-task start frames to overlapping points, the learned p(c|ctx) there can only reproduce the
MARGINAL mode weights — the 50/50 coin flip — regardless of head architecture. The information
exists in the raw observation (different scenes AND different prompts at t=0), and base features
separate starts (bimodality audit), so overlap would mean joint training COLLAPSED it.

Measures, per checkpoint and trajectory-phase bin: leave-one-episode-out linear-probe accuracy for
left-vs-right, and the Fisher ratio (between-class distance over within-class spread) of the
head's own pooled ctx.
  SNMVP_* env must match each checkpoint (set by the runner script).
"""
import argparse
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
BINS = [(0.0, 0.05, "start"), (0.05, 0.3, "early"), (0.3, 0.7, "transit"), (0.85, 1.0, "tail")]


def pooled_ctx(policy, raws):
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
    return np.asarray(ctx.reshape(prefix_out.shape[0], -1), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--n-eps", type=int, default=16)
    a = ap.parse_args()
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.training.config as C
    policy = PC.create_trained_policy(C.get_config(a.config), a.ckpt)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)

    meta = json.load(open(f"{RD}/data_gate_synth/meta.json"))
    rng = np.random.default_rng(0)
    feats, phase, task, epid = [], [], [], []
    for tk, tid in (("left", 2), ("right", 3)):
        keys = [k for k in sorted(meta) if meta[k]["task"] == tid]
        for k in rng.choice(keys, a.n_eps, replace=False):
            d = np.load(f"{RD}/data_gate_synth/{k}.npz")
            T = len(d["state"])
            for lo, hi, _ in BINS:
                for t in rng.choice(range(max(1, int(lo * T)), max(2, int(hi * T))),
                                    2, replace=False):
                    raw = {"observation/image": r224(d["image"][t]),
                           "observation/wrist_image": r224(d["wrist"][t]),
                           "observation/state": d["state"][t].astype(np.float32),
                           "prompt": meta[k]["lang"]}
                    feats.append(raw); phase.append(t / (T - 1)); task.append(tid); epid.append(k)
    X = []
    for i in range(0, len(feats), 8):
        X.append(pooled_ctx(policy, feats[i:i + 8]))
    X = np.concatenate(X, 0)
    phase, task, epid = np.array(phase), np.array(task), np.array(epid)

    print(f"== {a.label}: task separability of the head's own pooled features, by phase ==")
    for lo, hi, name in BINS:
        m = (phase >= lo) & (phase < hi)
        Xm, ym, em = X[m], (task[m] == 2).astype(int), epid[m]
        mu0, mu1 = Xm[ym == 0].mean(0), Xm[ym == 1].mean(0)
        sw = 0.5 * (Xm[ym == 0].std(0).mean() + Xm[ym == 1].std(0).mean())
        fisher = np.linalg.norm(mu1 - mu0) / (sw * np.sqrt(Xm.shape[1]) + 1e-9)
        # leave-one-episode-out nearest-centroid probe
        correct = 0
        for e in set(em.tolist()):
            tr = em != e
            c0, c1 = Xm[tr & (ym == 0)].mean(0), Xm[tr & (ym == 1)].mean(0)
            te = em == e
            pred = (np.linalg.norm(Xm[te] - c1, axis=1) < np.linalg.norm(Xm[te] - c0, axis=1)).astype(int)
            correct += (pred == ym[te]).sum()
        acc = correct / m.sum()
        print(f"  {name:8s} n={m.sum():3d}  probe-acc={acc:.2f}  fisher={fisher:.3f}")
    print("SEP_PROBE_DONE")


if __name__ == "__main__":
    main()
