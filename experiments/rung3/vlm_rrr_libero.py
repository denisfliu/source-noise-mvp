"""VLM-grounded pin, generalization test on multi-task LIBERO (goal 10-19 language-driven, object 20-29
state-driven). Idea: keep c a real action coordinate c=U^T a, but DEFINE U as the VLM-predictable action
subspace (RRR with pi0's own VLM representation as the predictor), and use VLM->c as the prior. This
replaces hand-crafted instruction encodings (one-hot / direction:target: slots) with the VLM's grounding,
so ONE U + ONE prior should carry the instruction across BOTH suites. We compare, per suite, held-out c-
prediction R^2 and subspace coverage for: VLM-RRR (this) vs STATE-RRR + state+lang prior (current recipe).
Feature backend is pluggable (--feat prefix|context); prefix = pre-fusion embed_prefix mean-pool (cheap
baseline), context = fused contextualized prefix (set by the extractor once wired)."""
import argparse
import json
import os

import numpy as np
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32


def resize224(img):
    return np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC)).astype(np.uint8)


def make_obs(ep, t):
    img = resize224(ep["image"][t]); wrist = resize224(ep["wrist"][t]) if "wrist" in ep else img.copy()
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": ep["state"][t], "prompt": ep["lang"]}


def seg_to_Y(seg, amean, astd):  # flattened normalized action chunk (1600) -- RRR target, no projection
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def rrr_U(X, Y, K):
    """U = top-K eigenvectors of Cov(Yhat), Yhat = X @ OLS(X->Y); orthonormal in action-chunk space."""
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Yhat = Xb @ W
    Yc = Yhat - Yhat.mean(0)
    C = (Yc.T @ Yc) / len(Yc)
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :K].astype(np.float32)  # [1600, K], orthonormal


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def coverage(C, Ctot_var):
    return float(C.var(0).sum() / (Ctot_var + 1e-9))


def fit_mlp(Xtr, Ytr, Xte, steps=4000):
    import torch, torch.nn as nn
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.SiLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(((Xtr - m) / s).astype(np.float32)), torch.tensor(Ytr.astype(np.float32))
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,))
        loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        import torch as T
        return net(T.tensor(((Xte - m) / s).astype(np.float32))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/code/openpi/checkpoints/pi0_libero_shared/snmvp_src_pin_rrr/4999"))
    ap.add_argument("--config", default="pi0_libero_shared")
    ap.add_argument("--norm", default=os.path.join(RD, "norm_shared_libero"))
    ap.add_argument("--raw", default=os.path.join(RD, "data_libero_multi"))
    ap.add_argument("--feat", default="prefix", choices=["prefix", "context"])
    ap.add_argument("--K", type=int, default=5); ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--heldout_tasks", default="", help="comma task ids held out ENTIRELY (task-generalization test)")
    ap.add_argument("--save_U", default="", help="build VLM-RRR U on train frames and save to this path")
    args = ap.parse_args()
    HELD = set(int(x) for x in args.heldout_tasks.split(",") if x != "")
    cache = os.path.join(RD, f"vlm_feat_{args.feat}.npz")

    import openpi.shared.normalize as _normalize
    ns = _normalize.load(args.norm)
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
    smean, sstd = np.asarray(ns["state"].mean), np.asarray(ns["state"].std)

    meta = json.load(open(os.path.join(args.raw, "meta.json")))
    keys = sorted(meta); tasks = sorted({meta[k]["task"] for k in keys}); tid = {t: i for i, t in enumerate(tasks)}
    eps = []
    for k in keys:
        d = np.load(os.path.join(args.raw, k + ".npz"))
        eps.append({"image": d["image"], "wrist": d["wrist"], "state": d["state"].astype(np.float32),
                    "action": d["action"].astype(np.float32), "task": meta[k]["task"], "lang": meta[k]["lang"]})
    rng = np.random.default_rng(0); idx = rng.permutation(len(eps)); ntr = int(0.7 * len(eps))
    trep = set(idx[:ntr].tolist())

    recs = []
    for ei, ep in enumerate(eps):
        T = len(ep["action"])
        for t in range(0, T, args.stride):
            recs.append({"ei": ei, "t": t, "task": ep["task"], "sp": "tr" if ei in trep else "te",
                         "suite": "goal" if ep["task"] < 20 else "object",
                         "Y": seg_to_Y(ep["action"][t:], amean, astd),
                         "st": np.asarray(0)})  # state filled after policy load
    print(f"frames={len(recs)} tasks={len(tasks)} feat={args.feat}", flush=True)

    # ---- features ----
    if os.path.exists(cache):
        z = np.load(cache); X = z["X"]; ST = z["ST"]
        print(f"loaded cached {args.feat} features {X.shape}", flush=True)
    else:
        import jax, jax.numpy as jnp
        import openpi.training.config as _config
        import openpi.policies.policy_config as _policy_config
        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask
        policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.ckpt, norm_stats=ns)

        def _obs(raws):
            tds = [policy._input_transform(dict(r)) for r in raws]
            batched = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
            return _model.preprocess_observation(None, _model.Observation.from_dict(batched), train=False)

        def prefix_pool(raws):  # pre-fusion embed_prefix mean-pool (cheap baseline)
            obs = _obs(raws)
            tokens, mask, _ = policy._model.embed_prefix(obs)
            m = mask[..., None].astype(jnp.float32); tk = tokens.astype(jnp.float32)
            return np.asarray((tk * m).sum(1) / jnp.clip(m.sum(1), 1e-6)), np.asarray(obs.state)

        def prefix_context(raws):  # fused: post-transformer contextualized prefix (image+language attended)
            obs = _obs(raws)
            tokens, mask, ar_mask = policy._model.embed_prefix(obs)
            attn_mask = make_attn_mask(mask, ar_mask); positions = jnp.cumsum(mask, axis=1) - 1
            outputs, _ = policy._model.PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions)
            po = outputs[0].astype(jnp.float32); m = mask[..., None].astype(jnp.float32)
            return np.asarray((po * m).sum(1) / jnp.clip(m.sum(1), 1e-6)), np.asarray(obs.state)

        extractor = prefix_context if args.feat == "context" else prefix_pool
        Xs, STs = [], []
        for i in range(0, len(recs), args.bs):
            raws = [make_obs(eps[r["ei"]], r["t"]) for r in recs[i:i + args.bs]]
            fx, st = extractor(raws); Xs.append(fx); STs.append(st)
            if i % (args.bs * 20) == 0:
                print(f"  feat {i}/{len(recs)}", flush=True)
        X = np.concatenate(Xs, 0).astype(np.float32); ST = np.concatenate(STs, 0).astype(np.float32)
        np.savez_compressed(cache, X=X, ST=ST)
        print(f"extracted {X.shape} -> cached", flush=True)

    Y = np.stack([r["Y"] for r in recs]).astype(np.float32)
    sp = np.array([r["sp"] for r in recs]); suite = np.array([r["suite"] for r in recs])
    rtask = np.array([r["task"] for r in recs])
    STn = ST.reshape(len(ST), -1)  # model state (already normalized by policy input transform)
    if HELD:  # task-generalization: held-out TASKS never seen in training
        te = np.isin(rtask, list(HELD)); tr = ~te
        print(f"TASK-HELDOUT mode: held-out tasks {sorted(HELD)}  train frames={tr.sum()} test={te.sum()}", flush=True)
    else:
        tr, te = sp == "tr", sp == "te"
    suite_te = suite[te]

    if args.save_U:
        U_all = rrr_U(X[tr], Y[tr], args.K)
        np.save(args.save_U, U_all)
        print(f"SAVED VLM-RRR U ({args.feat}) {U_all.shape} -> {args.save_U}", flush=True)

    def eval_recipe(feat_tr, feat_all, name):
        U = rrr_U(feat_tr, Y[tr], args.K)
        C = Y @ U
        Ctot = C[tr].var(0).sum()
        pred = fit_mlp(feat_all[tr], C[tr], feat_all[te])  # rows align with C[te]
        Cte = C[te]
        print(f"\n== {name}: U=RRR K={args.K}, prior=MLP(feat) ==", flush=True)
        for su in ["goal", "object"]:
            m = suite_te == su
            print(f"  [{su:6s}] held c-R^2 = {r2(pred[m], Cte[m]):+.3f}   coverage(U) = {coverage(Cte[m], Ctot):.3f}   n={m.sum()}")
        print(f"  [all   ] held c-R^2 = {r2(pred, Cte):+.3f}", flush=True)
        return U

    # VLM recipe
    eval_recipe(X[tr], X, "VLM (feat=%s)" % args.feat)
    # STATE reference: U=state-RRR, prior=state+lang onehot
    oh = np.zeros((len(recs), len(tasks)), np.float32)
    for i, r in enumerate(recs):
        oh[i, tid[r["task"]]] = 1
    Xstate = np.concatenate([STn, oh], 1).astype(np.float32)
    eval_recipe(STn[tr], Xstate, "STATE+lang reference")
    print("VLM_RRR_DONE", flush=True)


if __name__ == "__main__":
    main()
