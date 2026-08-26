"""Per-domain pin bases (queue #3, Denis 2026-08-07): fit the RRR basis U separately on
real and synthetic gate demos (+ pooled), same recipe for all, and report principal
angles vs the deployed synth-only basis. Ridge (lam=10) instead of OLS for conditioning
on the smaller real set — matches the 2026-08-06 rrr_domain measurement recipe.

Labels are authoritative (synth: episode-index task map; real: meta.json lang) — the old
vlm_feat_gate_prefix.npz cache carries the geometric is_left contamination and is NOT
reused. Only the two gate prompts enter the fit (synth eps 100-199, real all 100).

Outputs: pin_U_gate_rrr_{real,pooled,synthLR}_k5.npy + feature caches
vlm_feat_gate_prefix_{real,synthLR}.npz. Prior retrains happen in the chain script.
"""
import json
import os

import numpy as np
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
HFB = "/home/ubuntu/hf_bundle/gate-drone-pi0"
H, AD, K = 50, 32, 5
STRIDE, BS, LAM = 8, 16, 10.0
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = LEFT.replace("left", "right")


def resize224(img):
    return np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC)).astype(np.uint8)


def seg_to_Y(seg, amean, astd):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def ridge_rrr_U(X, Y, k, lam):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float64)
    W = np.linalg.solve(A, Xb.T @ Y)
    Yhat = Xb @ W
    Yc = Yhat - Yhat.mean(0)
    C = (Yc.T @ Yc) / len(Yc)
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k].astype(np.float32)


def angles(U1, U2):
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1))).round(1)


def load_domain(kind):
    """-> list of {image, wrist, state, action, lang} with authoritative labels."""
    eps = []
    if kind == "synthLR":
        for i in range(100, 200):  # eps 100-149 LEFT, 150-199 RIGHT per the task map
            d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
            eps.append({"image": d["image"], "wrist": d["wrist"],
                        "state": d["state"].astype(np.float32),
                        "action": d["action"].astype(np.float32),
                        "lang": LEFT if i < 150 else RIGHT})
    else:
        meta = json.load(open(f"{RD}/data_gate_real/meta.json"))
        for key in sorted(meta):
            d = np.load(f"{RD}/data_gate_real/{key}.npz", allow_pickle=True)
            eps.append({"image": d["image"], "wrist": d["wrist"],
                        "state": d["state"].astype(np.float32),
                        "action": d["action"].astype(np.float32),
                        "lang": meta[key]["lang"]})
    return eps


def main():
    import openpi.shared.normalize as _normalize
    ns = _normalize.load(f"{HFB}/assets/gate_nav")
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)

    policy = None

    def get_policy():
        nonlocal policy
        if policy is None:
            import openpi.training.config as _config
            import openpi.policies.policy_config as _policy_config
            policy = _policy_config.create_trained_policy(
                _config.get_config("pi0_gate"), f"{HFB}/checkpoints/gate_both_pin", norm_stats=ns)
        return policy

    def extract(eps, recs, cache):
        if os.path.exists(cache):
            z = np.load(cache)
            print(f"loaded {cache} {z['X'].shape}", flush=True)
            return z["X"]
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        pol = get_policy()

        def prefix_pool(raws):
            tds = [pol._input_transform(dict(r)) for r in raws]
            batched = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
            obs = _model.preprocess_observation(None, _model.Observation.from_dict(batched), train=False)
            tokens, mask, _ = pol._model.embed_prefix(obs)
            m = mask[..., None].astype(jnp.float32)
            tk = tokens.astype(jnp.float32)
            return np.asarray((tk * m).sum(1) / jnp.clip(m.sum(1), 1e-6))

        Xs = []
        for i in range(0, len(recs), BS):
            raws = []
            for r in recs[i:i + BS]:
                ep = eps[r["ei"]]
                img = resize224(ep["image"][r["t"]])
                raws.append({"observation/image": img,
                             "observation/wrist_image": resize224(ep["wrist"][r["t"]]),
                             "observation/state": ep["state"][r["t"]], "prompt": ep["lang"]})
            Xs.append(prefix_pool(raws))
            if i % (BS * 20) == 0:
                print(f"  feat {i}/{len(recs)}", flush=True)
        X = np.concatenate(Xs, 0).astype(np.float32)
        np.savez_compressed(cache, X=X)
        print(f"extracted {cache} {X.shape}", flush=True)
        return X

    dom = {}
    for kind in ("synthLR", "real"):
        eps = load_domain(kind)
        rng = np.random.default_rng(0)
        trep = set(rng.permutation(len(eps))[:int(0.8 * len(eps))].tolist())
        recs = [{"ei": ei, "t": t, "tr": ei in trep}
                for ei, ep in enumerate(eps) for t in range(0, len(ep["action"]), STRIDE)]
        X = extract(eps, recs, f"{RD}/vlm_feat_gate_prefix_{kind}.npz")
        Y = np.stack([seg_to_Y(eps[r["ei"]]["action"][r["t"]:], amean, astd) for r in recs])
        tr = np.array([r["tr"] for r in recs])
        dom[kind] = {"X": X, "Y": Y.astype(np.float32), "tr": tr}
        print(f"{kind}: eps={len(eps)} rows={len(recs)} train={int(tr.sum())}", flush=True)

    U = {k: ridge_rrr_U(d["X"][d["tr"]], d["Y"][d["tr"]], K, LAM) for k, d in dom.items()}
    Xp = np.concatenate([dom["synthLR"]["X"][dom["synthLR"]["tr"]], dom["real"]["X"][dom["real"]["tr"]]])
    Yp = np.concatenate([dom["synthLR"]["Y"][dom["synthLR"]["tr"]], dom["real"]["Y"][dom["real"]["tr"]]])
    U["pooled"] = ridge_rrr_U(Xp, Yp, K, LAM)

    U_dep = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32)
    for a, b, ua, ub in [("synthLR", "real", U["synthLR"], U["real"]),
                         ("synthLR", "deployed", U["synthLR"], U_dep),
                         ("real", "deployed", U["real"], U_dep),
                         ("pooled", "deployed", U["pooled"], U_dep),
                         ("pooled", "real", U["pooled"], U["real"]),
                         ("pooled", "synthLR", U["pooled"], U["synthLR"])]:
        print(f"angles {a:8s} vs {b:8s}: {angles(ua, ub)}", flush=True)

    for k, name in [("real", "pin_U_gate_rrr_real_k5.npy"),
                    ("pooled", "pin_U_gate_rrr_pooled_k5.npy"),
                    ("synthLR", "pin_U_gate_rrr_synthLR_k5.npy")]:
        np.save(f"{RD}/{name}", U[k])
        print("SAVED", f"{RD}/{name}", U[k].shape, flush=True)
    print("PER_DOMAIN_U_DONE", flush=True)


if __name__ == "__main__":
    main()
