"""Rebuild the VLM-RRR K=5 pin basis on the LOCAL machine (2026-08-19 continuation).

Adapted from tmp_scripts_rescue/make_u_rrr_gate.py (the box recipe: OLS(pi0 VLM prefix features
-> normalized H=50 chunk), U = top-K eigenvectors of Cov(Yhat)) with two deliberate changes:

  1. Paths are this machine's (~/hf_bundle, local data_gate_synth regenerated 2026-08-19 — episode
     ORDER may differ from the box's, so nothing here assumes alignment with old artifacts).
  2. The per-episode language comes from data_gate_synth/meta.json (the dataset's own task
     strings) instead of the binary is_left trajectory labeler. The record board flags that
     binary labeler as the label-bug class that folded center demos into left/right; meta.json
     carries all four tasks.

Sanity check (box reference: RRR ~= PCA within 0.2 deg on 4/5 directions): principal angles
between the fitted RRR basis and chunk-PCA are printed, plus held-out c-R2 for RRR vs PCA priors.
Prints the sha256 of the saved basis (pin_basis.stamp uses the same hash at prior-build time).

Feature checkpoint: gate_both_pin (pre-box generation, restored from HF bucket). The basis is
checkpoint-robust per refit_rrr_basis findings; the ordering rule (U -> flow -> features -> prior)
is unaffected because U is ~chunk statistics (that is what the RRR~=PCA check verifies).

Run:  PYTHONPATH=~/code/openpi-snmvp/src ~/code/openpi/.venv/bin/python make_u_rrr_gate_local.py
"""
import hashlib
import json
import os

import numpy as np
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
DD = os.path.join(RD, "data_gate_synth")
HFB = os.path.expanduser("~/hf_bundle/gate-drone-pi0")
# hf_bundle's checkpoint dirs hold only assets on this machine; the restored params live in
# the falsify checkout (smoke-verified SMOKE_OK 2026-08-19)
FEAT_CKPT = os.path.expanduser("~/code/falsify/local/checkpoints/gate_both_pin")
H, AD, K = 50, 32, 5
STRIDE = 8
BS = 16
CACHE = os.path.join(RD, "vlm_feat_gate_prefix_local.npz")
OUT = os.path.join(RD, "pin_U_gate_rrr_k5.npy")


def resize224(img):
    return np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC)).astype(np.uint8)


def seg_to_Y(seg, amean, astd):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def rrr_U(X, Y, k):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Yhat = Xb @ W
    Yc = Yhat - Yhat.mean(0)
    C = (Yc.T @ Yc) / len(Yc)
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k].astype(np.float32)


def pca_U(Y, k):
    Yc = Y - Y.mean(0)
    C = (Yc.T @ Yc) / len(Yc)
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k].astype(np.float32)


def principal_angles(A, B):
    qa = np.linalg.qr(A)[0]
    qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def fit_mlp(Xtr, Ytr, Xte, steps=4000):
    import torch
    import torch.nn as nn
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.SiLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt = torch.tensor(((Xtr - m) / s).astype(np.float32))
    yt = torch.tensor(Ytr.astype(np.float32))
    for _ in range(steps):
        b = torch.randint(0, len(xt), (256,))
        loss = ((net(xt[b]) - yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(((Xte - m) / s).astype(np.float32))).numpy()


def main():
    import openpi.shared.normalize as _normalize
    ns = _normalize.load(f"{HFB}/assets/gate_nav")
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
    meta = json.load(open(f"{DD}/meta.json"))
    eps = []
    for name in sorted(meta):
        d = np.load(f"{DD}/{name}.npz", allow_pickle=True)
        eps.append({"image": d["image"], "wrist": d["wrist"],
                    "state": d["state"].astype(np.float32),
                    "action": d["action"].astype(np.float32),
                    "lang": str(meta[name]["lang"]), "task": int(meta[name]["task"])})
    tasks = sorted(set(e["task"] for e in eps))
    print(f"eps={len(eps)} tasks={ {t: sum(e['task'] == t for e in eps) for t in tasks} }", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(eps))
    ntr = int(0.8 * len(eps))
    trep = set(idx[:ntr].tolist())
    recs = []
    for ei, ep in enumerate(eps):
        T = len(ep["action"])
        for t in range(0, T, STRIDE):
            recs.append({"ei": ei, "t": t, "sp": "tr" if ei in trep else "te",
                         "Y": seg_to_Y(ep["action"][t:], amean, astd)})
    print(f"frames={len(recs)}", flush=True)

    def make_obs(ep, t):
        img = resize224(ep["image"][t])
        wrist = resize224(ep["wrist"][t])
        return {"observation/image": img, "observation/wrist_image": wrist,
                "observation/state": ep["state"][t], "prompt": ep["lang"]}

    if os.path.exists(CACHE):
        z = np.load(CACHE)
        X, ST = z["X"], z["ST"]
        print("loaded cached feats", X.shape, flush=True)
    else:
        import jax
        import jax.numpy as jnp
        import openpi.policies.policy_config as _policy_config
        import openpi.training.config as _config
        from openpi.models import model as _model
        policy = _policy_config.create_trained_policy(
            _config.get_config("pi0_gate"), FEAT_CKPT, norm_stats=ns)

        def _obs(raws):
            tds = [policy._input_transform(dict(r)) for r in raws]
            batched = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
            return _model.preprocess_observation(None, _model.Observation.from_dict(batched), train=False)

        def prefix_pool(raws):
            obs = _obs(raws)
            tokens, mask, _ = policy._model.embed_prefix(obs)
            m = mask[..., None].astype(jnp.float32)
            tk = tokens.astype(jnp.float32)
            return np.asarray((tk * m).sum(1) / jnp.clip(m.sum(1), 1e-6)), np.asarray(obs.state)

        Xs, STs = [], []
        for i in range(0, len(recs), BS):
            raws = [make_obs(eps[r["ei"]], r["t"]) for r in recs[i:i + BS]]
            fx, st = prefix_pool(raws)
            Xs.append(fx); STs.append(st)
            if i % (BS * 20) == 0:
                print(f"  feat {i}/{len(recs)}", flush=True)
        X = np.concatenate(Xs, 0).astype(np.float32)
        ST = np.concatenate(STs, 0).astype(np.float32)
        np.savez_compressed(CACHE, X=X, ST=ST)
        print("extracted", X.shape, flush=True)

    Y = np.stack([r["Y"] for r in recs]).astype(np.float32)
    sp = np.array([r["sp"] for r in recs])
    tr, te = sp == "tr", sp == "te"
    STn = ST.reshape(len(ST), -1)
    oh = np.zeros((len(recs), len(tasks)), np.float32)
    for i, r in enumerate(recs):
        oh[i, tasks.index(eps[r["ei"]]["task"])] = 1.0
    Xstate = np.concatenate([STn, oh], 1).astype(np.float32)

    Urrr = rrr_U(X[tr], Y[tr], K)
    Upca = pca_U(Y[tr], K)

    def evalU(U, prior_feat, name):
        C = Y @ U
        pred = fit_mlp(prior_feat[tr], C[tr], prior_feat[te])
        cov = float(C[tr].var(0).sum() / (Y[tr].var(0).sum() + 1e-9))
        print("== %-26s held c-R2 = %+.3f  coverage=%.3f" % (name, r2(pred, C[te]), cov), flush=True)

    evalU(Urrr, X, "VLM-RRR (feat prior)")
    evalU(Upca, Xstate, "PCA U (state+task prior)")
    evalU(Urrr, Xstate, "VLM-RRR (state+task prior)")

    ang = principal_angles(Urrr, Upca)
    print(f"principal angles RRR vs PCA (deg): {np.round(ang, 2)}  "
          f"(box reference: ~=PCA within 0.2 deg on 4/5 dirs)", flush=True)

    np.save(OUT, Urrr)
    sha = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
    print(f"SAVED {OUT} {Urrr.shape} sha256={sha}", flush=True)
    print(f"feat_ckpt={FEAT_CKPT}", flush=True)
    print("RRR_GATE_LOCAL_DONE", flush=True)


if __name__ == "__main__":
    main()
