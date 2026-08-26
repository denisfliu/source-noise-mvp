"""Drone sim->real, the target problem. Combine the three fixes found in the trajectory deep-dive:
  (A) jitter  -> build c on a LOW-PASS-smoothed action chunk (teleop high-freq jitter averaged out)
  (C) domain-sensitive basis -> define U as VLM-predictable (RRR vs pi0_gate's VLM), grounded & more
      domain-invariant than an action-variance basis
  prior -> VLM->c, which grounds the instruction the same way in sim and real.
Everything is built on SIM only and evaluated ZERO-SHOT on REAL (no real data), the honest sim->real
test; few-shot (prior fit on real) is the reference ceiling. We ablate basis {vlm,state} x smoothing
{raw,smooth} x prior {vlm,state} so we can see which fix carries it. Metric: pin-channel c-R^2 on real,
plus the c domain-gap/left-right-sep ratio per (basis,smoothing) (invariant when <1). Baseline that
failed before: state-RRR + raw + state prior, zero-shot (was ~-0.5 to -1.7)."""
import argparse
import json
import os

import numpy as np
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = "go through the gate on the right and hover over the stuffed animal"


def resize224(img):
    return np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC)).astype(np.uint8)


def make_obs(ep, t):
    img = resize224(ep["image"][t]); wrist = resize224(ep["wrist"][t])
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": ep["state"][t], "prompt": ep["lang"]}


def seg_to_Y(seg, amean, astd, smooth):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    if smooth > 1:  # low-pass along time to remove teleop jitter (factor A)
        k = np.ones(smooth, np.float32) / smooth
        seg = np.stack([np.convolve(seg[:, j], k, mode="same") for j in range(r)], 1)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1)


def rrr_U(X, Y, K):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Yh = Xb @ W; Yc = Yh - Yh.mean(0); C = (Yc.T @ Yc) / len(Yc)
    _, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :K].astype(np.float32)


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


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
    import torch as T
    with T.no_grad():
        return net(T.tensor(((Xte - m) / s).astype(np.float32))).numpy()


def load(raw, amean, astd, smooth):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    recs = []
    for k in sorted(meta):
        if meta[k]["lang"] not in (LEFT, RIGHT):
            continue
        d = np.load(os.path.join(raw, k + ".npz"))
        acts = d["action"].astype(np.float32); T = len(acts)
        recs.append({"k": k, "image": d["image"], "wrist": d["wrist"], "state": d["state"].astype(np.float32),
                     "action": acts, "lang": meta[k]["lang"], "g": 0 if meta[k]["lang"] == LEFT else 1, "T": T})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/code/openpi/checkpoints/pi0_gate/gate_synth_pin/4999"))
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--norm", default=os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
    ap.add_argument("--smooth", type=int, default=7); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--stride", type=int, default=6); ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    import openpi.shared.normalize as _normalize
    ns = _normalize.load(args.norm)
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)

    sim = load(os.path.join(RD, "data_gate_synth"), amean, astd, args.smooth)
    real = load(os.path.join(RD, "data_gate_real"), amean, astd, args.smooth)

    # anchors + targets (raw and smoothed Y) for each domain
    def anchors(recs):
        A = []
        for ei, ep in enumerate(recs):
            for t in range(0, ep["T"], args.stride):
                A.append((ei, t, ep["g"]))
        return A
    aS, aR = anchors(sim), anchors(real)
    print(f"sim frames={len(aS)} real frames={len(aR)} smooth={args.smooth}", flush=True)

    def targets(recs, A, smooth):
        return np.stack([seg_to_Y(recs[ei]["action"][t:], amean, astd, smooth) for ei, t, _ in A]).astype(np.float32)

    cache = os.path.join(RD, "drone_vlm_feat.npz")
    if os.path.exists(cache):
        z = np.load(cache); XS, STS, XR, STR = z["XS"], z["STS"], z["XR"], z["STR"]
        print(f"loaded cached drone VLM feats sim{XS.shape} real{XR.shape}", flush=True)
    else:
        import jax, jax.numpy as jnp
        import openpi.training.config as _config
        import openpi.policies.policy_config as _policy_config
        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask
        policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.ckpt, norm_stats=ns)

        def ctx(raws):
            tds = [policy._input_transform(dict(r)) for r in raws]
            b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
            obs = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
            tok, mask, ar = policy._model.embed_prefix(obs)
            out, _ = policy._model.PaliGemma.llm([tok, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
            po = out[0].astype(jnp.float32); m = mask[..., None].astype(jnp.float32)
            return np.asarray((po * m).sum(1) / jnp.clip(m.sum(1), 1e-6)), np.asarray(obs.state)

        def extract(recs, A):
            Xs, Ss = [], []
            for i in range(0, len(A), args.bs):
                raws = [make_obs(recs[ei], t) for ei, t, _ in A[i:i + args.bs]]
                fx, st = ctx(raws); Xs.append(fx); Ss.append(st)
                if i % (args.bs * 20) == 0:
                    print(f"  feat {i}/{len(A)}", flush=True)
            return np.concatenate(Xs).astype(np.float32), np.concatenate(Ss).astype(np.float32)
        print("extract SIM...", flush=True); XS, STS = extract(sim, aS)
        print("extract REAL...", flush=True); XR, STR = extract(real, aR)
        np.savez_compressed(cache, XS=XS, STS=STS, XR=XR, STR=STR)
        print("cached drone VLM feats", flush=True)

    gS = np.array([g for _, _, g in aS]); gR = np.array([g for _, _, g in aR])
    STS2, STR2 = STS.reshape(len(STS), -1), STR.reshape(len(STR), -1)

    def run(basis, smooth, prior):
        Ys = targets(sim, aS, smooth); Yr = targets(real, aR, smooth)
        featS = XS if basis == "vlm" else STS2
        U = rrr_U(featS, Ys, args.K)  # U built on SIM only (zero-shot constraint)
        Cs, Cr = Ys @ U, Yr @ U
        # domain-invariance of c
        ml_s, mr_s = Cs[gS == 0].mean(0), Cs[gS == 1].mean(0)
        ml_r, mr_r = Cr[gR == 0].mean(0), Cr[gR == 1].mean(0)
        dom = 0.5 * (np.linalg.norm(ml_s - ml_r) + np.linalg.norm(mr_s - mr_r)); sep = np.linalg.norm(ml_r - mr_r)
        pfS = XS if prior == "vlm" else STS2
        pfR = XR if prior == "vlm" else STR2
        # zero-shot: prior fit on SIM, eval REAL
        zs = r2(fit_mlp(pfS, Cs, pfR), Cr)
        # few-shot: prior fit on REAL(70%), eval REAL(30%)
        rng = np.random.default_rng(0); idx = rng.permutation(len(Cr)); cut = int(0.7 * len(Cr))
        fs = r2(fit_mlp(pfR[idx[:cut]], Cr[idx[:cut]], pfR[idx[cut:]]), Cr[idx[cut:]])
        print(f"  basis={basis:5s} smooth={'Y' if smooth>1 else 'N'} prior={prior:5s} | c-domain-ratio={dom/(sep+1e-9):5.2f} | ZERO-SHOT R2={zs:+.3f} | few-shot R2={fs:+.3f}", flush=True)

    print("\n== drone sim->real (U built on SIM; zero-shot = prior fit SIM eval REAL) ==", flush=True)
    run("state", 1, "state")           # the baseline that failed
    run("vlm", 1, "vlm")               # C only (VLM basis + VLM prior, raw c)
    run("state", args.smooth, "state") # A only (smoothed, state)
    run("vlm", args.smooth, "vlm")     # A+C: the combined fix
    run("vlm", args.smooth, "state")   # A+C basis but state prior (isolates prior)
    print("DRONE_S2R_DONE", flush=True)


if __name__ == "__main__":
    main()
