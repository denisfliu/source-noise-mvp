"""Offline sim->real test of the source-noise pin. The FROZEN sim flow (pi0 source-pin,
trained on LIBERO) is served with BRIDGE normalization injected. On held-out real Bridge
trajectories we compare, in a controlled way (same Gaussian per sample; arms differ only in
the pinned coordinate c), how each arm predicts the real action chunk:

  no_pin     : unpinned Gaussian -> the sim flow alone on real.
  mean_c     : c = mean training c (state-independent).
  real_prior : c = prior(state), prior refit on real Bridge demos (THE HYPOTHESIS).
  oracle     : c = U^T (ground-truth real action chunk) -> what a perfect pin channel gives.

Because K=5 is a tiny fraction of the 1600-dim action, full-action error is dominated by the
free complement the sim flow generates (out of domain on real), so we DECOMPOSE the metric:
  (1) PASS-THROUGH: for pinned arms, ||c_pred - c_injected|| / ||c_injected|| -- does the frozen
      flow still pass the pinned coordinate through when fed real inputs?
  (2) SUBSPACE R^2: R^2 of c_pred vs c_gt in the K-dim pin subspace -- does the pin CHANNEL
      re-ground to the real instruction (isolates the pin from the complement)?
  (3) FULL-ACTION R^2 / MSE: standardized per-channel over the horizon overlap -- does it
      produce correct real actions overall (complement included)?
c_pred/c_gt are U^T of the normalized, zero-time-padded action chunk (Bridge episodes ~34 < H=50)."""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/code/source-noise-mvp/experiments/rung3"))
import pca_pin as PP  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.policies.policy_config as _policy_config  # noqa: E402
import openpi.shared.normalize as _normalize  # noqa: E402

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")


def load_eps(raw_dir):
    eps = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "ep_*.npz"))):
        d = np.load(f)
        ep = {"image": d["image"], "state": d["state"].astype(np.float32),
              "action": d["action"].astype(np.float32)}
        if "wrist" in d:
            ep["wrist"] = d["wrist"]
        eps.append(ep)
    return eps


def resize224(img):
    if img.shape[:2] != (224, 224):
        img = np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC))
    return img.astype(np.uint8)


def make_obs(ep, t, prompt):
    img = resize224(ep["image"][t])
    wrist = resize224(ep["wrist"][t]) if "wrist" in ep else img.copy()
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": ep["state"][t], "prompt": prompt}


def seg_to_c(seg_raw, H, amean, astd, ad, U):
    """Normalized, zero-time-padded (H, ad) chunk from a raw (m,7) segment -> U-coordinate."""
    m, r = seg_raw.shape
    seg = seg_raw[:H] if m >= H else np.concatenate([seg_raw, np.zeros((H - m, r), np.float32)], 0)
    segn = (seg - amean[:r]) / (astd[:r] + 1e-6)
    chunk = np.zeros((H, ad), np.float32)
    chunk[:, :r] = segn
    return chunk.reshape(-1) @ U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--U", default=os.path.join(RD, "pin_U_pca_k5.npy"))
    ap.add_argument("--config", default="pi0_libero_low_mem_finetune")
    ap.add_argument("--norm", default=os.path.join(RD, "bridge_norm"))
    ap.add_argument("--raw_dir", default=os.path.join(RD, "data_bridge_raw"))
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=80)
    ap.add_argument("--offsets", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RD, "simreal_offline.json"))
    args = ap.parse_args()

    U = PP.load_U(args.U)
    ns = _normalize.load(args.norm)
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
    cfg = _config.get_config(args.config)
    H, ad = cfg.model.action_horizon, cfg.model.action_dim
    policy = _policy_config.create_trained_policy(cfg, args.ckpt, norm_stats=ns)

    eps = load_eps(args.raw_dir)
    meta = json.load(open(os.path.join(args.raw_dir, "meta.json")))
    lang_list = [meta[k]["lang"] for k in sorted(meta.keys())]
    train_eps = eps[:args.n_train]
    eval_eps = eps[args.n_train:args.n_train + args.n_eval]
    eval_langs = lang_list[args.n_train:args.n_train + args.n_eval]

    # --- fit real prior state(32) -> c(K) on train episodes ---
    S, C = [], []
    for ep in train_eps:
        T = len(ep["action"])
        for t in range(0, T, max(1, T // 6)):
            st = np.asarray(policy._input_transform(make_obs(ep, t, ""))["state"]).reshape(-1)
            S.append(st)
            C.append(seg_to_c(ep["action"][t:], H, amean, astd, ad, U))
    S, C = np.asarray(S), np.asarray(C)
    ntr = int(0.85 * len(S))
    W, b = PP.fit_state_prior(S[:ntr], C[:ntr])
    Cte = C[ntr:]
    prior_r2 = 1 - ((Cte - PP.apply_prior(W, b, S[ntr:])) ** 2).sum() / (((Cte - Cte.mean(0)) ** 2).sum() + 1e-9)
    mean_c = C.mean(0)
    print(f"real prior fit: n={len(S)} K={C.shape[1]} heldout state->c R^2={prior_r2:.3f}")

    arms = ["no_pin", "mean_c", "real_prior", "oracle"]
    # accumulators
    full = {a: {"res": np.zeros(7), "sg": np.zeros(7), "sg2": np.zeros(7), "n": 0,
                "pmag": 0.0, "np": 0} for a in arms}
    cg_all = []                                   # c_gt per sample (shared)
    cp_all = {a: [] for a in arms}                # c_pred per sample per arm
    ci_all = {a: [] for a in arms if a != "no_pin"}   # c_injected per arm
    gmag = 0.0
    rng = np.random.default_rng(args.seed)
    n_samples = 0
    for ei, (ep, lang) in enumerate(zip(eval_eps, eval_langs)):
        T = len(ep["action"])
        if T < 4:
            continue
        for t in np.linspace(0, max(0, T - 4), args.offsets).astype(int):
            obs = make_obs(ep, t, lang)
            st = np.asarray(policy._input_transform(obs)["state"]).reshape(-1)
            c_real = PP.apply_prior(W, b, st[None])[0]
            overlap = min(H, T - t)
            gt = ep["action"][t:t + overlap]
            c_gt = seg_to_c(gt, H, amean, astd, ad, U)
            cg_all.append(c_gt)
            gmag += np.abs(gt).mean()
            g = rng.standard_normal((H, ad)).astype(np.float32)
            gf = g.reshape(-1)

            def pin(c):
                return (gf - (gf @ U) @ U.T + (c @ U.T)).reshape(H, ad).astype(np.float32)

            cinj = {"mean_c": mean_c, "real_prior": c_real, "oracle": c_gt}
            noises = {"no_pin": g, "mean_c": pin(mean_c), "real_prior": pin(c_real), "oracle": pin(c_gt)}
            for a in arms:
                pred = np.asarray(policy.infer(obs, noise=noises[a])["actions"])[:overlap, :7]
                d = full[a]
                d["res"] += ((pred - gt) ** 2).sum(0); d["sg"] += gt.sum(0)
                d["sg2"] += (gt ** 2).sum(0); d["n"] += overlap
                d["pmag"] += np.abs(pred).mean(); d["np"] += 1
                cp_all[a].append(seg_to_c(pred, H, amean, astd, ad, U))
                if a != "no_pin":
                    ci_all[a].append(cinj[a])
            n_samples += 1
        if (ei + 1) % 20 == 0:
            print(f"...eval {ei + 1}/{len(eval_eps)} eps, {n_samples} samples")

    # --- metrics ---
    cg = np.asarray(cg_all)
    tot_c = ((cg - cg.mean(0)) ** 2).sum() + 1e-9
    w = 1.0 / (astd[:7] ** 2 + 1e-9)
    rows = {}
    for a in arms:
        d = full[a]
        tot = d["sg2"] - d["sg"] ** 2 / max(d["n"], 1)
        full_r2 = 1 - (d["res"] * w).sum() / ((tot * w).sum() + 1e-9)
        mse_std = (d["res"] * w).sum() / (d["n"] * 7 + 1e-9)
        cp = np.asarray(cp_all[a])
        sub_r2 = 1 - ((cp - cg) ** 2).sum() / tot_c
        row = {"full_r2": float(full_r2), "mse_std": float(mse_std), "subspace_r2": float(sub_r2),
               "pred_mag": float(d["pmag"] / max(d["np"], 1))}
        if a != "no_pin":
            ci = np.asarray(ci_all[a])
            row["passthrough_relerr"] = float(np.linalg.norm(cp - ci, axis=1).mean()
                                              / (np.linalg.norm(ci, axis=1).mean() + 1e-9))
        rows[a] = row

    print(f"\ngt_action_mag={gmag / max(n_samples,1):.4f}  (raw |action| mean; pred_mag should be similar)\n")
    print(f"{'arm':>11} | {'subspace_R2':>11} | {'passthru_err':>12} | {'full_R2':>8} | {'MSE_std':>7} | {'pred_mag':>8}")
    print("-" * 78)
    for a in arms:
        r = rows[a]
        print(f"{a:>11} | {r['subspace_r2']:11.3f} | "
              f"{r.get('passthrough_relerr', float('nan')):12.3f} | "
              f"{r['full_r2']:8.3f} | {r['mse_std']:7.3f} | {r['pred_mag']:8.4f}")
    out = {"prior_r2": float(prior_r2), "n_samples": int(n_samples), "n_train_eps": len(train_eps),
           "n_eval_eps": len(eval_eps), "K": int(U.shape[1]), "gt_action_mag": float(gmag / max(n_samples, 1)),
           "arms": rows, "ckpt": args.ckpt}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}\nOFFLINE_EVAL_DONE")


if __name__ == "__main__":
    main()
