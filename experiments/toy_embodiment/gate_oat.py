"""Rung-1 OAT gate (go/no-go for the OAT-as-invariant direction).

Train ONE OAT tokenizer jointly on set-A {arm2,arm3,arm4} canonical task-space
chunks, then ask whether the coarse->fine token ordering aligns with a
shared->body-specific split:

  (1) recon MSE vs prefix length K       -> confirms coarse->fine (paper Fig 2).
  (2) per-token MI with body / goal       -> does token 0 carry goal not body?
  (3) prefix decodability of body / goal  -> the transfer-relevant question:
      if we FREEZE the first K tokens as the shared invariant, how much body
      identity leaks in (near chance = clean) vs how much goal is captured?

GATE PASSES if some prefix length K* is goal-rich AND body-poor (body decode
near chance while goal decode high), and body decodability RISES with K (finer
tokens absorb the body-idiosyncratic residual). That means the early-token
prefix is a learned, ordered, bottlenecked cross-embodiment invariant and the
VLA^2 / OAT-invariant experiments are worth building. If early tokens already
leak body identity, OAT's reconstruction ordering is transcribing and we need a
hybrid external ordering signal before proceeding.
"""

import json
import os
import sys

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
import embodiments as emb          # noqa: E402
import mb_dataset as ds            # noqa: E402
import dataset as tfd              # noqa: E402  (toy_frame)
import oat                         # noqa: E402

SET_A = ["arm2", "arm3", "arm4"]
OUT = os.path.join(HERE, "results", "oat_gate")


# ------------------------------- data ---------------------------------------

def build_setA(n_scenes=300, n_demos=8, seed=7):
    """Pooled set-A canonical chunks with per-sample body id + goal features.
    Goal features are in the CANONICAL frame (target bearing already removed),
    so what remains is the embodiment-shared task content: how far (radius) and
    the obstacle-detour geometry (signed lateral clearance)."""
    bodies = emb.make_bodies()
    scenes, obs, angles, chunks = ds.make_dataset(
        {b: bodies[b] for b in SET_A}, n_scenes, n_demos, np.random.default_rng(seed))
    X, body, radius, lateral = [], [], [], []
    for bi, b in enumerate(SET_A):
        can = tfd.to_canonical(chunks[b], angles[:, None])   # (M,N,H,2)
        M, N = can.shape[:2]
        X.append(can.reshape(M * N, -1))
        body.append(np.full(M * N, bi))
        radius.append(np.repeat([s["radius"] for s in scenes], N))
        lateral.append(np.repeat([s["lateral"] for s in scenes], N))
    return (np.concatenate(X), np.concatenate(body).astype(int),
            np.concatenate(radius), np.concatenate(lateral))


def qbin(x, q=5):
    """Quantile-bin a continuous array into q roughly-equal-mass bins."""
    edges = np.quantile(x, np.linspace(0, 1, q + 1)[1:-1])
    return np.digitize(x, edges)


# ------------------------------- MI -----------------------------------------

def mi_bits(x, y):
    """Plug-in mutual information (bits) between two discrete int arrays."""
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = x.max() + 1, y.max() + 1
    joint = np.zeros((nx, ny))
    np.add.at(joint, (x, y), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (px @ py)[nz])))


def mi_profile(tokens, label):
    """Per-token MI(T_k; label) and a shuffle floor (estimator bias)."""
    rng = np.random.default_rng(0)
    lab_s = rng.permutation(label)
    return ([round(mi_bits(tokens[:, k], label), 4) for k in range(tokens.shape[1])],
            round(float(np.mean([mi_bits(tokens[:, k], lab_s)
                                 for k in range(tokens.shape[1])])), 4))


# --------------------- prefix decodability (logistic) -----------------------

def logistic_acc(feats, y, seed=0, iters=1500):
    """Test accuracy of multinomial logistic regression y ~ feats (standardized).
    60/40 train/test split. Chance = 1/n_classes."""
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    n, d = feats.shape
    C = int(y.max() + 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(0.6 * n)
    tr, te = perm[:cut], perm[cut:]
    Xtr, ytr, Xte, yte = feats[tr], y[tr], feats[te], y[te]
    W0 = rng.normal(size=(d, C)) * 0.01
    b0 = np.zeros(C)

    def loss(p, it):
        W, b = p
        z = Xtr @ W + b
        z = z - anp.max(z, axis=1, keepdims=True)
        logp = z - anp.log(anp.sum(anp.exp(z), axis=1, keepdims=True))
        return -anp.mean(logp[anp.arange(len(ytr)), ytr]) + 1e-3 * anp.sum(W ** 2)

    W, b = adam(grad(loss), (W0, b0), num_iters=iters, step_size=5e-2)
    pred = np.argmax(np.asarray(Xte) @ np.asarray(W) + np.asarray(b), axis=1)
    return round(float((pred == yte).mean()), 4), round(1.0 / C, 4)


def prefix_decode(q_lat, y, cfg, seed=0):
    """Decodability of y from the first-K token latents, K=1..H_l."""
    out = []
    for K in range(1, cfg.H_l + 1):
        feats = q_lat[:, :K, :].reshape(q_lat.shape[0], -1)
        acc, chance = logistic_acc(feats, y, seed)
        out.append({"K": K, "acc": acc, "chance": chance})
    return out


# ------------------------------- main ---------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    print("building set-A pooled canonical chunks ...", flush=True)
    X, body, radius, lateral = build_setA()
    rbin, latbin = qbin(radius), qbin(np.abs(lateral))   # |lateral| = clearance size
    print(f"  N={len(X)} samples, {len(SET_A)} bodies, chunk dim={X.shape[1]}",
          flush=True)

    cfg = oat.OATConfig(H=ds.H, D=2, H_l=8, d_fsq=2, levels=5, hid=128)
    print(f"training OAT (H_l={cfg.H_l}, {cfg.d_fsq} FSQ dims x {cfg.levels} "
          f"levels = codebook {cfg.codebook}/token) ...", flush=True)
    params = oat.train(cfg, X, seed=0, iters=8000)

    # (1) coarse->fine: recon MSE vs prefix length
    full = oat.recon_at_K(params, cfg, X, cfg.H_l)
    mse_K = [round(oat.recon_at_K(params, cfg, X, K), 5) for K in range(1, cfg.H_l + 1)]
    var = float(X.var())
    print(f"recon MSE full={full:.5f} (data var={var:.4f}); MSE vs K={mse_K}",
          flush=True)

    # (2) per-token MI
    tokens = oat.tokenize(params, cfg, X)                # (N,H_l)
    mi_body, floor_body = mi_profile(tokens, body)
    mi_rad, floor_rad = mi_profile(tokens, rbin)
    mi_lat, floor_lat = mi_profile(tokens, latbin)
    print(f"MI(T_k;body)   = {mi_body}  (shuffle floor {floor_body})", flush=True)
    print(f"MI(T_k;radius) = {mi_rad}  (floor {floor_rad})", flush=True)
    print(f"MI(T_k;latera) = {mi_lat}  (floor {floor_lat})", flush=True)

    # (3) prefix decodability (the transfer-leakage question)
    q_lat = np.asarray(oat.encode_q(params, cfg, X))     # (N,H_l,d_fsq)
    dec_body = prefix_decode(q_lat, body, cfg)
    dec_rad = prefix_decode(q_lat, rbin, cfg)
    print("prefix body-decode:", [f"K{d['K']}:{d['acc']}" for d in dec_body],
          f"(chance {dec_body[0]['chance']})", flush=True)
    print("prefix goal-decode:", [f"K{d['K']}:{d['acc']}" for d in dec_rad],
          f"(chance {dec_rad[0]['chance']})", flush=True)

    # verdict: find a prefix that is goal-rich AND body-poor
    chance_b = dec_body[0]["chance"]
    best = None
    for k in range(cfg.H_l):
        goal_gain = dec_rad[k]["acc"] - dec_rad[k]["chance"]
        body_excess = dec_body[k]["acc"] - chance_b
        # goal captured, body near chance (allow 8pt slack over chance)
        if goal_gain > 0.15 and body_excess < 0.08:
            best = {"K": k + 1, "goal_acc": dec_rad[k]["acc"],
                    "body_acc": dec_body[k]["acc"], "chance_body": chance_b}
            break
    body_rises = dec_body[-1]["acc"] - dec_body[0]["acc"] > 0.05
    coarse_to_fine = all(mse_K[i] >= mse_K[i + 1] - 1e-4 for i in range(len(mse_K) - 1))
    gate_pass = bool(best is not None and body_rises)

    verdict = {"gate_pass": gate_pass, "clean_prefix": best,
               "body_decode_rises_with_K": bool(body_rises),
               "coarse_to_fine_recon": bool(coarse_to_fine),
               "recon_mse_full": round(full, 5), "data_var": round(var, 4),
               "note": "gate_pass = a frozen early-token prefix carries the goal "
                       "(decode >> chance) while leaking little body identity "
                       "(decode ~ chance), and body id concentrates in later "
                       "tokens. If true, OAT gives an ordered cross-embodiment "
                       "invariant and VLA^2/OAT-invariant transfer is worth building."}
    result = {"config": {"H_l": cfg.H_l, "d_fsq": cfg.d_fsq, "levels": cfg.levels,
                         "codebook_per_token": cfg.codebook, "N": len(X),
                         "set_A": SET_A},
              "mse_vs_K": mse_K, "recon_full": round(full, 5), "data_var": round(var, 4),
              "mi_token_body": mi_body, "mi_floor_body": floor_body,
              "mi_token_radius": mi_rad, "mi_floor_radius": floor_rad,
              "mi_token_lateral": mi_lat, "mi_floor_lateral": floor_lat,
              "prefix_decode_body": dec_body, "prefix_decode_goal": dec_rad,
              "verdict": verdict}
    json.dump(result, open(os.path.join(OUT, "gate.json"), "w"), indent=2)
    print("\nVERDICT:", json.dumps(verdict, indent=2))
    print("GATE_DONE=ok")


if __name__ == "__main__":
    main()
