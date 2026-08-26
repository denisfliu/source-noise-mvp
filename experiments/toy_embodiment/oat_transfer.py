"""OAT-invariant cross-embodiment transfer battery (the "old idea", with OAT).

Replaces the hand-built coherence frame S_A with a FROZEN OAT tokenizer's coarse
token prefix as the embodiment-shared invariant, and runs the same freeze-and-
adapt transfer as experiments/toy_embodiment/battery.py, HEAD-TO-HEAD with the
coherence arms on identical data/seeds/scenes.

Front-half (frozen, trained on set A {arm2,arm3,arm4}):
  - OAT tokenizer  (oat.py, nested-dropout ordered, FSQ bottleneck)
  - scene-obs -> shared coarse-prefix-latents prior (regression, set-A pooled)

Per held-out body B, adapt ONLY a small flow executor on B's few demos. Arms:
  OATpin    prefix pinned into a reserved SUBSPACE of the source noise
            (the project's "invariant in source noise" mechanism, generalized
            from FFT-phase pins to a learned latent subspace), command = prior
  OATcond   prefix fed as a CONDITIONING input instead of pinned, command=prior
  OATorac   OATpin executor commanded by body B's OWN prefix (ceiling)
  S         scratch (no invariant) -- shared with the coherence battery
  Trand     prefix pinned but commanded by a RANDOM (shuffled-scene) prefix
  cohT      coherence S_A pinned into FFT-phase noise, prior command (old method)
  cohCond   coherence S_A conditioned, prior command

Gate (results/oat_gate) already showed the coarse prefix is goal-rich and
body-agnostic; this battery asks whether that makes it a USEFUL transfer
invariant (beats scratch) and how it compares to the coherence frame.
"""

import json
import os
import sys
import time

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
import embodiments as emb          # noqa: E402
import mb_dataset as ds            # noqa: E402
import dataset as tfd              # noqa: E402
import flow_embod as fe            # noqa: E402  (coherence arms + eval helpers)
import transfer_smoke as sm        # noqa: E402  (flat_body, success_rate)
import oat                         # noqa: E402
from pin import extract_mags, extract_phases  # noqa: E402

H = ds.H
HID = 128
EULER = 20
OBS = 5
SET_A = fe.SET_A
K_PREF = 4                          # coarse-prefix length (gate: goal captured, body low)
OUT = os.path.join(HERE, "results", "oat_transfer")

BODIES_B = ["point", "arm4"]
SEEDS = [0, 1, 2]
NS = [10, 25]
ITERS = 10000
N_ROLL = 8
N_EVAL = 100


# --------------------------- OAT invariant ----------------------------------

def prefix_latents(tok_params, cfg, chunks_canon_flat):
    """Coarse-prefix continuous latents (N, K_PREF*d_fsq) from canonical chunks."""
    q = np.asarray(oat.encode_q(tok_params, cfg, chunks_canon_flat))
    return q[:, :K_PREF, :].reshape(q.shape[0], -1)


class OATFrame:
    """Frozen OAT tokenizer + standardization of the coarse-prefix latents."""

    def __init__(self, tok_params, cfg, mu, sd):
        self.tok, self.cfg, self.mu, self.sd = tok_params, cfg, mu, sd
        self.m = K_PREF * cfg.d_fsq

    def encode(self, chunks_canon):
        """chunks_canon (...,H,2) -> standardized prefix (..., m)."""
        flat = np.asarray(chunks_canon).reshape(-1, self.cfg.in_dim)
        z = (prefix_latents(self.tok, self.cfg, flat) - self.mu) / self.sd
        return z.reshape(*np.asarray(chunks_canon).shape[:-2], self.m)


def fit_frame(A_chunks, A_angles, seed=0):
    cfg = oat.OATConfig(H=H, D=2, H_l=8, d_fsq=2, levels=5, hid=128)
    pool = np.concatenate([tfd.to_canonical(A_chunks[b], A_angles[:, None])
                           for b in SET_A], axis=1).reshape(-1, cfg.in_dim)
    tok = oat.train(cfg, pool, seed=seed, iters=8000)
    z = prefix_latents(tok, cfg, pool)
    return OATFrame(tok, cfg, z.mean(0), z.std(0) + 1e-6)


# --------------------------- executor (flow) --------------------------------

def _mlp_init(in_dim, out_dim, rng):
    dims = [in_dim, HID, HID, HID, out_dim]
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def _fwd(params, xt, t, obs, feats):
    parts = [xt.reshape(xt.shape[0], -1), t.reshape(-1, 1), obs]
    if feats is not None:
        parts.append(feats)
    h = anp.concatenate(parts, axis=1)
    for w, b in params[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = params[-1]
    return (h @ w + b).reshape(xt.shape[0], H, 2)


def _pin_subspace(eps, inv):
    """Overwrite the first m coords of flattened source noise with the invariant
    (the reserved subspace). eps (N,H,2), inv (N,m) -> pinned eps (N,H,2)."""
    flat = eps.reshape(eps.shape[0], -1).copy()
    flat[:, :inv.shape[1]] = inv
    return flat.reshape(eps.shape)


def make_loss(obs, chunks, canon, inv, cond, batch=256):
    """inv (N,m) per-sample invariant (for the pin arm, or None).
    cond=True -> feed inv as conditioning feats instead of pinning."""
    n = obs.shape[0]

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, n, size=batch)
        a0 = chunks[idx]
        eps = rng.normal(size=(batch, H, 2))
        if inv is not None and not cond:
            eps = _pin_subspace(eps, inv[idx])
        feats = inv[idx] if (inv is not None and cond) else None
        t = rng.uniform(0, 1, size=batch)
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a0
        v = eps - a0
        return anp.mean((_fwd(params, xt, t, obs[idx], feats) - v) ** 2)

    return loss


def train_exec(obs, chunks, canon, inv, cond, seed, iters):
    in_dim = H * 2 + 1 + OBS + ((inv.shape[1]) if (inv is not None and cond) else 0)
    params = _mlp_init(in_dim, H * 2, np.random.default_rng(seed))
    return adam(grad(make_loss(obs, chunks, canon, inv, cond)), params,
                num_iters=iters, step_size=1e-3)


def rollout(params, obs, inv, cond, rng):
    n = obs.shape[0]
    eps = rng.normal(size=(n, H, 2))
    if inv is not None and not cond:
        eps = _pin_subspace(eps, inv)
    feats = inv if (inv is not None and cond) else None
    x = eps
    dt = 1.0 / EULER
    for k in range(EULER):
        t = np.full(n, 1.0 - k * dt)
        x = x - dt * _fwd(params, x, t, obs, feats)
    return x


def eval_oat(params, scenes, obs, inv, cond, n_roll, seed):
    rng = np.random.default_rng(seed)
    M = obs.shape[0]
    obs_r = np.tile(obs, (n_roll, 1))
    inv_r = np.tile(inv, (n_roll, 1)) if inv is not None else None
    ch = rollout(params, obs_r, inv_r, cond, rng).reshape(n_roll, M, H, 2)
    return sm.success_rate(scenes, ch)


# ------------------------------ prior ----------------------------------------

def train_prefix_prior(obs, y, seed, iters=4000):
    """obs (M,5) -> shared prefix (M,m) regression."""
    rng = np.random.default_rng(seed)
    M, m = y.shape
    params = _mlp_init2([OBS, 64, 64, m], rng)
    ys = y

    def loss(p, it):
        r = np.random.default_rng(it)
        idx = r.integers(0, M, size=min(128, M))
        return anp.mean((_net(p, obs[idx]) - ys[idx]) ** 2)

    return adam(grad(loss), params, num_iters=iters, step_size=1e-3)


def _mlp_init2(dims, rng):
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def _net(p, x):
    h = x
    for w, b in p[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = p[-1]
    return h @ w + b


def predict_prefix(prior, obs):
    return np.asarray(_net(prior, obs))


def shared_prefix_target(frame, A_chunks, A_angles):
    """Per-scene set-A-shared prefix = mean prefix over bodies+demos (gate showed
    the coarse prefix is body-agnostic, so this is a clean shared target)."""
    per = []
    for b in SET_A:
        can = tfd.to_canonical(A_chunks[b], A_angles[:, None])   # (M,N,H,2)
        per.append(frame.encode(can))                            # (M,N,m)
    return np.mean(np.stack(per), axis=(0, 2))                   # (M,m)


# ------------------------------- battery -------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "rows.jsonl")
    open(rows_path, "w").close()
    t0 = time.time()
    bodies = emb.make_bodies()

    A_scenes, A_obs, A_angles, A_chunks = ds.make_dataset(
        bodies, 200, 8, np.random.default_rng(7))
    print(f"[{time.time()-t0:.0f}s] freezing OAT frame + coherence S_A ...", flush=True)
    frame = fit_frame(A_chunks, A_angles, seed=0)
    S_A, _ = fe.freeze_frame(A_chunks, A_angles)
    A_pool = np.concatenate([tfd.to_canonical(A_chunks[b], A_angles[:, None])
                             for b in SET_A], axis=1).reshape(-1, H, 2)
    mag_norm = extract_mags(A_pool, S_A).mean(axis=0)

    he_scenes, he_obs, he_angles, he_chunks = ds.make_dataset(
        bodies, N_EVAL, 8, np.random.default_rng(7777))

    # frozen priors (per seed): OAT prefix + coherence
    oat_prior, coh_prior = {}, {}
    y_shared = shared_prefix_target(frame, A_chunks, A_angles)
    for s in SEEDS:
        oat_prior[s] = train_prefix_prior(A_obs, y_shared, seed=300 + s)
        coh_prior[s] = fe.build_shared_prior(A_chunks, A_obs, A_angles, SET_A, S_A, 100 + s)

    rows = []
    for B in BODIES_B:
        heB0 = tfd.to_canonical(he_chunks[B][:, 0], he_angles)      # (100,H,2)
        inv_oracle = frame.encode(heB0)                            # B's own prefix
        for s in SEEDS:
            oat_cmd = predict_prefix(oat_prior[s], he_obs)         # prior prefix
            # random-scene control: shuffle the prior prefix across held-out scenes
            rperm = np.random.default_rng(900 + s).permutation(len(he_obs))
            oat_rand = oat_cmd[rperm]
            ct, cm, _ = fe.prior_predict(coh_prior[s], he_obs, S_A)
            coh_feats = sm._feats_from_targets(ct, cm, S_A, mag_norm)
            for n in NS:
                ad_s, ad_o, ad_a, ad_c = ds.make_dataset(
                    bodies, n, 8, np.random.default_rng(1234 + s))
                obsB, chB, angB = sm.flat_body(ad_c, ad_o, ad_a, B, 8)
                canB = tfd.to_canonical(chB, angB)
                invB = frame.encode(canB)                          # train-time pin/cond

                pPin = train_exec(obsB, chB, canB, invB, False, s, ITERS)
                pCon = train_exec(obsB, chB, canB, invB, True, s, ITERS)
                pS = train_exec(obsB, chB, canB, None, False, s, ITERS)
                # coherence arms via flow_embod
                pcohT = fe.train_executor(obsB, chB, angB, S_A, None, mag_norm, s, ITERS)
                pcohC = fe.train_executor(obsB, chB, angB, [], S_A, mag_norm, s, ITERS)

                row = {"B": B, "seed": s, "n": n, "K_pref": K_PREF,
                       "OATpin": eval_oat(pPin, he_scenes, he_obs, oat_cmd, False, N_ROLL, 50),
                       "OATcond": eval_oat(pCon, he_scenes, he_obs, oat_cmd, True, N_ROLL, 51),
                       "OATorac": eval_oat(pPin, he_scenes, he_obs, inv_oracle, False, N_ROLL, 52),
                       "Trand": eval_oat(pPin, he_scenes, he_obs, oat_rand, False, N_ROLL, 53),
                       "S": eval_oat(pS, he_scenes, he_obs, None, False, N_ROLL, 54),
                       "cohT": sm.eval_arm(pcohT, he_scenes, he_obs, he_angles, S_A, ct, cm, None, N_ROLL, 55),
                       "cohCond": sm.eval_arm(pcohC, he_scenes, he_obs, he_angles, [], None, None, coh_feats, N_ROLL, 56)}
                row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                rows.append(row)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{time.time()-t0:.0f}s] {B} s{s} n{n}: "
                      f"OATpin={row['OATpin']} OATcond={row['OATcond']} "
                      f"S={row['S']} cohT={row['cohT']} cohCond={row['cohCond']} "
                      f"(OATorac={row['OATorac']} Trand={row['Trand']})", flush=True)

    verdict = summarize(rows)
    json.dump({"rows": rows, "verdict": verdict, "K_pref": K_PREF},
              open(os.path.join(OUT, "battery.json"), "w"), indent=2)
    print("VERDICT:", json.dumps(verdict, indent=2))
    print(f"OAT_TRANSFER_DONE=ok in {time.time()-t0:.0f}s")


def summarize(rows):
    out = {}
    for B in BODIES_B:
        d = {}
        for n in NS:
            r = [x for x in rows if x["B"] == B and x["n"] == n]
            for k in ["OATpin", "OATcond", "OATorac", "Trand", "S", "cohT", "cohCond"]:
                d[f"{k}_n{n}"] = round(float(np.mean([x[k] for x in r])), 3)
        # pooled over n
        r = [x for x in rows if x["B"] == B]
        mean = {k: float(np.mean([x[k] for x in r]))
                for k in ["OATpin", "OATcond", "OATorac", "Trand", "S", "cohT", "cohCond"]}
        d["OATpin_gt_S"] = bool(mean["OATpin"] > mean["S"])
        d["OATpin_gt_Trand"] = bool(mean["OATpin"] > mean["Trand"])
        d["OATpin_vs_cohT"] = round(mean["OATpin"] - mean["cohT"], 3)
        d["OATcond_vs_OATpin"] = round(mean["OATcond"] - mean["OATpin"], 3)
        d["best_vs_scratch"] = round(max(mean["OATpin"], mean["OATcond"]) - mean["S"], 3)
        out[B] = d
    out["_reading"] = ("OATpin>S = OAT prefix transfers; OATpin>Trand = the "
                       "LEARNED prefix specifically (not any subspace pin); "
                       "OATpin_vs_cohT = OAT invariant vs hand-built coherence "
                       "frame, same data; OATorac = ceiling if the prefix were "
                       "known for B.")
    return out


if __name__ == "__main__":
    main()
