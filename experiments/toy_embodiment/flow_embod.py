"""Cross-embodiment executors + freeze-and-adapt transfer (Rung 1, Steps 2-4).

Self-contained (no snmvp): reuses pin.py (pin_noise/extract_phases/extract_mags)
and mb_dataset. The flow-matching executor + confidence-gated scene prior are
adapted from toy_frame/flow.py (same conventions, RNG discipline).

Architecture (docs/cross_embodiment_plan.md):
  - Frozen shared frame S_A: coherence-discovered pin set over set A, with the
    toy_frame energy-floor + magnitude-CV gating (frozen BEFORE any executor
    training). freeze_frame() below.
  - Frozen shared front-half: scene->invariant prior trained on set A pooled
    over bodies+demos (the set-A-shared invariant per scene).
  - Per-body executor: flow head, S_A pinned into its (task-space) source noise.
    The ONLY thing re-learned on a new body.

Transfer arms on held-out body B (held-out scenes, no oracle unless noted):
  T        executor adapted with S_A pin,   command from frozen shared prior
  S        executor adapted plain (no pin),  no command (obs->action)
  Cond     executor adapted with invariant as CONDITIONING input, prior command
  Trand    executor adapted with a RANDOM frame pin, its own random-frame prior
  Toracle  T's executor, commanded by body B's OWN ground-truth invariant
"""

import json
import os
import sys

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "toy_frame"))
sys.path.insert(0, os.path.dirname(__file__))
import mb_dataset as ds                 # noqa: E402
import coherence_xembod as cx           # noqa: E402
import dataset as tfd                   # noqa: E402  (toy_frame, to_canonical)
from pin import (circular_dist, extract_mags, extract_phases,  # noqa: E402
                 pin_noise)

H = ds.H
HID = 128
EULER_STEPS = 20
OBS_DIM = 5
SET_A = ["arm2", "arm3", "arm4"]
THETAS = np.linspace(0.0, np.pi, 90)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ------------------------- frozen shared frame S_A --------------------------

def freeze_frame(chunks, angles, set_a=SET_A, energy_floor=0.10,
                 cv_thresh=0.15, coh_thresh=0.6):
    """Energy-floored, CV-gated cross-body pin set on the {progress,lateral}
    axes (canonical frame). Returns the pin list AND a diagnostics dict."""
    pooled = np.concatenate([tfd.to_canonical(chunks[b], angles[:, None]).reshape(-1, H, 2)
                             for b in set_a])              # (K,H,2)
    stk = cx.stack_bodies(chunks, angles, set_a)           # (M,|A|,H,2)
    B = H // 2 + 1
    S, diag = [], {}
    for name, (ux, uy) in (("prog", (1.0, 0.0)), ("lat", (0.0, 1.0))):
        u = np.array([ux, uy])
        zp = pooled @ u
        E = (np.abs(np.fft.rfft(zp, axis=-1)) ** 2).mean(0)
        Efrac = E / E.sum()
        mag = np.abs(np.fft.rfft(zp, axis=-1))
        cv = mag.std(0) / (mag.mean(0) + 1e-9)
        phi = np.angle(np.fft.rfft(stk @ u, axis=-1))      # (M,|A|,B)
        g = np.abs(np.exp(1j * phi).mean(1)).mean(0)
        g2 = np.abs(np.exp(2j * phi).mean(1)).mean(0)
        for om in range(B):
            if Efrac[om] < energy_floor:
                continue
            real_bin = om == 0 or om == H // 2
            has_mag = bool(cv[om] < cv_thresh)
            if g[om] > coh_thresh:
                S.append({"axis": (ux, uy), "omega": om, "mode": "mod2pi", "mag": has_mag})
            elif (not real_bin) and g2[om] > coh_thresh:
                S.append({"axis": (ux, uy), "omega": om, "mode": "modpi", "mag": has_mag})
        diag[name] = {"energy_frac": np.round(Efrac, 3).tolist(),
                      "gamma": np.round(g, 3).tolist(),
                      "gamma2": np.round(g2, 3).tolist(),
                      "cv": np.round(cv, 3).tolist()}
    return S, diag


def rand_frame(seed):
    r = np.random.default_rng(2000 + seed)
    th = np.radians(r.uniform(25, 65))
    u1 = (float(np.cos(th)), float(np.sin(th)))
    u2 = (float(-np.sin(th)), float(np.cos(th)))
    bins = r.choice(np.arange(1, 9), size=4, replace=False)
    return [{"axis": u1, "omega": int(bins[0]), "mode": "mod2pi", "mag": True},
            {"axis": u1, "omega": int(bins[1]), "mode": "mod2pi", "mag": True},
            {"axis": u2, "omega": int(bins[2]), "mode": "modpi", "mag": True},
            {"axis": u2, "omega": int(bins[3]), "mode": "modpi", "mag": False}]


# ------------------------------- executor ----------------------------------

def n_mag(pins):
    return sum(1 for p in pins if p.get("mag"))


def features(chunks_canonical, pins, mag_norm):
    phi = extract_phases(chunks_canonical, pins)
    mag = extract_mags(chunks_canonical, pins)
    feats = []
    for k, p in enumerate(pins):
        mult = 2.0 if p["mode"] == "modpi" else 1.0
        feats += [np.cos(mult * phi[..., k]), np.sin(mult * phi[..., k])]
    for k, p in enumerate(pins):
        if p.get("mag"):
            feats.append(mag[..., k] / mag_norm[k])
    return np.stack(feats, axis=-1)


def input_dim(cond_pins):
    extra = (2 * len(cond_pins) + n_mag(cond_pins)) if cond_pins else 0
    return H * 2 + 1 + OBS_DIM + extra


def init_params(cond_pins, rng):
    dims = [input_dim(cond_pins), HID, HID, HID, H * 2]
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def forward(params, xt, t, obs, feats):
    parts = [xt.reshape(xt.shape[0], -1), t.reshape(-1, 1), obs]
    if feats is not None:
        parts.append(feats)
    h = anp.concatenate(parts, axis=1)
    for w, b in params[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = params[-1]
    return (h @ w + b).reshape(xt.shape[0], H, 2)


def make_loss(obs_all, chunks_all, angles_all, pins, cond_pins, mag_norm,
              batch=256):
    canon_all = tfd.to_canonical(chunks_all, angles_all)
    n = obs_all.shape[0]

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, n, size=batch)
        eps = rng.normal(size=(batch, H, 2))
        t = rng.uniform(0, 1, size=batch)
        obs, a0, ang = obs_all[idx], chunks_all[idx], angles_all[idx]
        a0c = canon_all[idx]
        if pins:                                  # pin into (task-space) noise
            eps_c = tfd.to_canonical(eps, ang)
            eps_c = pin_noise(eps_c, pins, extract_phases(a0c, pins),
                              mag_targets=extract_mags(a0c, pins))
            eps = tfd.to_canonical(eps_c, -ang)
        feats = features(a0c, cond_pins, mag_norm) if cond_pins else None
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a0
        v = eps - a0
        return anp.mean((forward(params, xt, t, obs, feats) - v) ** 2)

    return loss


def train_executor(obs, chunks, angles, pins, cond_pins, mag_norm, seed,
                   iters):
    loss = make_loss(obs, chunks, angles, pins, cond_pins, mag_norm)
    params = init_params(cond_pins, np.random.default_rng(seed))
    return adam(grad(loss), params, num_iters=iters, step_size=1e-3)


def rollout(params, obs, angles, pins, pin_targets, cond_feats, rng,
            mag_targets=None):
    n = obs.shape[0]
    eps = rng.normal(size=(n, H, 2))
    if pins and pin_targets is not None:
        eps_c = tfd.to_canonical(eps, angles)
        eps_c = pin_noise(eps_c, pins, pin_targets, mag_targets=mag_targets,
                          orient_from_noise=True)
        eps = tfd.to_canonical(eps_c, -angles)
    x = eps
    dt = 1.0 / EULER_STEPS
    for k in range(EULER_STEPS):
        t = np.full(n, 1.0 - k * dt)
        x = x - dt * forward(params, x, t, obs, cond_feats)
    return x


# ---------------------------- shared prior ---------------------------------

def prior_targets(chunks_canon, pins):
    phi = extract_phases(chunks_canon, pins)
    mag = extract_mags(chunks_canon, pins)
    M = phi.shape[0]
    res = np.zeros((M, len(pins), 2))
    mg = np.full((M, len(pins)), np.nan)
    for k, p in enumerate(pins):
        mult = 2.0 if p["mode"] == "modpi" else 1.0
        z = np.exp(1j * mult * phi[..., k]).mean(axis=1)
        res[:, k, 0], res[:, k, 1] = z.real, z.imag
        if p.get("mag"):
            mg[:, k] = mag[..., k].mean(axis=1)
    return res, mg


def train_prior(obs_all, res, mg, pins, seed, iters=4000):
    rng = np.random.default_rng(seed)
    M, P, _ = res.shape
    mag_cols = [k for k, p in enumerate(pins) if p.get("mag")]
    mag_scale = np.nanmean(mg, axis=0)
    y = np.concatenate([res.reshape(M, -1)] +
                       ([mg[:, mag_cols] / mag_scale[mag_cols]] if mag_cols else []),
                       axis=1)
    dims = [OBS_DIM, 64, 64, y.shape[1]]
    params = [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
              for a, b in zip(dims[:-1], dims[1:])]

    def net(p, x):
        h = x
        for w, b in p[:-1]:
            h = anp.maximum(0.0, h @ w + b)
        w, b = p[-1]
        return h @ w + b

    def loss(p, it):
        r = np.random.default_rng(it)
        idx = r.integers(0, M, size=min(128, M))
        return anp.mean((net(p, obs_all[idx]) - y[idx]) ** 2)

    params = adam(grad(loss), params, num_iters=iters, step_size=1e-3)
    return {"params": params, "net": net, "mag_cols": mag_cols,
            "mag_scale": mag_scale, "P": P}


def prior_predict(prior, obs, pins, conf_threshold=0.6):
    P = prior["P"]
    out = np.asarray(prior["net"](prior["params"], obs))
    v = out[:, :P * 2].reshape(-1, P, 2)
    conf = np.linalg.norm(v, axis=-1)
    ang = np.arctan2(v[..., 1], v[..., 0])
    targets = np.where(conf > conf_threshold, ang, np.nan)
    mags = np.full((obs.shape[0], P), np.nan)
    for j, k in enumerate(prior["mag_cols"]):
        m = out[:, P * 2 + j] * prior["mag_scale"][k]
        mags[:, k] = np.where(conf[:, k] > conf_threshold, m, np.nan)
    for k, p in enumerate(pins):
        if p["mode"] == "modpi":
            targets[:, k] = targets[:, k] / 2.0
    return targets, mags, conf


def build_shared_prior(chunks, obs, angles, set_a, pins, seed):
    """Prior over the set-A-SHARED invariant: pool bodies+demos per scene."""
    pooled = np.concatenate([tfd.to_canonical(chunks[b], angles[:, None])
                             for b in set_a], axis=1)      # (M, |A|*N, H, 2)
    res, mg = prior_targets(pooled, pins)
    return train_prior(obs, res, mg, pins, seed=seed)
