"""Flow-matching arms for the learned-frame toy — HYBRID pin version (D1/H).

Derived from experiments/toy/toy_flow.py conventions. v2 (2026-07-05): arms
use the frozen hybrid set pin.HYBRID_PINS (phase pins everywhere in S_H,
magnitude additionally pinned where cross-demo CV < 0.15 — pre-registered).
v1 (phase-only) results are archived in results/battery_phase_only/.

Arms:
  A       plain noise                                  (floor)
  Cdisp   displacement pin (snmvp SourceConstructor)   (hand-defined reference)
  F       hybrid pin, HYBRID_PINS                      (learned frame)
  Frand   hybrid pin, random frame (per-seed mirror)   (control)
  Bphase  plain noise + phase&mag features as input    (channel control)

RNG discipline: idx, eps, t drawn in identical order in every arm; pins
consume no RNG. F with pins=[] is bit-identical to A (verified v1).
"""

import pickle
import sys
import time
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from snmvp import SourceConstructor, extract_invariant  # noqa: E402
from dataset import ACT_SCALE, H, make_dataset, to_canonical  # noqa: E402
from pin import HYBRID_PINS, extract_mags, extract_phases, pin_noise  # noqa: E402

HID = 128
ITERS = 10000
BATCH = 256
EULER_STEPS = 20
OBS_DIM = 5
OUT_DIR = Path(__file__).parent / "results"
N_TRAIN_SCENES, N_DEMOS, N_HELD = 200, 8, 100
DATA_SEED = 7

N_MAG = sum(1 for p in HYBRID_PINS if p.get("mag"))


def rand_pins(seed):
    """Random-frame mirror of HYBRID_PINS: same mode/mag structure."""
    r = np.random.default_rng(1000 + seed)
    th = np.radians(r.uniform(25, 65))
    u1 = (float(np.cos(th)), float(np.sin(th)))
    u2 = (float(-np.sin(th)), float(np.cos(th)))
    bins = r.choice(np.arange(1, 10), size=4, replace=False)
    return [{"axis": u1, "omega": int(bins[0]), "mode": "mod2pi", "mag": True},
            {"axis": u1, "omega": int(bins[1]), "mode": "mod2pi", "mag": True},
            {"axis": u2, "omega": int(bins[2]), "mode": "modpi", "mag": True},
            {"axis": u2, "omega": int(bins[3]), "mode": "modpi", "mag": False}]


def arm_pins(arm, seed):
    if arm == "F":
        return HYBRID_PINS
    if arm == "Frand":
        return rand_pins(seed)
    return []


def features(chunks_canonical, pins, mag_norm):
    """Conditioning features: (cos,sin) per pin (doubled angle for modpi)
    + normalized magnitude per mag pin. Information parity with the pin."""
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


def input_dim(arm):
    return H * 2 + 1 + OBS_DIM + (
        (2 * len(HYBRID_PINS) + N_MAG) if arm == "Bphase" else 0)


def init_params(arm, rng):
    dims = [input_dim(arm), HID, HID, HID, H * 2]
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


def prep_pinned_noise(arm, eps, a0_canon, angles, pins, sc_disp, a0_global):
    if arm == "Cdisp":
        return sc_disp(eps, extract_invariant(a0_global))
    if pins:
        eps_c = to_canonical(eps, angles)
        phis = extract_phases(a0_canon, pins)
        mags = extract_mags(a0_canon, pins)
        eps_c = pin_noise(eps_c, pins, phis, mag_targets=mags)
        return to_canonical(eps_c, -angles)
    return eps


def make_loss(arm, obs_all, chunks_all, angles_all, pins):
    canon_all = to_canonical(chunks_all, angles_all)
    sc_disp = SourceConstructor()
    n = obs_all.shape[0]
    mag_norm = extract_mags(canon_all, HYBRID_PINS).mean(axis=0)

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, n, size=BATCH)
        eps = rng.normal(size=(BATCH, H, 2))
        t = rng.uniform(0, 1, size=BATCH)
        obs, a0, ang = obs_all[idx], chunks_all[idx], angles_all[idx]
        a0c = canon_all[idx]
        eps = prep_pinned_noise(arm, eps, a0c, ang, pins, sc_disp, a0)
        feats = features(a0c, HYBRID_PINS, mag_norm) if arm == "Bphase" else None
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a0
        v = eps - a0
        vhat = forward(params, xt, t, obs, feats)
        return anp.mean((vhat - v) ** 2)

    return loss


def rollout(params, arm, obs, angles, pins, pin_targets, feats, rng,
            mag_targets=None, orient_from_noise=True):
    n = obs.shape[0]
    eps = rng.normal(size=(n, H, 2))
    if arm == "Cdisp" and pin_targets is not None:
        eps = SourceConstructor()(eps, pin_targets)
    elif pins and pin_targets is not None:
        eps_c = to_canonical(eps, angles)
        eps_c = pin_noise(eps_c, pins, pin_targets, mag_targets=mag_targets,
                          orient_from_noise=orient_from_noise)
        eps = to_canonical(eps_c, -angles)
    x = eps
    dt = 1.0 / EULER_STEPS
    for k in range(EULER_STEPS):
        t = np.full(n, 1.0 - k * dt)
        x = x - dt * forward(params, x, t, obs, feats)
    return x


# ------------- scene prior (confidence-gated, phases + magnitudes) -------------

def prior_targets(chunks_canon, pins):
    """(M, P, 2) resultants + (M, P) mean magnitudes (NaN for non-mag pins)."""
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
    mag_scale = np.nanmean(mg, axis=0)                # per-pin normalizer
    y = np.concatenate(
        [res.reshape(M, -1)] +
        ([mg[:, mag_cols] / mag_scale[mag_cols]] if mag_cols else []), axis=1)
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
    """-> (phase targets (n,P) with NaN below conf, mag targets (n,P) with
    NaN for non-mag/below-conf, confidences (n,P))."""
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


def train_arm(arm, seed):
    rng = np.random.default_rng(DATA_SEED)
    scenes, obs, chunks, angles = make_dataset(N_TRAIN_SCENES, N_DEMOS, rng)
    flat_obs = np.repeat(obs, N_DEMOS, axis=0)
    flat_chunks = chunks.reshape(-1, H, 2)
    flat_angles = np.repeat(angles, N_DEMOS)
    pins = arm_pins(arm, seed)
    loss = make_loss(arm, flat_obs, flat_chunks, flat_angles, pins)
    params = init_params(arm, np.random.default_rng(seed))
    t0 = time.time()
    params = adam(grad(loss), params, num_iters=ITERS, step_size=1e-3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"arm_{arm}_seed{seed}.pkl", "wb") as f:
        pickle.dump({"params": params, "pins": pins, "arm": arm, "seed": seed},
                    f)
    print(f"trained {arm} seed {seed} in {time.time()-t0:.0f}s "
          f"final_loss={float(loss(params, 999_999)):.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["A", "Cdisp", "F", "Frand", "Bphase"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train_arm(args.arm, args.seed)
