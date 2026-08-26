"""Toy-scale end-to-end validation of source-noise action steering.

Task: 2D point robot. Scene = a target placement p (the "object"). A
demonstration is an H-step delta-action chunk tracing a curved path from the
origin to p, bending left or right of the straight line (multimodal style),
with a randomized speed profile. The chunk invariant L(a) = sum_t a_t equals
p exactly by construction.

Arms (all tiny flow-matching MLPs, identical capacity where possible):
  A  v(x_t, t, obs)            plain noise            (vanilla policy)
  B  v(x_t, t, obs, m)         plain noise            (conditioning branch)
  C  v(x_t, t, obs)            pinned noise L(eps)=m  (source-carried, ours)
  D  v(x_t, t, obs, m)         pinned noise           (both)
  P  v(x_t, t, obs)            pin DROPOUT p=0.2      (CFG-style: pin present
     for 80% of training samples, plain Gaussian for the rest; one model
     supports both pinned sampling [steering] and unpinned sampling [plain
     policy], with inference-time pin strength alpha as the guidance knob)

Evaluations:
  in-dist / held-out endpoint error   train angles [0,300); test [300,360)
  wrong-invariant probe               obs says p, m commands R(120)p:
                                      which one does the rollout follow?
  diversity                           at fixed (obs, m): do both left/right
                                      styles still appear across draws?

Usage:
  python toy_flow.py --arm C          # train + eval one arm -> results JSON
  python toy_flow.py --report         # aggregate table from JSONs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from snmvp import SourceConstructor, extract_invariant  # noqa: E402

H = 20
D = 2
HID = 128
ITERS = 10000
BATCH = 256
ACT_SCALE = 5.0  # normalize actions to O(1), the toy analog of q01/q99 norm
EULER_STEPS = 20
TRAIN_ANGLES = (0.0, 300.0)
HELDOUT_ANGLES = (300.0, 360.0)
SUCCESS_TOL = 0.15
OUT_DIR = Path(__file__).parent / "results"


def make_episodes(n, rng, angle_range):
    """Returns obs (n,2), chunks (n,H,2). L(chunk) == obs exactly."""
    ang = np.deg2rad(rng.uniform(*angle_range, size=n))
    rad = rng.uniform(1.0, 2.0, size=n)
    p = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    side = rng.choice([-1.0, 1.0], size=(n, 1))
    gamma = rng.uniform(0.7, 1.4, size=(n, 1))
    s = np.linspace(0, 1, H + 1)[None, :] ** gamma  # (n, H+1) progress
    perp = np.stack([-p[:, 1], p[:, 0]], axis=1)
    perp = perp / np.linalg.norm(perp, axis=1, keepdims=True)
    bump = 0.5 * side * np.sin(np.pi * s)  # zero at both ends
    curve = p[:, None, :] * s[..., None] + perp[:, None, :] * bump[..., None]
    chunks = np.diff(curve, axis=1)  # (n, H, 2), sums to p exactly
    noise = 0.02 * rng.normal(size=chunks.shape)
    chunks = chunks + noise - noise.mean(axis=1, keepdims=True)  # keep sum
    return p, chunks * ACT_SCALE  # model works in normalized units;
    # L(chunk) = ACT_SCALE * p, endpoints divided back out in evaluate()


PIN_DROPOUT = 0.2  # arm P: fraction of training samples left unpinned
ALPHA_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)


def uses_m_input(arm):
    return arm in ("B", "D")


def pins_noise(arm):
    return arm in ("C", "D", "P", "Q")


def input_dim(arm):
    return (H * D + 1 + 2 + (2 if uses_m_input(arm) else 0)
            + (1 if arm == "Q" else 0))


def init_params(arm, rng):
    dims = [input_dim(arm), HID, HID, HID, H * D]
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def forward(params, xt, t, obs, m, arm, pin_flag=None):
    parts = [xt.reshape(xt.shape[0], -1), t.reshape(-1, 1), obs]
    if uses_m_input(arm):
        parts.append(m)
    if arm == "Q":  # CFG-style presence flag: 1 = noise is pinned
        parts.append(pin_flag.reshape(-1, 1))
    h = anp.concatenate(parts, axis=1)
    for w, b in params[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = params[-1]
    return (h @ w + b).reshape(xt.shape[0], H, D)


def make_loss(arm, obs_all, chunks_all, sc):
    m_all = extract_invariant(chunks_all)

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, len(obs_all), size=BATCH)
        obs, a0, m = obs_all[idx], chunks_all[idx], m_all[idx]
        eps = rng.normal(size=a0.shape)
        flag = anp.ones(BATCH)
        if arm in ("P", "Q"):
            keep = (rng.random(BATCH) >= PIN_DROPOUT)[:, None, None]
            eps = np.where(keep, sc(eps, m), eps)
            flag = keep[:, 0, 0].astype(float)
        elif pins_noise(arm):
            eps = sc(eps, m)
        t = rng.uniform(0, 1, size=BATCH)
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a0
        v = eps - a0
        vhat = forward(params, xt, t, obs, m, arm, pin_flag=flag)
        return anp.mean((vhat - v) ** 2)

    return loss


def rollout(params, arm, obs, m, sc, rng):
    """Euler ODE from t=1 (noise) to t=0 (actions). Returns (n, H, D)."""
    eps = rng.normal(size=(obs.shape[0], H, D))
    x = sc(eps, m) if pins_noise(arm) else eps
    flag = np.full(obs.shape[0], float(getattr(sc, "alpha", 1.0) > 0))
    dt = 1.0 / EULER_STEPS
    for k in range(EULER_STEPS):
        t = np.full(obs.shape[0], 1.0 - k * dt)
        v = forward(params, x, t, obs, m, arm, pin_flag=flag)
        x = x - dt * v
    return x


def evaluate(params, arm, sc, rng):
    res = {}
    for name, angles in (("in_dist", TRAIN_ANGLES), ("held_out", HELDOUT_ANGLES)):
        p, _ = make_episodes(200, rng, angles)
        chunks = rollout(params, arm, p, p * ACT_SCALE, sc, rng)
        err = np.linalg.norm(chunks.sum(1) / ACT_SCALE - p, axis=1)
        res[f"{name}_err"] = float(err.mean())
        res[f"{name}_success"] = float((err < SUCCESS_TOL).mean())

    # wrong-invariant probe: obs says p, invariant commands q = R(120deg) p
    if arm != "A":
        p, _ = make_episodes(200, rng, TRAIN_ANGLES)
        c, s = np.cos(np.deg2rad(120)), np.sin(np.deg2rad(120))
        q = p @ np.array([[c, s], [-s, c]])
        chunks = rollout(params, arm, p, q * ACT_SCALE, sc, rng)
        end = chunks.sum(1) / ACT_SCALE
        d_cmd = np.linalg.norm(end - q, axis=1)
        d_obs = np.linalg.norm(end - p, axis=1)
        res["probe_follow_noise_rate"] = float((d_cmd < d_obs).mean())
        res["probe_err_to_command"] = float(d_cmd.mean())

    # arm P extras: (b) unpinned-mode endpoint error (does pin training hurt or
    # help the plain obs-following policy?) and an alpha guidance sweep on the
    # wrong-invariant probe (CFG-style steering strength).
    if arm in ("P", "Q"):
        p, _ = make_episodes(200, rng, TRAIN_ANGLES)
        sc0 = SourceConstructor(alpha=0.0)
        chunks = rollout(params, arm, p, p * ACT_SCALE, sc0, rng)
        err = np.linalg.norm(chunks.sum(1) / ACT_SCALE - p, axis=1)
        res["unpinned_in_dist_err"] = float(err.mean())
        p_h, _ = make_episodes(200, rng, HELDOUT_ANGLES)
        chunks = rollout(params, arm, p_h, p_h * ACT_SCALE, sc0, rng)
        err = np.linalg.norm(chunks.sum(1) / ACT_SCALE - p_h, axis=1)
        res["unpinned_held_out_err"] = float(err.mean())

        c120, s120 = np.cos(np.deg2rad(120)), np.sin(np.deg2rad(120))
        p, _ = make_episodes(200, rng, TRAIN_ANGLES)
        q = p @ np.array([[c120, s120], [-s120, c120]])
        sweep = {}
        for a in ALPHA_SWEEP:
            sca = SourceConstructor(alpha=a)
            chunks = rollout(params, arm, p, q * ACT_SCALE, sca, rng)
            end = chunks.sum(1) / ACT_SCALE
            sweep[str(a)] = {
                "err_to_command": float(np.linalg.norm(end - q, axis=1).mean()),
                "err_to_obs": float(np.linalg.norm(end - p, axis=1).mean()),
                "follow_noise_rate": float(
                    (np.linalg.norm(end - q, axis=1)
                     < np.linalg.norm(end - p, axis=1)).mean()),
            }
        res["alpha_sweep"] = sweep

    # diversity: fixed scene, 40 draws; does bimodal left/right style survive?
    p, _ = make_episodes(1, rng, TRAIN_ANGLES)
    p = np.repeat(p, 40, axis=0)
    chunks = rollout(params, arm, p, p * ACT_SCALE, sc, rng)
    mid = chunks[:, : H // 2].sum(1) / ACT_SCALE  # midpoint of each rollout
    perp = np.array([-p[0, 1], p[0, 0]]) / np.linalg.norm(p[0])
    offs = mid @ perp
    res["diversity_frac_left"] = float((offs > 0).mean())
    res["diversity_mid_spread"] = float(offs.std())
    return res


def train_arm(arm, seed):
    rng = np.random.default_rng(seed)
    obs, chunks = make_episodes(4096, rng, TRAIN_ANGLES)
    sc = SourceConstructor()
    loss = make_loss(arm, obs, chunks, sc)
    params = init_params(arm, rng)
    t0 = time.time()
    trace = []

    def cb(p, it, g):
        if it % 500 == 0:
            trace.append((it, float(loss(p, 10 ** 6 + it))))

    params = adam(grad(loss), params, num_iters=ITERS, step_size=1e-3,
                  callback=cb)
    res = evaluate(params, arm, sc, np.random.default_rng(seed + 1))
    res.update({"arm": arm, "seed": seed, "train_seconds": time.time() - t0,
                "loss_trace": trace})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"arm_{arm}_seed{seed}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "loss_trace"},
                     indent=2))


def report():
    rows = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("arm_*.json"))]
    if not rows:
        print("no results yet")
        return
    cols = ["arm", "seed", "in_dist_err", "held_out_err", "held_out_success",
            "probe_follow_noise_rate", "diversity_frac_left",
            "diversity_mid_spread"]
    print("  ".join(f"{c:>22}" for c in cols))
    for r in rows:
        print("  ".join(f"{r.get(c, float('nan')):>22.3f}"
                        if not isinstance(r.get(c), str)
                        else f"{r.get(c):>22}" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list("ABCDPQ"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        train_arm(args.arm, args.seed)
