"""B1 — toy multi-continuation de-risk gate.

Question: can a pinned flow executor learn MULTI-CONTINUATION data — the SAME
observation carrying several different continuations (forward demo,
time-reversed demo b_t = -a_{H-1-t}, hover = zero chunk), each row pinned with
its own chunk's invariant exactly as standard training does — and select among
them by the commanded invariant at inference, without mode collapse and
without degrading the forward task?

Reuses (imports, not copies):
  - toy_frame/dataset.py  : scenes/demos/canonical frame/success
  - toy_frame/pin.py      : pin_noise / extract_phases / extract_mags
  - toy_embodiment/flow_embod.py : make_loss/train_executor/rollout (H=20,
    OBS_DIM=5; loss + pin construction untouched)

Pins: ALL-LINEAR set — mode "mod2pi" with mag=True at omegas {0,1,2} on both
canonical axes. Each such pin fixes the full complex rfft coefficient
F_omega(u.a) = sum_t (u.a_t) e^{-2pi i omega t / H}, i.e. two real LINEAR
functionals of the chunk; omega=0 on each axis is exactly the Phase-1
chunk-displacement invariant. (The frozen HYBRID_PINS include mod-pi /
phase-only entries, which are NOT linear functionals — excluded per the
pre-registration rule "pin only linear functionals".)

Err-to-command (the standard realized-vs-commanded metric, normalized units):
per rollout, ||F_realized - F_commanded||_2 over the 6 pinned complex
coefficients, extracted in the canonical frame, divided by the RMS command
norm of the FORWARD held-out commands (one fixed scale for all types, so
forward/reverse/hover numbers are directly comparable).

Arms (same architecture / iters / seeds):
  A  trained on FWD-ONLY rows
  B  trained on MULTI rows (fwd + reversed + hover from the SAME obs)

Pre-registered bars:
  bar1  B: reverse err <= 2 x B fwd err  AND  hover err <= 2 x B fwd err
  bar2  B fwd err <= 1.5 x A fwd err
  bar3  A (fwd-only) FAILS reverse — reported as its raw err (expected >> its
        fwd err; we flag A_rev > 2 x A_fwd as the failure confirmation)

Run:
  ~/.local/bin/uv run --with autograd --with numpy --python 3.11 \
      python experiments/toy_multicont/multicont.py [--smoke]
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
import dataset as tf                      # noqa: E402  (toy_frame)
import flow_embod as fe                   # noqa: E402  (toy_embodiment)
from pin import extract_mags, extract_phases  # noqa: E402

H = tf.H
PINS = ([{"axis": (1.0, 0.0), "omega": w, "mode": "mod2pi", "mag": True}
         for w in (0, 1, 2)] +
        [{"axis": (0.0, 1.0), "omega": w, "mode": "mod2pi", "mag": True}
         for w in (0, 1, 2)])

SMOKE = "--smoke" in sys.argv
SEEDS = [0] if SMOKE else [0, 1, 2]
ITERS = 300 if SMOKE else 8000
N_TRAIN_SCENES, N_TRAIN_DEMOS = 60, 4
N_EVAL_SCENES = 10 if SMOKE else 60
N_ROLL = 4
OUT = os.path.join(HERE, "results")


def reverse_chunks(ch):
    """Time-reversed, negated: b_t = -a_{H-1-t} (starts at the same state,
    traces the path shape backwards, ends at -endpoint)."""
    return -ch[..., ::-1, :]


def coeffs(chunks_canon):
    """Pinned complex coefficients (..., n_pins)."""
    ph = extract_phases(chunks_canon, PINS)
    mg = extract_mags(chunks_canon, PINS)
    return mg * np.exp(1j * ph)


def flatten(obs, chunks, angles):
    n_demos = chunks.shape[1]
    return (np.repeat(obs, n_demos, axis=0),
            chunks.reshape(-1, H, 2),
            np.repeat(angles, n_demos))


def eval_type(params, scenes, obs, angles, cmd_world, scale, eval_seed,
              check_success=False):
    """Pinned command-following: rollout N_ROLL per scene with the command
    chunk's invariant pinned; err = ||F_real - F_cmd|| / scale."""
    obs_t = np.repeat(obs, N_ROLL, axis=0)
    ang_t = np.repeat(angles, N_ROLL)
    cmd_t = np.repeat(cmd_world, N_ROLL, axis=0)
    cmd_c = tf.to_canonical(cmd_t, ang_t)
    ph = extract_phases(cmd_c, PINS)
    mg = extract_mags(cmd_c, PINS)
    rng = np.random.default_rng(eval_seed)
    x = fe.rollout(params, obs_t, ang_t, PINS, ph, None, rng, mag_targets=mg)
    xc = tf.to_canonical(x, ang_t)
    err = np.linalg.norm(coeffs(xc) - coeffs(cmd_c), axis=-1) / scale
    out = {"err": round(float(err.mean()), 4),
           "err_std": round(float(err.std()), 4)}
    # diagnostic (NOT a bar): full-chunk RMSE to the command chunk, canonical
    # frame. Style freedom makes this nonzero even for perfect following, but
    # a "Frankenstein" chunk (pinned coords match, rest forward-shaped) shows
    # up here while hiding in the pinned err.
    out["chunk_rmse"] = round(float(np.sqrt(((xc - cmd_c) ** 2).mean())), 4)
    # diagnostics: realized chunk size (hover should be ~0 everywhere, not
    # just at pinned coords) and endpoint spread across rollouts (diversity)
    out["mean_abs_action"] = round(float(np.abs(x).mean()), 4)
    ends = np.cumsum(x / tf.ACT_SCALE, axis=1)[:, -1].reshape(-1, N_ROLL, 2)
    out["endpoint_spread"] = round(float(ends.std(axis=1).mean()), 4)
    if check_success:
        succ = [tf.success(scenes[i], x[i * N_ROLL + r])
                for i in range(len(scenes)) for r in range(N_ROLL)]
        out["success"] = round(float(np.mean(succ)), 3)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # ---- data ----
    _, obs, chunks, angles = tf.make_dataset(
        N_TRAIN_SCENES, N_TRAIN_DEMOS, np.random.default_rng(11))
    obs_f, ch_f, ang_f = flatten(obs, chunks, angles)
    ch_rev = reverse_chunks(ch_f).copy()
    ch_hov = np.zeros_like(ch_f)
    # MULTI: contradictory continuations from the SAME observations; each row
    # is pinned with its own chunk's invariant inside fe.make_loss, exactly
    # as standard training.
    obs_m = np.concatenate([obs_f] * 3)
    ch_m = np.concatenate([ch_f, ch_rev, ch_hov])
    ang_m = np.concatenate([ang_f] * 3)

    he_scenes, he_obs, he_chunks, he_angles = tf.make_dataset(
        N_EVAL_SCENES, 1, np.random.default_rng(7777))
    cmd_fwd = he_chunks[:, 0]
    cmd_rev = reverse_chunks(cmd_fwd).copy()
    cmd_hov = np.zeros_like(cmd_fwd)
    scale = float(np.sqrt((np.abs(coeffs(tf.to_canonical(cmd_fwd, he_angles)))
                           ** 2).sum(axis=-1).mean()))
    print(f"train rows: A={len(ch_f)} B={len(ch_m)}  eval scenes="
          f"{N_EVAL_SCENES}x{N_ROLL} rollouts  scale={scale:.3f}", flush=True)

    rows = []
    for s in SEEDS:
        pA = fe.train_executor(obs_f, ch_f, ang_f, PINS, None, None, s, ITERS)
        print(f"[{time.time()-t0:.0f}s] seed {s}: A trained", flush=True)
        pB = fe.train_executor(obs_m, ch_m, ang_m, PINS, None, None, s, ITERS)
        print(f"[{time.time()-t0:.0f}s] seed {s}: B trained", flush=True)
        row = {"seed": s}
        for arm, p in (("A", pA), ("B", pB)):
            row[arm] = {
                "fwd": eval_type(p, he_scenes, he_obs, he_angles, cmd_fwd,
                                 scale, 9000 + s, check_success=True),
                "rev": eval_type(p, he_scenes, he_obs, he_angles, cmd_rev,
                                 scale, 9100 + s),
                "hover": eval_type(p, he_scenes, he_obs, he_angles, cmd_hov,
                                   scale, 9200 + s)}
        rows.append(row)
        print(f"[{time.time()-t0:.0f}s] seed {s}: " + " ".join(
            f"{a}.{k}={row[a][k]['err']}" for a in ("A", "B")
            for k in ("fwd", "rev", "hover")), flush=True)

    # ---- pooled bars ----
    pool = {a: {k: float(np.mean([r[a][k]["err"] for r in rows]))
                for k in ("fwd", "rev", "hover")} for a in ("A", "B")}
    bars = {
        "bar1_B_rev_le_2x_Bfwd": bool(pool["B"]["rev"] <= 2 * pool["B"]["fwd"]),
        "bar1_B_hover_le_2x_Bfwd": bool(pool["B"]["hover"] <= 2 * pool["B"]["fwd"]),
        "bar2_Bfwd_le_1p5x_Afwd": bool(pool["B"]["fwd"] <= 1.5 * pool["A"]["fwd"]),
        "bar3_A_fails_rev (A_rev > 2x A_fwd)": bool(
            pool["A"]["rev"] > 2 * pool["A"]["fwd"]),
    }
    bars["bar1"] = bars["bar1_B_rev_le_2x_Bfwd"] and bars["bar1_B_hover_le_2x_Bfwd"]
    out = {"pins": [{**p, "axis": list(p["axis"])} for p in PINS],
           "config": {"seeds": SEEDS, "iters": ITERS,
                      "train_scenes": N_TRAIN_SCENES, "demos": N_TRAIN_DEMOS,
                      "eval_scenes": N_EVAL_SCENES, "n_roll": N_ROLL,
                      "err_scale": round(scale, 4), "smoke": SMOKE},
           "rows": rows, "pooled_err": {a: {k: round(v, 4) for k, v in d.items()}
                                        for a, d in pool.items()},
           "bars": bars,
           "runtime_s": round(time.time() - t0, 1)}
    name = "multicont_smoke.json" if SMOKE else "multicont.json"
    json.dump(out, open(os.path.join(OUT, name), "w"), indent=2)
    print(json.dumps({"pooled_err": out["pooled_err"], "bars": bars}, indent=2))
    print(f"MULTICONT_DONE=ok in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
