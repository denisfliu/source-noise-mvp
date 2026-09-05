"""Physical realism of gate flights: is a trajectory something a quadrotor could fly the way the
demonstrations were flown? (2026-09-05, Denis: "the benefit over SDEdit is predicting physically
realizable trajectories; the command is embodiment-agnostic, denoising supplies the embodiment".)

Every trajectory here is a 10 Hz position stream (gate_nav3 meta/info.json fps=10; the sim executes
one action per frame; the hardware client publishes each step as a mocap setpoint at 10 Hz). The
2026-09-04 RESEARCH_LOG numbers were scaled as if 25 Hz -- multiply their speeds by 0.4 and their
accelerations by 0.16 to compare with this file.

Sources compared per cell: the decoded command U c executed verbatim (no flow), the pinned source
sample z executed verbatim (no flow), the pin arm (flow denoises z), SDEdit on the unpinned flow
guided by the same command, the unpinned flow alone, and the demonstrations (real mocap flights,
synth planner flights) as the realism reference.

Metric families (all on the active segment: start -> first goal-box entry + 1 s; hover tail scored
separately):
  kinematic envelope   speed / accel / jerk p95, raw finite differences AND a 0.7 s Savitzky-Golay
                       fit that is identical for every source (mocap noise inflates raw demo jerk)
  smoothness           dimensionless jerk log10(T^5/L^2 * int |j|^2 dt) (Hogan & Sternad 2009);
                       fraction of velocity power at >= 1 Hz (Nyquist 5 Hz)
  command signature    zero-accel fraction (piecewise-constant velocity: |dv| < 2 cm/s per step);
                       replan-seam ratio (|dv| at 50-step chunk boundaries / elsewhere)
  quadrotor feasibility  differential flatness: required tilt, thrust/weight, body rate from the
                       smoothed (a, j); fraction beyond the PX4 default position-controller limits
                       (MPC_ACC_HOR 3 m/s^2, MPC_JERK_MAX 8 m/s^3, MPC_TILTMAX_AIR 45 deg) and beyond
                       the real demos' own p99 envelope
  trackability         a PX4-shaped cascade (P position -> PID velocity -> accel with the limits
                       above, 100 Hz, zero-order-hold setpoints + finite-difference velocity
                       feed-forward) tracks the stream: RMS / max tracking error, chunk-endpoint error
  distinguishability   ROC-AUC of a logistic regression on 1 s windows of local kinematic features,
                       flights vs real demos, 5-fold CV grouped by trajectory (0.5 = indistinguishable;
                       synth-vs-real and real-vs-real rows calibrate the scale)

  /home/dfliu/miniforge3/bin/python3 realism.py            # all cells, writes realism_results.json
  /home/dfliu/miniforge3/bin/python3 realism.py --cells cfr --arms pin,dec
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter

DT = 0.1
G = 9.81
RUN = "/home/dfliu/ctxrun"
RD = os.path.dirname(os.path.abspath(__file__))
LEROBOT = os.path.expanduser("~/.cache/huggingface/lerobot/local/gate_nav3")
DEMO_CACHE = f"{RUN}/realism_demos.npz"
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])
SG_WIN, SG_POLY = 7, 3
CHUNK = 50
PX4 = dict(kp_xy=0.95, kp_z=1.0, kv_xy=1.8, kv_z=4.0, acc_h=3.0, acc_up=4.0, acc_dn=3.0, jerk=8.0, tilt_deg=45.0)
ZERO_ACC = 0.2      # m/s^2: |dv| < 2 cm/s per 0.1 s step
WIN, HOP = 10, 5    # 1 s classifier windows, 0.5 s hop

# cell -> ordered (arm key, label, ctxrun tag prefix). Tags from scripts/run_*.sh; see RESEARCH_LOG 2026-09-03/04.
CELLS = {
    "cfr": ("Center from right", "center", 3, [
        ("dec", "U c executed, no flow", "dec_cfr"),
        ("src", "z executed, no flow", "src_cfr"),
        ("pin", "pin (xswap s42)", "xswap_cfr"),
        ("pin_s7", "pin (xswap s7)", "xswaps7_cfr"),
        ("sde03", "SDEdit+head t0=0.3", "sdehead_t03_cfr"),
        ("sde05", "SDEdit+head t0=0.5", "sdehead_t05_cfr"),
        ("sde07", "SDEdit+head t0=0.7", "sdehead_t07_cfr"),
        ("sde09", "SDEdit+head t0=0.9", "sdehead_t09_cfr"),
        ("scratch", "unpinned flow alone (scratch3 s42)", "scr3_cfr"),
        ("scratch_s7", "unpinned flow alone (scratch3 s7)", "scr3s7_cfr"),
        ("pinoff", "pin flow, plain noise", "pinoff_cfr"),
    ]),
    "left": ("Left gate", "left", 0, [
        ("dec", "U c executed, no flow", "dec_left"),
        ("pin", "pin (xswap s42)", "armxswap_left"),
        ("pin_s7", "pin (xswap s7)", "armxswaps7_left"),
        ("sde03", "SDEdit+head t0=0.3", "sdehead_t03_left"),
        ("scratch", "unpinned flow alone (scratch3 s42)", "armscr3_left"),
        ("scratch_s7", "unpinned flow alone (scratch3 s7)", "armscr3s7_left"),
        ("pinoff", "pin flow, plain noise", "pinoff_left"),
    ]),
    "right": ("Right gate", "right", 1, [
        ("pin", "pin (xswap s42)", "armxswap_right"),
        ("pin_s7", "pin (xswap s7)", "armxswaps7_right"),
        ("sde03", "SDEdit+head t0=0.3", "sdehead_t03_right"),
        ("scratch", "unpinned flow alone (scratch3 s42)", "armscr3_right"),
        ("scratch_s7", "unpinned flow alone (scratch3 s7)", "armscr3s7_right"),
        ("pinoff", "pin flow, plain noise", "pinoff_right"),
    ]),
    "cfl": ("Center from left", "center", 2, [
        ("pin", "pin (xswap s42)", "xswap_cfl"),
        ("pin_s7", "pin (xswap s7)", "xswaps7_cfl"),
        ("sde03", "SDEdit+head t0=0.3", "sdehead_t03_cfl"),
        ("scratch", "unpinned flow alone (scratch3 s42)", "scr3_cfl"),
        ("scratch_s7", "unpinned flow alone (scratch3 s7)", "scr3s7_cfl"),
        ("pinoff", "pin flow, plain noise", "pinoff_cfl"),
    ]),
    "cmpl": ("Compound L->C, hand-drawn sketch", "left_and_center", None, [
        ("dec", "sketch U c executed, no flow", "dec_cmpl"),
        ("src", "sketch z executed, no flow", "src_cmpl"),
        ("pin", "pin (xswap s42)", "xsk42_cmpl_denis"),
        ("pin_s7", "pin (xswap s7)", "xsks7_cmpl_denis"),
        ("sde03", "SDEdit(sketch) t0=0.3", "sde_cmpl_t03"),
        ("sde05", "SDEdit(sketch) t0=0.5", "sde_cmpl_t05"),
        ("sde07", "SDEdit(sketch) t0=0.7", "sde_cmpl_t07"),
        ("sde09", "SDEdit(sketch) t0=0.9", "sde_cmpl_t09"),
        ("scratch", "unpinned flow, sketch noise ignored", "scrsk_cmpl"),
    ]),
}


# ----------------------------------------------------------------------------- loading
def trial_files(prefix):
    fs = [f for f in glob.glob(f"{RUN}/traj_{prefix}_*.npy")
          if re.fullmatch(rf"traj_{re.escape(prefix)}_\d+\.npy", os.path.basename(f))]
    return sorted(fs, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))


def load_arm(prefix):
    return [np.load(f)[:, :3].astype(np.float64) for f in trial_files(prefix)]


def load_demos():
    """{'real': {task: [P]}, 'synth': {task: [P]}} from gate_nav3 (episodes 0-99 real, 100-299 synth)."""
    if not os.path.exists(DEMO_CACHE):
        import pyarrow.parquet as pq
        P, off, ep, task = [], [0], [], []
        for f in sorted(glob.glob(f"{LEROBOT}/data/chunk-000/episode_*.parquet")):
            t = pq.read_table(f, columns=["state", "episode_index", "task_index"]).to_pydict()
            st = np.array(t["state"], np.float64)[:, :3]
            P.append(st); off.append(off[-1] + len(st)); ep.append(int(t["episode_index"][0])); task.append(int(t["task_index"][0]))
        np.savez(DEMO_CACHE, pos=np.concatenate(P), off=np.array(off), ep=np.array(ep), task=np.array(task))
    z = np.load(DEMO_CACHE)
    out = {"real": {}, "synth": {}}
    for i in range(len(z["ep"])):
        P = z["pos"][z["off"][i]:z["off"][i + 1]]
        out["real" if z["ep"][i] < 100 else "synth"].setdefault(int(z["task"][i]), []).append(P)
    return out


# ----------------------------------------------------------------------------- kinematics
MIN_DWELL = 2.0     # s inside the goal box that counts as arrival (CFR routes brush the box for ~0.2 s en route)


def box_runs(P):
    """[(start, end)) index runs of consecutive steps inside the judge goal box."""
    inside = np.all(np.abs(P - GOAL_C) <= GOAL_H, axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], inside.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def split_segment(P):
    """(active, hover): active = start -> arrival + 1 s, arrival = first goal-box dwell of >= MIN_DWELL s
    (else the longest dwell); if the box is never entered, up to the last step with smoothed speed > 5 cm/s
    + 1 s. hover = everything after arrival."""
    runs = box_runs(P)
    long = [r for r in runs if (r[1] - r[0]) * DT >= MIN_DWELL]
    idx = [long[0][0]] if long else ([max(runs, key=lambda r: r[1] - r[0])[0]] if runs else [])
    if len(idx):
        k = idx[0]
    else:
        sp = np.linalg.norm(smooth(P, 1), axis=1)
        mv = np.where(sp > 0.05)[0]
        k = int(mv[-1]) if len(mv) else len(P) - 1
    end = min(len(P), k + int(1.0 / DT) + 1)
    return P[:end], (P[k:] if len(idx) else None)


def smooth(P, deriv):
    n = len(P); w = min(SG_WIN, n if n % 2 else n - 1)
    if w <= SG_POLY:
        return np.zeros_like(P) if deriv else P.copy()
    return savgol_filter(P, w, SG_POLY, deriv=deriv, delta=DT, axis=0)


def kin(P):
    v_raw = np.diff(P, axis=0) / DT
    a_raw = np.diff(v_raw, axis=0) / DT
    return v_raw, a_raw, smooth(P, 1), smooth(P, 2), smooth(P, 3)


def flatness(a, j):
    """Quadrotor differential flatness: thrust vector f = a + g e_z. Returns tilt (deg), thrust/weight,
    body-rate magnitude (deg/s) = |f x j| / |f|^2."""
    f = a + np.array([0, 0, G]); fn = np.linalg.norm(f, axis=1)
    tilt = np.degrees(np.arccos(np.clip(f[:, 2] / fn, -1, 1)))
    rate = np.degrees(np.linalg.norm(np.cross(f, j), axis=1) / fn ** 2)
    return tilt, fn / G, rate


def hf_fraction(v_raw, f_cut=1.0):
    if len(v_raw) < 16:
        return float("nan")
    x = v_raw - v_raw.mean(0)
    pw = np.abs(np.fft.rfft(x, axis=0)) ** 2
    fr = np.fft.rfftfreq(len(x), DT)
    tot = pw[1:].sum()
    return float(pw[fr >= f_cut].sum() / tot) if tot > 0 else float("nan")


def track_px4(P_sp):
    """PX4-shaped cascade tracking a 10 Hz setpoint stream (see module doc). Returns (realized, err)."""
    sub = 10; dt = DT / sub
    v_ff = np.diff(P_sp, axis=0) / DT
    p, v, a = P_sp[0].copy(), np.zeros(3), np.zeros(3)
    real = [p.copy()]
    kp = np.array([PX4["kp_xy"], PX4["kp_xy"], PX4["kp_z"]]); kv = np.array([PX4["kv_xy"], PX4["kv_xy"], PX4["kv_z"]])
    for k in range(len(P_sp) - 1):
        sp, vf = P_sp[k + 1], v_ff[k]
        for _ in range(sub):
            a_des = kv * (kp * (sp - p) + vf - v)
            h = np.hypot(a_des[0], a_des[1])
            if h > PX4["acc_h"]:
                a_des[:2] *= PX4["acc_h"] / h
            a_des[2] = np.clip(a_des[2], -PX4["acc_dn"], PX4["acc_up"])
            a = a + np.clip(a_des - a, -PX4["jerk"] * dt, PX4["jerk"] * dt)
            v = v + a * dt; p = p + v * dt
        real.append(p.copy())
    real = np.array(real)
    return real, np.linalg.norm(real - P_sp, axis=1)


def metrics(P, env=None, seams=True):
    """Scalar realism metrics of one active-segment trajectory. env = real-demo envelope (p99 |a|, |j|)."""
    v_raw, a_raw, v, a, j = kin(P)
    sp, an, jn, an_raw = (np.linalg.norm(x, axis=1) for x in (v, a, j, a_raw))
    T = (len(P) - 1) * DT; L = np.linalg.norm(v_raw, axis=1).sum() * DT
    tilt, tw, rate = flatness(a, j)
    ah = np.hypot(a[:, 0], a[:, 1])
    m = dict(
        T=T, path=L, chord=float(np.linalg.norm(P[-1] - P[0])),
        speed_p95=np.percentile(sp, 95), speed_mean=sp.mean(),
        acc_p95_raw=np.percentile(an_raw, 95), acc_p95=np.percentile(an, 95),
        jerk_p95_raw=np.percentile(np.linalg.norm(np.diff(a_raw, axis=0) / DT, axis=1), 95) if len(a_raw) > 1 else np.nan,
        jerk_p95=np.percentile(jn, 95),
        dimless_jerk=float(np.log10((jn ** 2).sum() * DT * T ** 5 / max(L, 1e-6) ** 2)),
        hf_frac=hf_fraction(v_raw),
        zero_acc_frac=float((an_raw < ZERO_ACC).mean()),
        tilt_p99=np.percentile(tilt, 99), tw_max=tw.max(), tw_min=tw.min(), rate_p99=np.percentile(rate, 99),
        px4_acc_viol=float((ah > PX4["acc_h"]).mean()), px4_jerk_viol=float((jn > PX4["jerk"]).mean()),
        px4_tilt_viol=float((tilt > PX4["tilt_deg"]).mean()),
    )
    if env is not None:
        m["env_acc_viol"] = float((an > env["acc_p99"]).mean()); m["env_jerk_viol"] = float((jn > env["jerk_p99"]).mean())
    if seams and len(a_raw) > CHUNK:
        b = np.arange(CHUNK - 1, len(a_raw), CHUNK)
        mask = np.zeros(len(a_raw), bool); mask[b] = True
        m["seam_acc"] = float(np.median(an_raw[mask])); m["interior_acc"] = float(np.median(an_raw[~mask]))
        m["seam_ratio"] = float(m["seam_acc"] / max(m["interior_acc"], 0.01))
    real, err = track_px4(P)
    m["track_rmse"] = float(np.sqrt((err ** 2).mean())); m["track_max"] = float(err.max())
    ck = np.arange(CHUNK, len(P), CHUNK)
    m["chunk_end_err"] = float(err[ck].mean()) if len(ck) else np.nan
    return {k: float(x) for k, x in m.items()}


def hover_metrics(H):
    if H is None or len(H) < SG_WIN + 2:
        return {}
    v = smooth(H, 1); sp = np.linalg.norm(v, axis=1)
    return dict(hover_T=(len(H) - 1) * DT, hover_speed_rms=float(np.sqrt((sp ** 2).mean())),
                hover_pos_std=float(np.linalg.norm(H.std(0))))


# ----------------------------------------------------------------------------- distinguishability
FEATS = ["speed_mean", "speed_std", "acc_mean", "acc_max", "jerk_mean", "jerk_max", "vz_mean", "zero_acc", "lat_acc"]


def window_feats(P, shape=False):
    """1 s window features. shape=True divides the magnitude features by the window's mean speed, so a
    flight that is merely slower/faster than the demos (the known behavior gap) is not counted as unrealistic."""
    v_raw, a_raw, v, a, j = kin(P)
    sp, an, jn = (np.linalg.norm(x, axis=1) for x in (v, a, j))
    zero = np.concatenate([[True], an_raw_lt(a_raw), [True]])
    lat = np.linalg.norm(np.cross(v, a), axis=1) / np.maximum(sp, 1e-3)
    rows = []
    for s in range(0, len(P) - WIN + 1, HOP):
        e = s + WIN
        ms = sp[s:e].mean(); k = 1.0 / max(ms, 0.05) if shape else 1.0
        rows.append([ms * k, sp[s:e].std() * k, np.log1p(an[s:e].mean() * k), np.log1p(an[s:e].max() * k),
                     np.log1p(jn[s:e].mean() * k), np.log1p(jn[s:e].max() * k), np.abs(v[s:e, 2]).mean() * k,
                     zero[s:e].mean(), np.log1p(lat[s:e].mean() * k)])
    F = np.array(rows)
    return F[:, 1:] if shape else F


def an_raw_lt(a_raw):
    return np.linalg.norm(a_raw, axis=1) < ZERO_ACC


def auc_score(y, s):
    r = np.argsort(np.argsort(s)) + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        for u in np.where(cnt > 1)[0]:
            ranks[inv == u] = ranks[inv == u].mean()
        r = ranks
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logreg_fit(X, y, lam=1e-2):
    w1 = 0.5 / max((y == 1).mean(), 1e-6); w0 = 0.5 / max((y == 0).mean(), 1e-6)
    sw = np.where(y == 1, w1, w0)
    Xb = np.hstack([X, np.ones((len(X), 1))])

    def f(w):
        z = Xb @ w; p = 1 / (1 + np.exp(-z))
        ll = -(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).mean() + lam * (w[:-1] ** 2).sum()
        g = (Xb * (sw * (p - y))[:, None]).mean(0); g[:-1] += 2 * lam * w[:-1]
        return ll, g
    return minimize(f, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B").x


def distinguishability(trajs_a, trajs_b, folds=5, seed=0, shape=False):
    """Grouped 5-fold CV AUC of windows from trajs_a (label 1) vs trajs_b (label 0)."""
    X, y, g = [], [], []
    for lab, trajs in ((1, trajs_a), (0, trajs_b)):
        for i, P in enumerate(trajs):
            F = window_feats(P, shape)
            if len(F) == 0:
                continue
            X.append(F); y.append(np.full(len(F), lab)); g.append(np.full(len(F), lab * 10000 + i))
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
    mu, sd = X.mean(0), X.std(0) + 1e-9; X = (X - mu) / sd
    rng = np.random.default_rng(seed)
    fold_of = {}
    for lab in (0, 1):
        gs = np.unique(g[y == lab]); rng.shuffle(gs)
        for k, gg in enumerate(gs):
            fold_of[gg] = k % folds
    fold = np.array([fold_of[gg] for gg in g])
    scores = np.zeros(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        w = logreg_fit(X[tr], y[tr])
        scores[te] = np.hstack([X[te], np.ones((te.sum(), 1))]) @ w
    return auc_score(y, scores)


# ----------------------------------------------------------------------------- driver
def summarize(trajs, env, seams=True):
    rows = []
    for P in trajs:
        A, H = split_segment(P)
        m = metrics(A, env, seams); m.update(hover_metrics(H)); rows.append(m)
    keys = sorted({k for r in rows for k in r})
    out = {"n": len(rows), "per_flight": rows}
    for k in keys:
        x = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
        if len(x):
            out[k] = dict(med=float(np.median(x)), q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)), n=int(len(x)))
    return out


def row(name, s, auc=True):
    g = lambda k: s.get(k, {}).get("med", float("nan"))
    print(f"  {name:22s} n={s['n']:3d} T {g('T'):5.1f}s path {g('path'):4.1f} v95 {g('speed_p95'):.2f} | acc95 raw {g('acc_p95_raw'):.2f} sm {g('acc_p95'):.2f} "
          f"jerk95 sm {g('jerk_p95'):4.1f} dj {g('dimless_jerk'):.2f} hf {g('hf_frac'):.2f} zero {g('zero_acc_frac'):.2f} seam {g('seam_acc'):.2f}/{g('interior_acc'):.2f} "
          f"tilt99 {g('tilt_p99'):4.1f} rate99 {g('rate_p99'):5.1f} envA {g('env_acc_viol'):.3f} envJ {g('env_jerk_viol'):.3f} "
          f"track {g('track_rmse'):.3f}/{g('track_max'):.3f} hover {g('hover_speed_rms'):.3f}"
          + (f" | auc real {s['auc_vs_real']:.2f} ({s['auc_vs_real_shape']:.2f}) synth {s['auc_vs_synth']:.2f} ({s['auc_vs_synth_shape']:.2f})"
             if auc and "auc_vs_real" in s else ""))


def envelope(real_trajs):
    A = [kin(split_segment(P)[0]) for P in real_trajs]
    an = np.concatenate([np.linalg.norm(a, axis=1) for _, _, _, a, _ in A])
    jn = np.concatenate([np.linalg.norm(j, axis=1) for _, _, _, _, j in A])
    return dict(acc_p99=float(np.percentile(an, 99)), jerk_p99=float(np.percentile(jn, 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default=f"{RD}/realism_results.json")
    ap.add_argument("--no-auc", action="store_true")
    args = ap.parse_args()
    demos = load_demos()
    real_all = [P for ps in demos["real"].values() for P in ps]
    synth_all = [P for ps in demos["synth"].values() for P in ps]
    env = envelope(real_all)
    print(f"real-demo envelope (smoothed p99): |a| {env['acc_p99']:.2f} m/s^2, |j| {env['jerk_p99']:.1f} m/s^3")
    res = {"dt": DT, "px4": PX4, "envelope": env, "sg": [SG_WIN, SG_POLY], "cells": {}, "demos": {}}
    active = lambda ps: [split_segment(P)[0] for P in ps]
    res["demos"]["real"] = summarize(real_all, env, seams=False)
    res["demos"]["synth"] = summarize(synth_all, env, seams=False)
    for t, ps in demos["synth"].items():
        res["demos"][f"synth_task{t}"] = summarize(ps, env, seams=False)
    for t, ps in demos["real"].items():
        res["demos"][f"real_task{t}"] = summarize(ps, env, seams=False)
    for k in ("real", "synth"):
        row(f"demo {k}", res["demos"][k])
    if not args.no_auc:
        rng = np.random.default_rng(0); idx = rng.permutation(len(real_all))
        half = [real_all[i] for i in idx[:50]], [real_all[i] for i in idx[50:]]
        res["demos"]["auc_real_vs_real"] = distinguishability(active(half[0]), active(half[1]))
        res["demos"]["auc_synth_vs_real"] = distinguishability(active(synth_all), active(real_all))
        res["demos"]["auc_real_vs_real_shape"] = distinguishability(active(half[0]), active(half[1]), shape=True)
        res["demos"]["auc_synth_vs_real_shape"] = distinguishability(active(synth_all), active(real_all), shape=True)
        print(f"calibration AUC: real-vs-real {res['demos']['auc_real_vs_real']:.3f} (shape {res['demos']['auc_real_vs_real_shape']:.3f}), "
              f"synth-vs-real {res['demos']['auc_synth_vs_real']:.3f} (shape {res['demos']['auc_synth_vs_real_shape']:.3f})")
    want = set(a for a in args.arms.split(",") if a)
    for cell in args.cells.split(","):
        title, scene, task, arms = CELLS[cell]
        res["cells"][cell] = {"title": title, "scene": scene, "task": task, "arms": {}}
        synth_cell = demos["synth"].get(task, synth_all) if task is not None else synth_all
        for key, label, tag in arms:
            if want and key not in want:
                continue
            trajs = load_arm(tag)
            if not trajs:
                print(f"  [{cell}] {key}: no trajectories for {tag}"); continue
            s = summarize(trajs, env); s.update(label=label, tag=tag)
            if not args.no_auc:
                s["auc_vs_real"] = distinguishability(active(trajs), active(real_all))
                s["auc_vs_synth"] = distinguishability(active(trajs), active(synth_cell))
                s["auc_vs_real_shape"] = distinguishability(active(trajs), active(real_all), shape=True)
                s["auc_vs_synth_shape"] = distinguishability(active(trajs), active(synth_cell), shape=True)
            res["cells"][cell]["arms"][key] = s
            row(f"[{cell}] {key}", s)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
