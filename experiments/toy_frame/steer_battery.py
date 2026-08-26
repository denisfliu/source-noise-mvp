"""Steerability + interpretability battery: WHERE should the steering command
enter a flow-matching action model?

Project core (Denis, 2026-07-20): steer actions by intelligently constructing
the source noise; steerability + interpretability are the goal, x-embodiment a
bonus. Compares four injection points for the SAME scalar-complex steering
command C = the action's lateral bend (canonical lateral FFT bin-1, [Re,Im]):

  plain      no command; source N(0,I)                         (task baseline)
  condition  C -> velocity-net input; source N(0,I)            (soft channel)
  pin        C overwrites lateral bin-1 of a FIXED N(0,I) src  (basis paper; the
             current method — exact steering but pays a success tax)
  csfm       C -> LEARNED condition-dependent source mu_phi(C,obs), var-only
             reg + directional align (arXiv:2602.05951, Eqs 7-11) -- steer by
             positioning the source instead of overwriting it

Hypothesis (the synthesis): CSFM matches the pin's steerability (follow) while
removing the success tax (it minimizes the intrinsic-variance term the pin
inflates) and giving cleaner disentangled steering. Three metrics, one per goal:
  steerability = follow error |C_produced - C_commanded|
  tax          = task success when commanding the scene's OWN bend, vs plain
  interpretability = sweep C -> monotone bend response + endpoint invariance
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
from dataset import (ACT_SCALE, H, make_dataset, success,  # noqa: E402
                     to_canonical)
from pin import extract_mags, extract_phases, pin_noise    # noqa: E402

HID = 128
ITERS = 12000
BATCH = 256
EULER = 20
OBS = 5
LAT_PIN = [{"axis": (0.0, 1.0), "omega": 1, "mode": "mod2pi", "mag": True}]
LAM_VAR = 0.1
LAM_ALIGN = 0.1
OUT = os.path.join(HERE, "results", "steer_battery")


# ------------------------- steering coordinate ------------------------------

def bend_coeff(chunk_canon):
    """Lateral FFT bin-1 as [Re,Im] (..., 2) — the interpretable steer coord."""
    z = chunk_canon[..., 1]
    c1 = np.fft.rfft(z, axis=-1)[..., 1]
    return np.stack([c1.real, c1.imag], axis=-1)


def coeff_to_pin_targets(C):
    """[Re,Im] -> (phase (n,1), mag (n,1)) for the lateral bin-1 pin."""
    c = C[:, 0] + 1j * C[:, 1]
    return np.angle(c)[:, None], np.abs(c)[:, None]


# ------------------------------ networks ------------------------------------

def mlp_init(dims, rng):
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def mlp(params, x, out_act=None):
    h = x
    for w, b in params[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = params[-1]
    o = h @ w + b
    return out_act(o) if out_act else o


def vfield(vparams, xt, t, obs, cmd):
    parts = [xt.reshape(xt.shape[0], -1), t.reshape(-1, 1), obs]
    if cmd is not None:
        parts.append(cmd)
    return mlp(vparams, anp.concatenate(parts, axis=1)).reshape(xt.shape[0], H, 2)


def source_gen(gparams, C, obs):
    """CSFM source: (C,obs) -> mu (n,H,2), log_sigma (n,H,2)."""
    out = mlp(gparams, anp.concatenate([C, obs], axis=1))
    mu, logsig = out[:, :H * 2], out[:, H * 2:]
    return mu.reshape(-1, H, 2), logsig.reshape(-1, H, 2)


def v_in_dim(arm):
    return H * 2 + 1 + OBS + (2 if arm == "condition" else 0)


def init_params(arm, rng):
    p = {"v": mlp_init([v_in_dim(arm), HID, HID, HID, H * 2], rng)}
    if arm == "csfm":
        p["g"] = mlp_init([2 + OBS, 64, 64, H * 2 * 2], rng)
    return p


# ------------------------------ training ------------------------------------

def make_loss(arm, obs_all, chunks_all, angles_all):
    canon_all = to_canonical(chunks_all, angles_all)
    C_all = bend_coeff(canon_all)                          # (n,2) each action's bend
    n = obs_all.shape[0]

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, n, size=BATCH)
        a0 = chunks_all[idx]
        obs, ang, C = obs_all[idx], angles_all[idx], C_all[idx]
        eps = rng.normal(size=(BATCH, H, 2))
        cmd = None
        reg = 0.0
        if arm == "pin":
            eps_c = to_canonical(eps, ang)
            ph, mg = coeff_to_pin_targets(C)
            eps_c = pin_noise(eps_c, LAT_PIN, ph, mag_targets=mg)
            S = to_canonical(eps_c, -ang)
        elif arm == "csfm":
            mu, logsig = source_gen(params["g"], C, obs)
            sig = anp.exp(logsig)
            S = mu + sig * eps
            var = sig ** 2
            l_var = 0.5 * anp.mean(var - 1.0 - 2.0 * logsig)          # KL, mean-free
            dot = anp.sum(S * a0, axis=(1, 2))
            cos = dot / (anp.sqrt(anp.sum(S ** 2, (1, 2))) *
                         anp.sqrt(anp.sum(a0 ** 2, (1, 2))) + 1e-8)
            l_align = anp.mean(1.0 - cos)
            reg = LAM_VAR * l_var + LAM_ALIGN * l_align
        else:                                              # plain / condition
            S = eps
            if arm == "condition":
                cmd = C
        t = rng.uniform(0, 1, size=BATCH)
        xt = t[:, None, None] * S + (1 - t[:, None, None]) * a0
        v = S - a0
        return anp.mean((vfield(params["v"], xt, t, obs, cmd) - v) ** 2) + reg

    return loss


def train(arm, obs, chunks, angles, seed):
    params = init_params(arm, np.random.default_rng(seed))
    loss = make_loss(arm, obs, chunks, angles)
    return adam(grad(loss), params, num_iters=ITERS, step_size=1e-3)


# ------------------------------ rollout -------------------------------------

def integrate(arm, params, obs, angles, C_cmd, rng):
    """Sample the (arm-specific) source given command C_cmd, integrate the ODE
    t:1->0, return the produced action chunk (n,H,2)."""
    n = obs.shape[0]
    eps = rng.normal(size=(n, H, 2))
    cmd = None
    if arm == "pin":
        eps_c = to_canonical(eps, angles)
        ph, mg = coeff_to_pin_targets(C_cmd)
        eps_c = pin_noise(eps_c, LAT_PIN, ph, mag_targets=mg, orient_from_noise=False)
        S = to_canonical(eps_c, -angles)
    elif arm == "csfm":
        mu, logsig = source_gen(params["g"], C_cmd, obs)
        S = np.asarray(mu) + np.exp(np.asarray(logsig)) * eps
    else:
        S = eps
        if arm == "condition":
            cmd = C_cmd
    x = S
    dt = 1.0 / EULER
    for k in range(EULER):
        t = np.full(n, 1.0 - k * dt)
        x = x - dt * np.asarray(vfield(params["v"], x, t, obs, cmd))
    return x


def produced_bend(chunk, angles):
    return bend_coeff(to_canonical(chunk, angles))


# ------------------------------- battery ------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(7)
    scenes, obs, chunks, angles = make_dataset(200, 8, rng)
    fo = np.repeat(obs, 8, axis=0)
    fc = chunks.reshape(-1, H, 2)
    fa = np.repeat(angles, 8)

    he_scenes, he_obs, he_chunks, he_ang = make_dataset(100, 8, np.random.default_rng(7777))
    # natural bend per held scene = mean bend over its demos (the scene's structure)
    he_canon = to_canonical(he_chunks, he_ang[:, None])                 # (M,8,H,2)
    C_nat = bend_coeff(he_canon).mean(axis=1)                           # (M,2)
    typ = float(np.sqrt((C_nat ** 2).sum(1)).mean())                    # typical |C|
    print(f"typical |C_nat| = {typ:.3f}", flush=True)

    arms = ["plain", "condition", "pin", "csfm"]
    result = {"typical_C": round(typ, 4), "arms": {}}
    for arm in arms:
        params = train(arm, fo, fc, fa, seed=0)
        er = np.random.default_rng(100)

        def follow_and_success(C_cmd):
            ch = integrate(arm, params, he_obs, he_ang, C_cmd, er)
            C_prod = produced_bend(ch, he_ang)
            ferr = float(np.sqrt(((C_prod - C_cmd) ** 2).sum(1)).mean())
            succ = float(np.mean([success(he_scenes[i], ch[i]) for i in range(len(he_scenes))]))
            return ferr, succ, C_prod

        zeroC = np.zeros_like(C_nat)
        # plain: no command -> use zeros (source N(0,I)); its success is the baseline
        f_nat, s_nat, _ = follow_and_success(C_nat if arm != "plain" else zeroC)
        f_con, s_con, _ = follow_and_success(-C_nat if arm != "plain" else zeroC)

        # interpretability sweep: command C = alpha * unit_dir, alpha in grid,
        # measure produced bend along that axis + endpoint (progress bin-0) drift
        udir = np.array([0.0, 1.0])                                     # steer Im (side)
        alphas = np.linspace(-2 * typ, 2 * typ, 9)
        prod_along, end_drift = [], []
        for al in alphas:
            Ccmd = np.tile(al * udir, (len(he_scenes), 1))
            ch = integrate(arm, params, he_obs, he_ang, Ccmd, er)
            cc = to_canonical(ch, he_ang)
            prod_along.append(float(bend_coeff(cc)[:, 1].mean()))       # produced Im-bend
            prog0 = np.abs(np.fft.rfft(cc[..., 0], axis=-1)[:, 0])      # endpoint (progress bin0)
            end_drift.append(float(prog0.mean()))
        # steerability = slope of produced-vs-commanded (1=perfect), R^2
        A = np.array(alphas); P = np.array(prod_along)
        slope = float(np.polyfit(A, P, 1)[0])
        r2 = float(1 - np.sum((P - np.polyval(np.polyfit(A, P, 1), A)) ** 2) /
                   (np.sum((P - P.mean()) ** 2) + 1e-9))
        end_var = float(np.std(end_drift) / (np.mean(end_drift) + 1e-9))  # endpoint stability

        result["arms"][arm] = {
            "follow_err_natural": round(f_nat, 4),
            "follow_err_contra": round(f_con, 4),
            "success_natural": round(s_nat, 4),
            "success_contra": round(s_con, 4),
            "steer_slope": round(slope, 4), "steer_r2": round(r2, 4),
            "endpoint_cv_under_sweep": round(end_var, 4),
            "sweep_alphas": [round(a, 3) for a in alphas],
            "sweep_produced_bend": [round(p, 4) for p in prod_along],
        }
        print(f"[{time.time()-t0:.0f}s] {arm}: follow(nat/con)="
              f"{f_nat:.3f}/{f_con:.3f} succ(nat/con)={s_nat:.3f}/{s_con:.3f} "
              f"steer_slope={slope:.2f} r2={r2:.2f} endCV={end_var:.3f}", flush=True)

    base = result["arms"]["plain"]["success_natural"]
    for arm in arms:
        result["arms"][arm]["tax_vs_plain"] = round(base - result["arms"][arm]["success_natural"], 4)
    result["verdict"] = {
        "baseline_success": base,
        "note": "steerability = follow_err_contra (lower=obeys contradictory "
                "command) + steer_slope~1/r2~1; tax = success drop vs plain when "
                "commanding the scene's own bend; interpretability = steer_slope "
                "near 1, high r2, low endpoint_cv (one knob moves bend only). "
                "Hypothesis: csfm ~ pin on follow, but lower tax + cleaner sweep."}
    json.dump(result, open(os.path.join(OUT, "battery.json"), "w"), indent=2)
    print("VERDICT:", json.dumps(result["verdict"], indent=1))
    print(f"STEER_DONE=ok in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
