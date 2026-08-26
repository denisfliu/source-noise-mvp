"""Multi-axis interpretable steering — the differentiator between constructing
the source and plain conditioning, and the presentable demo.

Two independent steering knobs, both lateral-shape Fourier modes of the action:
  axis 1 = lateral FFT bin-1  (primary bend: curve left / right, how much)
  axis 2 = lateral FFT bin-2  (S-curve / double-bend)
Command C = [Re1, Im1, Re2, Im2] (4-D). We ask whether a mechanism gives
DISENTANGLED, COMPOSABLE control:
  - disentanglement Jacobian J[i,j] = d(produced coord i)/d(commanded coord j):
    identity = each knob moves only its own attribute (off-diagonal = leakage).
  - composition additivity: does commanding (A+B) produce action(A)+action(B)-base?
    (linear, predictable control in command space)
Arms: csfm (learned condition-dependent source, Eqs 7-11), condition (velocity
input), pin (overwrite lateral bins 1&2 of a fixed Gaussian). Also dumps a grid
of produced trajectories over (Im1, Im2) for visualization.
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
from dataset import ACT_SCALE, H, make_dataset, success, to_canonical  # noqa: E402
from pin import pin_noise                                              # noqa: E402

HID = 128
ITERS = 12000
BATCH = 256
EULER = 20
OBS = 5
CMD = 4                                    # [Re1,Im1,Re2,Im2] lateral bins 1,2
BINS = [1, 2]
PINSPEC = [{"axis": (0.0, 1.0), "omega": b, "mode": "mod2pi", "mag": True} for b in BINS]
LAM_VAR, LAM_ALIGN = 0.1, 0.1
OUT = os.path.join(HERE, "results", "steer_multiaxis")


def bend4(chunk_canon):
    """Lateral bins 1,2 as [Re1,Im1,Re2,Im2] (...,4)."""
    z = chunk_canon[..., 1]
    spec = np.fft.rfft(z, axis=-1)
    return np.stack([spec[..., 1].real, spec[..., 1].imag,
                     spec[..., 2].real, spec[..., 2].imag], axis=-1)


def cmd_to_pin(C):
    """4-D command -> (phase (n,2), mag (n,2)) for the two lateral bin pins."""
    c1 = C[:, 0] + 1j * C[:, 1]
    c2 = C[:, 2] + 1j * C[:, 3]
    ph = np.stack([np.angle(c1), np.angle(c2)], axis=1)
    mg = np.stack([np.abs(c1), np.abs(c2)], axis=1)
    return ph, mg


def mlp_init(dims, rng):
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def mlp(params, x):
    h = x
    for w, b in params[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = params[-1]
    return h @ w + b


def vfield(vp, xt, t, obs, cmd):
    parts = [xt.reshape(xt.shape[0], -1), t.reshape(-1, 1), obs]
    if cmd is not None:
        parts.append(cmd)
    return mlp(vp, anp.concatenate(parts, axis=1)).reshape(xt.shape[0], H, 2)


def source_gen(gp, C, obs):
    out = mlp(gp, anp.concatenate([C, obs], axis=1))
    return out[:, :H * 2].reshape(-1, H, 2), out[:, H * 2:].reshape(-1, H, 2)


def init_params(arm, rng):
    vin = H * 2 + 1 + OBS + (CMD if arm == "condition" else 0)
    p = {"v": mlp_init([vin, HID, HID, HID, H * 2], rng)}
    if arm == "csfm":
        p["g"] = mlp_init([CMD + OBS, 64, 64, H * 2 * 2], rng)
    return p


def make_loss(arm, obs_all, chunks_all, angles_all):
    canon = to_canonical(chunks_all, angles_all)
    C_all = bend4(canon)
    n = obs_all.shape[0]

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, n, size=BATCH)
        a0, obs, ang, C = chunks_all[idx], obs_all[idx], angles_all[idx], C_all[idx]
        eps = rng.normal(size=(BATCH, H, 2))
        cmd, reg = None, 0.0
        if arm == "pin":
            ec = to_canonical(eps, ang)
            ph, mg = cmd_to_pin(C)
            ec = pin_noise(ec, PINSPEC, ph, mag_targets=mg)
            S = to_canonical(ec, -ang)
        elif arm == "csfm":
            mu, ls = source_gen(params["g"], C, obs)
            sig = anp.exp(ls)
            S = mu + sig * eps
            l_var = 0.5 * anp.mean(sig ** 2 - 1.0 - 2.0 * ls)
            cos = anp.sum(S * a0, (1, 2)) / (anp.sqrt(anp.sum(S ** 2, (1, 2))) *
                                             anp.sqrt(anp.sum(a0 ** 2, (1, 2))) + 1e-8)
            reg = LAM_VAR * l_var + LAM_ALIGN * anp.mean(1.0 - cos)
        else:
            S = eps
            if arm == "condition":
                cmd = C
        t = rng.uniform(0, 1, size=BATCH)
        xt = t[:, None, None] * S + (1 - t[:, None, None]) * a0
        return anp.mean((vfield(params["v"], xt, t, obs, cmd) - (S - a0)) ** 2) + reg

    return loss


def train(arm, obs, chunks, angles, seed=0):
    return adam(grad(make_loss(arm, obs, chunks, angles)),
                init_params(arm, np.random.default_rng(seed)),
                num_iters=ITERS, step_size=1e-3)


def integrate(arm, params, obs, angles, C, rng):
    n = obs.shape[0]
    eps = rng.normal(size=(n, H, 2))
    cmd = None
    if arm == "pin":
        ec = to_canonical(eps, angles)
        ph, mg = cmd_to_pin(C)
        ec = pin_noise(ec, PINSPEC, ph, mag_targets=mg, orient_from_noise=False)
        S = to_canonical(ec, -angles)
    elif arm == "csfm":
        mu, ls = source_gen(params["g"], C, obs)
        S = np.asarray(mu) + np.exp(np.asarray(ls)) * eps
    else:
        S = eps
        if arm == "condition":
            cmd = C
    x = S
    for k in range(EULER):
        t = np.full(n, 1.0 - k / EULER)
        x = x - (1.0 / EULER) * np.asarray(vfield(params["v"], x, t, obs, cmd))
    return x


def produced4(chunk, angles):
    return bend4(to_canonical(chunk, angles))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    sc, obs, ch, ang = make_dataset(200, 8, np.random.default_rng(7))
    fo, fc, fa = np.repeat(obs, 8, 0), ch.reshape(-1, H, 2), np.repeat(ang, 8)
    he_sc, he_o, he_ch, he_a = make_dataset(80, 8, np.random.default_rng(7777))
    C_nat = bend4(to_canonical(he_ch, he_a[:, None])).mean(1)          # (M,4)
    scale = np.abs(C_nat).mean(0) + 1e-6                                # per-coord scale
    print(f"per-coord |C_nat| scale = {np.round(scale,3)}", flush=True)

    result = {"scale": scale.tolist(), "arms": {}}
    traj = {"scene_obs": he_o[0].tolist(), "angle": float(he_a[0]), "grids": {}}
    for arm in ["condition", "pin", "csfm"]:
        params = train(arm, fo, fc, fa)
        er = np.random.default_rng(100)

        # disentanglement Jacobian: perturb each command coord, measure produced 4-D
        base = C_nat.copy()
        p0 = produced4(integrate(arm, params, he_o, he_a, base, er), he_a)
        J = np.zeros((4, 4))
        for j in range(4):
            d = np.zeros(4); d[j] = scale[j]                # +1 scale-unit on coord j
            pj = produced4(integrate(arm, params, he_o, he_a, base + d, er), he_a)
            J[:, j] = (pj - p0).mean(0) / scale[j]          # dproduced_i / dcommand_j
        diag = float(np.mean(np.abs(np.diag(J))))
        off = float((np.abs(J).sum() - np.abs(np.diag(J)).sum()) / 12.0)
        leakage = round(off / (diag + 1e-9), 4)

        # composition additivity: A (coord1), B (coord3), A+B
        A = base.copy(); A[:, 1] += 1.5 * scale[1]
        B = base.copy(); B[:, 3] += 1.5 * scale[3]
        AB = base.copy(); AB[:, 1] += 1.5 * scale[1]; AB[:, 3] += 1.5 * scale[3]
        pA = produced4(integrate(arm, params, he_o, he_a, A, er), he_a)
        pB = produced4(integrate(arm, params, he_o, he_a, B, er), he_a)
        pAB = produced4(integrate(arm, params, he_o, he_a, AB, er), he_a)
        add_err = float(np.abs(pAB - (pA + pB - p0)).mean() / np.abs(scale).mean())

        # follow (full 4-D) + success at natural command
        follow = float(np.sqrt(((p0 - base) ** 2).sum(1)).mean())
        succ = float(np.mean([success(he_sc[i], integrate(arm, params, he_o, he_a, base, er)[i])
                              for i in range(len(he_sc))]))

        # trajectory grid over (Im1, Im2) at scene 0, for the visual demo
        s0o = np.tile(he_o[0], (1, 1)); s0a = he_a[:1]
        vals = np.linspace(-2, 2, 5)
        grid = []
        for a1 in vals:
            row = []
            for a2 in vals:
                Cg = base[:1].copy(); Cg[0, 1] = a1 * scale[1]; Cg[0, 3] = a2 * scale[3]
                chk = integrate(arm, params, s0o, s0a, Cg, np.random.default_rng(0))
                world = to_canonical(chk, -s0a)
                pos = np.concatenate([[[0.0, 0.0]], np.cumsum(world[0] / ACT_SCALE, 0)])
                row.append(pos.tolist())
            grid.append(row)
        traj["grids"][arm] = grid

        result["arms"][arm] = {
            "jacobian": np.round(J, 3).tolist(), "diag_mean": round(diag, 3),
            "offdiag_mean": round(off, 3), "leakage_ratio": leakage,
            "composition_add_err": round(add_err, 4),
            "follow_err": round(follow, 4), "success_natural": round(succ, 4)}
        print(f"[{time.time()-t0:.0f}s] {arm}: diag={diag:.2f} offdiag={off:.3f} "
              f"leak={leakage:.3f} add_err={add_err:.3f} follow={follow:.3f} "
              f"succ={succ:.3f}", flush=True)

    result["traj_vals"] = np.linspace(-2, 2, 5).tolist()
    json.dump(result, open(os.path.join(OUT, "battery.json"), "w"), indent=2)
    json.dump(traj, open(os.path.join(OUT, "trajectories.json"), "w"))
    print("VERDICT:", json.dumps({a: {"leakage": m["leakage_ratio"],
          "add_err": m["composition_add_err"], "follow": m["follow_err"],
          "succ": m["success_natural"]} for a, m in result["arms"].items()}, indent=1))
    print(f"MULTIAXIS_DONE=ok in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
