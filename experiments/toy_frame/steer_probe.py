"""Deep steering probe: fair pin (phase vs magnitude), more axes (lateral bend +
S-curve + progress timing), and CSFM source-latent structure.

Command = 3 complex Fourier coords of the action (6-D):
  ('lat',1) lateral bend   ('lat',2) lateral S-curve   ('prog',1) progress timing
Arms: pin (overwrite fixed-Gaussian source), condition (velocity input), csfm
(learned condition-dependent source, Eqs 7-11).

A. FAIR PIN — decompose follow into PHASE (side/timing) vs MAGNITUDE (amount).
   Hypothesis: the source pin can carry PHASE (preserved along the flow) but not
   MAGNITUDE (the velocity field rescales it) -> explains the under-response and
   fairly characterizes the pin as a phase-steering channel; CSFM carries both.
B. MORE AXES — 6x6 disentanglement Jacobian; separately report lateral<->progress
   cross-leak ("where" vs "when" independence).
C. CSFM LATENT STRUCTURE — Jacobian dmu_phi/dC (40x6): SVD (effective rank),
   pairwise orthogonality of the 6 command->source directions, and cosine
   alignment of each with its ANALYTIC Fourier basis vector (does commanding
   coord j move the source mean along the matching Fourier mode?).
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

HID, ITERS, BATCH, EULER, OBS = 128, 12000, 256, 20, 5
AXES = {"lat": (0.0, 1.0), "prog": (1.0, 0.0)}
SPEC = [("lat", 1), ("lat", 2), ("prog", 1)]
CMD = 2 * len(SPEC)
PINSPEC = [{"axis": AXES[a], "omega": o, "mode": "mod2pi", "mag": True} for a, o in SPEC]
LAM_VAR, LAM_ALIGN = 0.1, 0.1
OUT = os.path.join(HERE, "results", "steer_probe")


def coeffs(chunk_canon):
    out = []
    for a, o in SPEC:
        z = chunk_canon @ np.asarray(AXES[a])
        c = np.fft.rfft(z, axis=-1)[..., o]
        out += [c.real, c.imag]
    return np.stack(out, axis=-1)


def cmd_to_pin(C):
    ph, mg = [], []
    for k in range(len(SPEC)):
        c = C[:, 2 * k] + 1j * C[:, 2 * k + 1]
        ph.append(np.angle(c)); mg.append(np.abs(c))
    return np.stack(ph, 1), np.stack(mg, 1)


def fourier_dir(name, om, part):
    """Analytic chunk-space (H*2,) unit direction for one Fourier coord."""
    u = np.zeros(H // 2 + 1, dtype=complex)
    u[om] = 1.0 if part == "re" else 1j
    time = np.fft.irfft(u, n=H)
    D = np.outer(time, np.asarray(AXES[name]))
    return (D / (np.linalg.norm(D) + 1e-9)).reshape(-1)


def mlp_init(dims, rng):
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def mlp(p, x):
    h = x
    for w, b in p[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = p[-1]
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
    C_all = coeffs(canon)
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
            sig = anp.exp(ls); S = mu + sig * eps
            lv = 0.5 * anp.mean(sig ** 2 - 1.0 - 2.0 * ls)
            cos = anp.sum(S * a0, (1, 2)) / (anp.sqrt(anp.sum(S ** 2, (1, 2))) *
                                             anp.sqrt(anp.sum(a0 ** 2, (1, 2))) + 1e-8)
            reg = LAM_VAR * lv + LAM_ALIGN * anp.mean(1.0 - cos)
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
                init_params(arm, np.random.default_rng(seed)), num_iters=ITERS, step_size=1e-3)


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


def prod_coeffs(chunk, angles):
    return coeffs(to_canonical(chunk, angles))


def circdist(a, b):
    d = (a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    sc, obs, ch, ang = make_dataset(200, 8, np.random.default_rng(7))
    fo, fc, fa = np.repeat(obs, 8, 0), ch.reshape(-1, H, 2), np.repeat(ang, 8)
    he_sc, he_o, he_ch, he_a = make_dataset(80, 8, np.random.default_rng(7777))
    C_nat = coeffs(to_canonical(he_ch, he_a[:, None])).mean(1)          # (M,6)
    scale = np.abs(C_nat).mean(0) + 1e-6
    result = {"spec": [f"{a}{o}" for a, o in SPEC], "scale": scale.round(3).tolist(), "arms": {}}

    for arm in ["condition", "pin", "csfm"]:
        params = train(arm, fo, fc, fa)
        er = np.random.default_rng(100)
        base = C_nat.copy()
        p0 = prod_coeffs(integrate(arm, params, he_o, he_a, base, er), he_a)

        # A/B: 6x6 disentanglement Jacobian
        J = np.zeros((CMD, CMD))
        for j in range(CMD):
            d = np.zeros(CMD); d[j] = scale[j]
            pj = prod_coeffs(integrate(arm, params, he_o, he_a, base + d, er), he_a)
            J[:, j] = (pj - p0).mean(0) / scale[j]
        diag = np.abs(np.diag(J))
        off = (np.abs(J).sum() - diag.sum()) / (CMD * CMD - CMD)
        # lateral (coords 0-3) vs progress (coords 4-5) cross-leak
        cross = (np.abs(J[0:4, 4:6]).mean() + np.abs(J[4:6, 0:4]).mean()) / 2

        # A: phase vs magnitude follow on lateral bin-1 (coords 0,1)
        mag0 = scale[0]  # rough magnitude scale for bin-1
        ph_err, mg_err = [], []
        for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            Cc = base.copy(); Cc[:, 0] = mag0 * np.cos(th); Cc[:, 1] = mag0 * np.sin(th)
            pc = prod_coeffs(integrate(arm, params, he_o, he_a, Cc, er), he_a)
            ph_prod = np.arctan2(pc[:, 1], pc[:, 0])
            ph_err.append(float(circdist(ph_prod, th).mean()))
        for m in [0.5, 1.0, 1.5, 2.0]:
            Cc = base.copy(); Cc[:, 0] = m * mag0; Cc[:, 1] = 0.0
            pc = prod_coeffs(integrate(arm, params, he_o, he_a, Cc, er), he_a)
            m_prod = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2).mean()
            mg_err.append(round(float(m_prod / (m * mag0)), 3))          # ratio, 1=perfect
        phase_follow = round(float(np.degrees(np.mean(ph_err))), 1)      # deg error
        mag_slope = round(float(np.polyfit([0.5, 1.0, 1.5, 2.0],
                          [r for r in mg_err], 1)[0]) * 1.0, 3)          # d(ratio)/d(mult)~0 good

        follow = float(np.sqrt(((p0 - base) ** 2).sum(1)).mean())
        succ = float(np.mean([success(he_sc[i], integrate(arm, params, he_o, he_a, base, er)[i])
                              for i in range(len(he_sc))]))

        entry = {"jacobian": J.round(3).tolist(), "diag_mean": round(float(diag.mean()), 3),
                 "offdiag_mean": round(float(off), 3),
                 "leakage_ratio": round(float(off / (diag.mean() + 1e-9)), 3),
                 "lat_prog_crossleak": round(float(cross), 3),
                 "phase_follow_deg": phase_follow, "mag_ratio_by_mult": mg_err,
                 "follow_err": round(follow, 3), "success": round(succ, 3)}

        # C: CSFM source-latent structure
        if arm == "csfm":
            mu0, _ = source_gen(params["g"], base, he_o)
            mu0 = np.asarray(mu0).reshape(len(he_o), -1)
            Jmu = np.zeros((H * 2, CMD))
            for j in range(CMD):
                d = np.zeros(CMD); d[j] = scale[j]
                muj, _ = source_gen(params["g"], base + d, he_o)
                Jmu[:, j] = (np.asarray(muj).reshape(len(he_o), -1) - mu0).mean(0) / scale[j]
            sv = np.linalg.svd(Jmu, compute_uv=False)
            cols = Jmu / (np.linalg.norm(Jmu, axis=0, keepdims=True) + 1e-9)
            gram = cols.T @ cols
            orth = float((np.abs(gram).sum() - CMD) / (CMD * CMD - CMD))   # mean |cos| off-diag
            parts = [("re", "im")[i % 2] for i in range(CMD)]
            names = [(SPEC[i // 2][0], SPEC[i // 2][1], "re" if i % 2 == 0 else "im")
                     for i in range(CMD)]
            align = [round(float(abs(np.dot(fourier_dir(*names[j]), cols[:, j]))), 3)
                     for j in range(CMD)]
            entry["latent"] = {"singular_values": sv.round(3).tolist(),
                               "eff_rank_0p1": int((sv > 0.1 * sv[0]).sum()),
                               "mean_offdiag_cos": round(orth, 3),
                               "fourier_alignment": align,
                               "mean_alignment": round(float(np.mean(align)), 3)}
        result["arms"][arm] = entry
        msg = (f"[{time.time()-t0:.0f}s] {arm}: diag={entry['diag_mean']} "
               f"leak={entry['leakage_ratio']} crossleak={entry['lat_prog_crossleak']} "
               f"phase_follow={phase_follow}deg magratio={mg_err} succ={entry['success']}")
        if arm == "csfm":
            msg += (f" | latent: eff_rank={entry['latent']['eff_rank_0p1']} "
                    f"offdiag_cos={entry['latent']['mean_offdiag_cos']} "
                    f"fourier_align={entry['latent']['mean_alignment']}")
        print(msg, flush=True)

    json.dump(result, open(os.path.join(OUT, "battery.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
