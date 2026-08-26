"""Temporal-phase pin construction (paper's eq. 6, temporal version) +
phase extraction. Per docs/learned_frame_toy_plan.md Step 2 with the
2026-07-05 amendments (energy floor upstream; confidence gating at call site).

A pin spec is a list of dicts:
    {"axis": 2-vector (unit), "omega": bin index, "mode": "mod2pi"|"modpi"}
Targets are per-sample phases (radians), one per spec entry. A negative
confidence/None target means "skip this pin for this sample" (the
confidence-gated path).

Training-side: targets come oriented from the sample's own action.
Inference-side (mod-pi pins): pass orient_from_noise=True so the noise draw
chooses the side (nearest of {phi, phi+pi}) — style stays with the draw.
"""

import numpy as np


def extract_phases(chunks, pins):
    """chunks (..., H, 2) -> phases (..., n_pins)."""
    out = []
    for p in pins:
        z = chunks @ np.asarray(p["axis"], dtype=float)
        spec = np.fft.rfft(z, axis=-1)
        out.append(np.angle(spec[..., p["omega"]]))
    return np.stack(out, axis=-1)


def pin_noise(eps, pins, targets, mag_targets=None, orient_from_noise=False):
    """eps (..., H, 2); targets (..., n_pins) phases, NaN = skip that pin.
    mag_targets (..., n_pins) optional magnitudes for pins with "mag": True
    (NaN or pin without the flag = keep the noise's magnitude — phase-only).

    Returns eps_tilde with the pinned coordinates' phase (and, where
    requested, magnitude) overwritten. Complement untouched. NOTE: magnitude
    overwrite breaks the Rayleigh marginal of that coordinate (exact-mode
    trade-off, same as the displacement pin's).
    """
    eps = np.asarray(eps, dtype=float)
    targets = np.asarray(targets, dtype=float)
    out = eps.copy()
    # group pins by axis so each axis's field is edited in one FFT pass
    axes = {}
    for k, p in enumerate(pins):
        axes.setdefault(tuple(np.round(np.asarray(p["axis"], float), 12)), []).append(k)
    for axis_key, idxs in axes.items():
        u = np.asarray(axis_key, dtype=float)
        z = out @ u                                   # (..., H)
        spec = np.fft.rfft(z, axis=-1)
        spec_new = spec.copy()
        for k in idxs:
            om = pins[k]["omega"]
            phi = targets[..., k]
            cur = spec[..., om]
            mag = np.abs(cur)
            if pins[k].get("mag") and mag_targets is not None:
                mt = mag_targets[..., k]
                mag = np.where(np.isnan(mt), mag, mt)
            if pins[k]["mode"] == "modpi" and orient_from_noise:
                # keep the noise's side: nearest of {phi, phi+pi}
                d = np.angle(cur * np.exp(-1j * phi))
                flip = (np.abs(d) > np.pi / 2).astype(float)
                phi = phi + np.pi * flip
            new = mag * np.exp(1j * phi)
            keep = np.isnan(targets[..., k])
            spec_new[..., om] = np.where(keep, cur, new)
        z_new = np.fft.irfft(spec_new, n=eps.shape[-2], axis=-1)
        out = out + (z_new - z)[..., None] * u        # lift back along axis
    return out


def extract_mags(chunks, pins):
    """|F| at each pin's (axis, omega): (..., n_pins)."""
    out = []
    for p in pins:
        z = chunks @ np.asarray(p["axis"], dtype=float)
        spec = np.fft.rfft(z, axis=-1)
        out.append(np.abs(spec[..., p["omega"]]))
    return np.stack(out, axis=-1)


def circular_dist(a, b, mod_pi=False):
    """|a - b| on the circle (optionally mod pi), radians."""
    period = np.pi if mod_pi else 2 * np.pi
    d = (a - b) % period
    return np.minimum(d, period - d)


def preservation_check(rng, pins, n=64, H=20, ts=(0.0, 0.25, 0.5, 0.75, 1.0),
                       atol_deg=1.0):
    """Sanity #2 (extended per the reply): phase of x_t constant at pinned
    bins for all t, AND the flow target v = eps~ - a lands on the phi-line
    (phase mod pi) at those bins."""
    a = rng.normal(size=(n, H, 2))
    phis = extract_phases(a, pins)                    # oriented, from action
    eps = rng.normal(size=(n, H, 2))
    et = pin_noise(eps, pins, phis)
    worst_x, worst_v = 0.0, 0.0
    for t in ts:
        xt = t * et + (1 - t) * a
        d = circular_dist(extract_phases(xt, pins), phis)
        worst_x = max(worst_x, float(np.degrees(d.max())))
    v = et - a
    dv = circular_dist(extract_phases(v, pins), phis, mod_pi=True)
    worst_v = float(np.degrees(dv.max()))
    ok = worst_x <= atol_deg and worst_v <= atol_deg
    return {"worst_xt_phase_dev_deg": round(worst_x, 4),
            "worst_v_line_dev_deg": round(worst_v, 4), "pass": bool(ok)}


APPROVED_PINS = [
    {"axis": (1.0, 0.0), "omega": 0, "mode": "mod2pi"},
    {"axis": (1.0, 0.0), "omega": 1, "mode": "mod2pi"},
    {"axis": (1.0, 0.0), "omega": 2, "mode": "mod2pi"},
    {"axis": (0.0, 1.0), "omega": 1, "mode": "modpi"},
]

# Hybrid set S_H — frozen 2026-07-05 on the FIXED generator, pre-registered
# rule: phases = coherence>0.6 + energy>=1% (mod-pi where gamma2 dominates);
# magnitudes = same bins with cross-demo CV of |F| < 0.15.
HYBRID_PINS = [
    {"axis": (1.0, 0.0), "omega": 0, "mode": "mod2pi", "mag": True},
    {"axis": (1.0, 0.0), "omega": 1, "mode": "mod2pi", "mag": True},
    {"axis": (0.0, 1.0), "omega": 1, "mode": "modpi", "mag": True},
    {"axis": (0.0, 1.0), "omega": 2, "mode": "modpi", "mag": False},
]

if __name__ == "__main__":
    import json
    res = preservation_check(np.random.default_rng(0), APPROVED_PINS)
    print(json.dumps(res, indent=2))
    print("PIN_SANITY=" + ("ok" if res["pass"] else "FAILED"))
