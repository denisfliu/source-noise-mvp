"""Coherence estimators and structural-set selection (Step 1 of the plan).

Two estimators over (direction theta, temporal frequency omega):

  gamma  (plain):        E_scenes | (1/N) sum_demos exp(j * phi) |
  gamma2 (angle-doubled): E_scenes | (1/N) sum_demos exp(j * 2*phi) |

gamma2 is invariant to per-demo pi flips (bimodal bend side), matching what
the noise channel can bind (phase mod pi). CAVEAT (surfaced in the reply
discussion): rFFT bins 0 (DC) and H/2 (Nyquist) have real coefficients, so
their phase is in {0, pi} and gamma2 is trivially 1 there — gamma2 is only
meaningful on complex bins 1..H/2-1 and is masked accordingly.

Selection rule (pre-registered in the reply addendum): structural set =
{(axis, omega)} with gamma (mod-2pi pin) or gamma2 (mod-pi pin) above 0.6,
axes restricted to {progress, lateral} after G1 confirms grid maxima land
near those axes.
"""

import numpy as np

GAMMA_THRESHOLD = 0.6


def phase_field(chunks_canonical, theta):
    """chunks (..., H, 2), theta scalar -> phases (..., n_bins)."""
    u = np.array([np.cos(theta), np.sin(theta)])
    z = chunks_canonical @ u                     # (..., H)
    return np.angle(np.fft.rfft(z, axis=-1))    # (..., H//2+1)


def coherence_maps(chunks_canonical, thetas):
    """chunks (M, N, H, 2) -> gamma, gamma2 of shape (len(thetas), n_bins)."""
    g, g2 = [], []
    for th in thetas:
        phi = phase_field(chunks_canonical, th)          # (M, N, B)
        r1 = np.abs(np.exp(1j * phi).mean(axis=1))       # (M, B) per-scene
        r2 = np.abs(np.exp(2j * phi).mean(axis=1))
        g.append(r1.mean(axis=0))
        g2.append(r2.mean(axis=0))
    g, g2 = np.array(g), np.array(g2)
    H = chunks_canonical.shape[-2]
    real_bins = [0] + ([H // 2] if H % 2 == 0 else [])
    g2_masked = g2.copy()
    g2_masked[:, real_bins] = np.nan                     # trivially 1, contentless
    return g, g2_masked


def select_structure(gamma, gamma2, thetas, axes=(0.0, np.pi / 2),
                     threshold=GAMMA_THRESHOLD, tol_deg=15.0):
    """Returns G1 verdict + selected (axis, omega, mode) triples."""
    n_bins = gamma.shape[1]
    # G1: do grid maxima land near the planted axes?
    best_theta_per_bin = thetas[np.nanargmax(gamma, axis=0)]
    best2_theta_per_bin = thetas[np.nanargmax(np.nan_to_num(gamma2), axis=0)]

    def near_axis(th):
        d = [min(abs(((th - ax) + np.pi / 2) % np.pi - np.pi / 2),
                 np.pi) for ax in axes]
        return np.degrees(min(d))

    selected = []
    for b in range(n_bins):
        for ax in axes:
            i = int(np.argmin(np.abs(((thetas - ax) + np.pi / 2) % np.pi - np.pi / 2)))
            if gamma[i, b] > threshold:
                selected.append({"axis_deg": round(np.degrees(ax), 1), "omega": b,
                                 "mode": "mod2pi", "gamma": round(float(gamma[i, b]), 3)})
            elif not np.isnan(gamma2[i, b]) and gamma2[i, b] > threshold:
                selected.append({"axis_deg": round(np.degrees(ax), 1), "omega": b,
                                 "mode": "modpi", "gamma2": round(float(gamma2[i, b]), 3)})
    strong_bins = [b for b in range(n_bins)
                   if np.nanmax(gamma[:, b]) > threshold or
                   np.nanmax(np.nan_to_num(gamma2)[:, b]) > threshold]
    axis_errs = {b: round(min(near_axis(best_theta_per_bin[b]),
                              near_axis(best2_theta_per_bin[b])), 1)
                 for b in strong_bins}
    g1_pass = all(v <= tol_deg for v in axis_errs.values()) and len(selected) > 0
    return {"g1_pass": bool(g1_pass), "axis_err_deg_per_strong_bin": axis_errs,
            "selected": selected}


def ascii_heatmap(grid, thetas, title, col_labels=None):
    """Coarse text rendering for terminal/README inspection."""
    chars = " .:-=+*#%@"
    lines = [title]
    n_bins = grid.shape[1]
    lines.append("theta\\w |" + "".join(f"{b:>4}" for b in range(n_bins)))
    for i in range(0, len(thetas), max(1, len(thetas) // 18)):
        row = ""
        for b in range(n_bins):
            v = grid[i, b]
            row += "   -" if np.isnan(v) else f"   {chars[min(int(v * 9.99), 9)]}"
        lines.append(f"{np.degrees(thetas[i]):6.1f}  |" + row)
    return "\n".join(lines)
