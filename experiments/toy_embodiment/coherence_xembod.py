"""Cross-EMBODIMENT coherence: the toy_frame estimator with the averaging axis
changed from 'demos of a scene' to 'bodies executing a scene'.

structure = tip-motion component that SYNCHRONIZES across embodiments doing the
same task; embodiment-private realization = the incoherent residual. Returns a
shared frame (like toy_frame Step 1) AND a pairwise coherence scalar c(i,j) that
quantifies how much any two bodies share (the number G-predict tests).

Reuses toy_frame/coherence.py (coherence_maps, select_structure, ascii_heatmap).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "toy_frame"))
import coherence as tfc                 # noqa: E402  (toy_frame/coherence.py)
import dataset as tfd                   # noqa: E402  (toy_frame/dataset.py, for to_canonical)


def _body_repr(chunks_body, angles):
    """(M,N,H,2) world -> per-scene representative canonical chunk (M,H,2)
    (mean over that body's demos, to average out within-body style)."""
    can = tfd.to_canonical(chunks_body, angles[:, None])  # (M,N,H,2), angle bcast over demos
    return can.mean(axis=1)                       # (M,H,2)


def stack_bodies(chunks, angles, body_names):
    """-> (M, n_bodies, H, 2) of per-scene body-representative canonical chunks.
    The n_bodies axis plays the role toy_frame's 'demos' axis played."""
    return np.stack([_body_repr(chunks[b], angles) for b in body_names], axis=1)


def coherence_over(chunks, angles, body_names, thetas):
    """gamma, gamma2 (len(thetas), n_bins) of cross-body phase agreement."""
    stk = stack_bodies(chunks, angles, body_names)
    return tfc.coherence_maps(stk, thetas)


def pairwise_c(chunks, angles, body_names, thetas, selected):
    """c(i,j) = mean over the SELECTED (axis,omega) bins of the two-body
    coherence (mod-2pi for mod2pi pins, mod-pi/gamma2 for modpi pins)."""
    if not selected:
        return {}
    out = {}
    for a in range(len(body_names)):
        for b in range(a + 1, len(body_names)):
            g, g2 = coherence_over(chunks, angles,
                                   [body_names[a], body_names[b]], thetas)
            vals = []
            for sel in selected:
                th = np.radians(sel["axis_deg"]); om = sel["omega"]
                i = int(np.argmin(np.abs(((thetas - th) + np.pi / 2) % np.pi - np.pi / 2)))
                grid = g2 if sel["mode"] == "modpi" else g
                v = grid[i, om]
                if not np.isnan(v):
                    vals.append(float(v))
            out[f"{body_names[a]}~{body_names[b]}"] = round(float(np.mean(vals)), 3)
    return out


def align_to_consensus(chunks, angles, set_a, bodyB, pins):
    """How well body B's phase ALIGNS with the set-A consensus on the S_A bins,
    in [-1,1] (1 = perfect alignment). This is the right per-body divergence
    measure for G-predict: unlike pooled concentration c(B,setA) (dominated by
    the set-A arms agreeing among themselves), this is sensitive to B's own
    phase. Lower = B's structure disagrees with the training bodies."""
    stkA = stack_bodies(chunks, angles, set_a)            # (M,|A|,H,2)
    repB = _body_repr(chunks[bodyB], angles)              # (M,H,2)
    vals = []
    for p in pins:
        u = np.asarray(p["axis"], dtype=float)
        mult = 2.0 if p["mode"] == "modpi" else 1.0
        phiA = np.angle(np.fft.rfft(stkA @ u, axis=-1))[..., p["omega"]]   # (M,|A|)
        cons = np.angle(np.exp(1j * mult * phiA).mean(axis=1))             # (M,) mult-space
        phiB = np.angle(np.fft.rfft(repB @ u, axis=-1))[..., p["omega"]]   # (M,)
        vals.append(float(np.mean(np.cos(mult * phiB - cons))))
    return round(float(np.mean(vals)), 3)


def synthetic_recovery(rng, n_scenes=60, n_bodies=3, H=20, theta_star_deg=35.0,
                       omega_star=2, thetas=None):
    """Plant a shared phase at (theta_star, omega_star) across synthetic bodies
    (body-private noise elsewhere), confirm the estimator recovers it, and
    confirm that adding a strongly divergent body LOWERS coherence at that bin."""
    if thetas is None:
        thetas = np.linspace(0.0, np.pi, 90)
    th = np.radians(theta_star_deg)
    u = np.array([np.cos(th), np.sin(th)]); u_perp = np.array([-u[1], u[0]])
    B = H // 2 + 1
    fields = np.zeros((n_scenes, n_bodies, H, 2))
    for s in range(n_scenes):
        shared_phase = rng.uniform(-np.pi, np.pi)     # scene-specific, body-shared
        for b in range(n_bodies):
            spec = (rng.normal(size=B) + 1j * rng.normal(size=B))
            spec[omega_star] = 2.0 * np.exp(1j * shared_phase)      # planted
            z_u = np.fft.irfft(spec, n=H)
            z_perp = rng.normal(size=H) * 0.5                        # private
            fields[s, b] = np.outer(z_u, u) + np.outer(z_perp, u_perp)
    g, g2 = tfc.coherence_maps(fields, thetas)
    i_star = int(np.argmin(np.abs(thetas - th)))
    rec_theta = float(np.degrees(thetas[np.nanargmax(g[:, omega_star])]))
    # divergent body: random phase at the planted bin -> should drop coherence
    div = fields.copy()
    for s in range(n_scenes):
        spec = rng.normal(size=B) + 1j * rng.normal(size=B)
        div[s, -1] = np.outer(np.fft.irfft(spec, n=H), u)
    g_div, _ = tfc.coherence_maps(div, thetas)
    return {"planted_theta_deg": theta_star_deg, "recovered_theta_deg": round(rec_theta, 1),
            "planted_omega": omega_star,
            "gamma_at_planted": round(float(g[i_star, omega_star]), 3),
            "gamma_with_divergent_body": round(float(g_div[i_star, omega_star]), 3),
            "theta_err_deg": round(abs(rec_theta - theta_star_deg), 1)}
