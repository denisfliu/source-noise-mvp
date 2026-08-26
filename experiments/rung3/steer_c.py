"""Steering interface for the pinned command (Denis, 2026-08-09).

c = U^T y is linear in the normalized chunk y, so a desired change in PHYSICAL motion has a
closed-form image in c. For "+d metres of net displacement along axis j", spread d over the
chunk and project onto the basis:

    dc = (d / astd[j]) * U^T m_j,     m_j = unit net-displacement mask for axis j

Measured expressivity of the deployed RRR basis (fraction of the nudge inside span(U)):
x 0.994, y 0.994, z 0.985, roll 0.972, pitch/yaw/gripper 0.000 — this basis carries
translation. The three translation nudges are near-orthogonal inside c (|cos| <= 0.005), so
axes can be commanded independently.

Open-loop the realized chunk displacement follows the command exactly in the pinned
subspace; CLOSED loop a constant nudge is opposed by the state prior's restoring field
(measured gain 0.39-0.73), so realized offset settles below commanded — report the
command-response curve, not a single number.
"""
import os

import numpy as np

H, AD = 50, 32
AXES = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5, "grip": 6}


def unit_mask(axis):
    """Chunk perturbation with net displacement 1.0 (normalized units) on `axis`."""
    j = AXES[axis] if isinstance(axis, str) else int(axis)
    m = np.zeros((H, AD), np.float64)
    m[:, j] = 1.0 / H
    return m.reshape(-1)


def nudge_vector(U, astd, axis, metres):
    """dc to add to c for `metres` of extra net displacement along `axis`.

    Raises for axes the data cannot express: astd == 0 means the demos never move that
    axis (yaw/gripper here), so "metres" has no normalized image at all.
    """
    j = AXES[axis] if isinstance(axis, str) else int(axis)
    s = float(astd[j])
    if s < 1e-6:
        raise ValueError(f"axis {axis!r} has zero action std in this dataset — not steerable")
    return ((metres / s) * (U.T @ unit_mask(axis))).astype(np.float32)


def expressivity(U, axis):
    """Fraction of the axis nudge that lies inside span(U) (1.0 = exactly commandable)."""
    m = unit_mask(axis)
    return float(np.linalg.norm(U @ (U.T @ m)) / np.linalg.norm(m))


def c_to_displacement(U, astd, c, axes=(0, 1, 2)):
    """Inverse view: the net physical displacement (metres) encoded by command `c`.

    Uses the minimum-norm chunk consistent with c (y = U c), so this is "the coarse
    movement this command asks for" — the quantity to draw as an arrow next to the drone.
    """
    y = (np.asarray(U, np.float64) @ np.asarray(c, np.float64)).reshape(H, AD)
    return np.array([y[:, j].sum() * float(astd[j]) for j in axes])


def parse_env(U, astd):
    """NUDGE='z:+0.30,y:-0.10' -> summed dc (None if unset). Used by the pin servers."""
    spec = os.environ.get("NUDGE", "").strip()
    if not spec:
        return None
    dc = np.zeros(U.shape[1], np.float32)
    for part in spec.split(","):
        ax, val = part.split(":")
        dc = dc + nudge_vector(U, astd, ax.strip(), float(val))
    return dc


if __name__ == "__main__":
    import openpi.shared.normalize as NZ
    RD = os.path.dirname(os.path.abspath(__file__))
    U = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float64)
    ns = NZ.load(os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    astd = np.asarray(ns["actions"].std)
    print(f"basis {U.shape}, action std (m/step) {np.round(astd[:3], 4)}")
    for ax in ("x", "y", "z", "roll", "yaw"):
        if float(astd[AXES[ax]]) < 1e-6:
            print(f"  {ax:5s} not steerable (zero action std in the demos)")
            continue
        print(f"  {ax:5s} expressivity {expressivity(U, ax):.3f}   "
              f"dc for +0.30 m = {np.round(nudge_vector(U, astd, ax, 0.30), 2)}")
    # verification: applying dc to a real chunk's c changes its pinned displacement by d
    import glob
    f = sorted(glob.glob(f"{RD}/data_gate_synth/ep_*.npz"))[110]
    d = np.load(f, allow_pickle=True)
    a = d["action"][40:40 + H].astype(np.float64)
    amean = np.asarray(ns["actions"].mean)
    y = np.zeros((H, AD)); y[:, :7] = (a[:, :7] - amean[:7]) / (astd[:7] + 1e-6)
    y = y.reshape(-1)
    for ax in ("x", "y", "z"):
        for metres in (0.1, 0.3):
            dc = nudge_vector(U, astd, ax, metres)
            y2 = y + (dc - 0 * dc) @ U.T  # apply the command change in the pinned subspace
            j = AXES[ax]
            disp = lambda v: (v.reshape(H, AD)[:, j] * astd[j] + amean[j]).sum()
            print(f"  check {ax} +{metres:.2f} m -> realized {disp(y2) - disp(y):+.3f} m "
                  f"(pinned-subspace, open loop)")
