"""Plan/elevation views of rollouts against the real gate geometry, as inline SVG.

The overlay videos are the drone's own camera, which makes it hard to judge how a flight sits
relative to the gate opening. This renders the same trajectories against the gate's Gaussian point
cloud (the geometry `gate_clearance.py` scores against) plus the goal box, from above and from the
side, so a miss is legible.

  from plan_view_svg import panels
  svg = panels("left", {"grounded": (trajs, "#e0787a"), "one-hot": (trajs, "#5fbe86")})

Simulator geometry, used for scoring and diagnostics only — never as training supervision.
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_clearance import gate_cloud
from gsplat_scene_edit import FALSIFY

GOAL_HALF = np.array([0.3, 0.3, 0.5])


def goal_position(scene):
    name = {"left": "left_gate", "right": "right_gate", "center": "center_gate"}.get(scene, scene)
    cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/{name}.yaml"))
    return np.asarray(cfg["goal_position_mocap"], float)


def _panel(G, goal, series, ax, ay, w, h, pad, label, seed=0):
    """One projection. ax/ay are the coordinate indices to plot (0=x, 1=y, 2=z)."""
    pts = np.concatenate([G[:, [ax, ay]]] +
                         [np.asarray(t)[:, [ax, ay]] for trajs, _ in series.values() for t in trajs])
    lo, hi = pts.min(0) - 0.25, pts.max(0) + 0.25
    span = np.maximum(hi - lo, 1e-6)
    sc = min((w - 2 * pad) / span[0], (h - 2 * pad) / span[1])
    ox = pad + ((w - 2 * pad) - span[0] * sc) / 2
    oy = pad + ((h - 2 * pad) - span[1] * sc) / 2

    def X(u):
        return ox + (u - lo[0]) * sc

    def Y(v):
        return h - oy - (v - lo[1]) * sc          # flip so +axis points up

    rng = np.random.default_rng(seed)
    k = rng.permutation(len(G))[:2600]
    dots = " ".join(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="1.1"/>' for p in G[k][:, [ax, ay]])
    gl, gh = goal[[ax, ay]] - GOAL_HALF[[ax, ay]], goal[[ax, ay]] + GOAL_HALF[[ax, ay]]
    box = (f'<rect x="{X(gl[0]):.1f}" y="{Y(gh[1]):.1f}" width="{(gh[0]-gl[0])*sc:.1f}" '
           f'height="{(gh[1]-gl[1])*sc:.1f}" fill="none" stroke="currentColor" stroke-width="1.2" '
           f'stroke-dasharray="5 4" opacity=".55"/>'
           f'<text x="{X(gh[0])+5:.1f}" y="{Y(gh[1])+11:.1f}" font-size="10.5" fill="currentColor" '
           f'opacity=".6">goal box</text>')
    paths = ""
    for name, (trajs, colour) in series.items():
        for t in trajs:
            t = np.asarray(t)
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(p[ax]):.1f},{Y(p[ay]):.1f}"
                         for i, p in enumerate(t[::2]))
            paths += f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.5" opacity=".82"/>'
        t0 = np.asarray(trajs[0])
        paths += (f'<circle cx="{X(t0[0, ax]):.1f}" cy="{Y(t0[0, ay]):.1f}" r="3.4" fill="{colour}"/>')
    names = "xyz"
    ticks = ""
    for u in np.arange(np.ceil(lo[0]), hi[0] + 1e-9, 1.0):
        ticks += (f'<line x1="{X(u):.1f}" y1="{h-oy:.1f}" x2="{X(u):.1f}" y2="{h-oy+4:.1f}" '
                  f'stroke="currentColor" opacity=".4"/>'
                  f'<text x="{X(u):.1f}" y="{h-oy+16:.1f}" font-size="10" text-anchor="middle" '
                  f'fill="currentColor" opacity=".5">{u:.0f}</text>')
    for v in np.arange(np.ceil(lo[1]), hi[1] + 1e-9, 1.0):
        ticks += (f'<line x1="{ox-4:.1f}" y1="{Y(v):.1f}" x2="{ox:.1f}" y2="{Y(v):.1f}" '
                  f'stroke="currentColor" opacity=".4"/>'
                  f'<text x="{ox-7:.1f}" y="{Y(v)+3.5:.1f}" font-size="10" text-anchor="end" '
                  f'fill="currentColor" opacity=".5">{v:.0f}</text>')
    return (f'<g><text x="{pad}" y="14" font-size="11.5" fill="currentColor" opacity=".72">{label}'
            f'  ({names[ax]} right, {names[ay]} up, metres)</text>'
            f'<g fill="currentColor" opacity=".33">{dots}</g>{box}{ticks}{paths}</g>')


def panels(scene, series, w=470, h=330, note=""):
    """Side-by-side plan (x-y) and elevation (x-z) SVG for one scene."""
    G = gate_cloud(scene)
    goal = goal_position(scene)
    a = _panel(G, goal, series, 0, 1, w, h, 34, "from above", seed=0)
    b = _panel(G, goal, series, 0, 2, w, h, 34, "from the side", seed=1)
    legend = "".join(
        f'<g><rect x="{8 + i * 150}" y="{h + 8}" width="16" height="3" fill="{c}"/>'
        f'<text x="{30 + i * 150}" y="{h + 14}" font-size="11.5" fill="currentColor">{n}</text></g>'
        for i, (n, (_, c)) in enumerate(series.items()))
    return (f'<svg viewBox="0 0 {2 * w + 16} {h + 26}" role="img" aria-label="{note or scene}">'
            f'{a}<g transform="translate({w + 16},0)">{b}</g>{legend}</svg>')
