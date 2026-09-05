"""Sketch geometry: evaluate a drawn command polyline against the judge apertures and the
gate cloud the flight scorer uses (gate_clearance.gate_cloud). Shared by
viz/build_sketchreview.py and by sketch editing.

  evaluate(points, scene) -> dict(minclr, minclr_xyz, minclr_seg, under, crossings, dense, dist)
"""
import os
import sys

import numpy as np
import torch
import yaml

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import gate_clearance as GC  # noqa: E402

FALSIFY = os.path.expanduser("~/code/falsify-pi")
SAFETY = {"left": "left_gate", "right": "right_gate", "center": "center_gate",
          "left_and_center": "left_and_center", "right_and_center": "right_and_center"}
BODY_R = GC.BODY_R
STEP = 0.01   # densification step along the polyline (m)
NEAR = 0.8    # report plane crossings only within this distance of the aperture rectangle (m)
_cloud, _gates = {}, {}


def gates_for(scene):
    safety = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{SAFETY[scene]}.yaml"))
    if "ordered_miss_gate" in safety:
        return [(g["name"], np.asarray(g["corners"], float)) for g in safety["ordered_miss_gate"]["gates"]]
    return [("gate", np.asarray(safety["miss_gate"]["corners"], float))]


def gate_cloud(scene):
    if scene not in _cloud:
        G = GC.gate_cloud(scene); G = G[::max(1, len(G) // 25000)]
        _cloud[scene] = torch.tensor(G, dtype=torch.float32)
        _gates[scene] = gates_for(scene)
    return _cloud[scene], _gates[scene]


def densify(P):
    out, seg = [], []
    for i in range(len(P) - 1):
        L = np.linalg.norm(P[i + 1] - P[i])
        n = max(2, int(np.ceil(L / STEP)) + 1)
        ts = np.linspace(0, 1, n, endpoint=(i == len(P) - 2))
        out.append(P[i] + ts[:, None] * (P[i + 1] - P[i]))
        seg.append(np.full(len(ts), i))
    return np.concatenate(out), np.concatenate(seg)


def crossings(dense, dist, gates):
    rows = []
    for name, C in gates:
        c0, c1, c3 = C[0], C[1], C[3]
        u = c1 - c0; W = np.linalg.norm(u); u /= W
        v = c3 - c0; H = np.linalg.norm(v); v /= H
        n = np.cross(u, v); n /= np.linalg.norm(n)
        s = (dense - c0) @ n
        for i in np.where(np.sign(s[:-1]) * np.sign(s[1:]) < 0)[0]:
            t = s[i] / (s[i] - s[i + 1]); x = dense[i] + t * (dense[i + 1] - dense[i])
            a, b = float((x - c0) @ u), float((x - c0) @ v)
            inside = 0 <= a <= W and 0 <= b <= H
            if inside:
                margin = min(a, W - a, b, H - b)
            else:
                da = max(0, -a, a - W); db = max(0, -b, b - H); margin = -float(np.hypot(da, db))
                if margin < -NEAR:      # a crossing of the extended plane far from the opening
                    continue
            rows.append({"gate": name, "xyz": [round(float(k), 3) for k in x], "u": round(a, 3), "v": round(b, 3),
                         "W": round(float(W), 3), "H": round(float(H), 3), "inside": bool(inside),
                         "edge_margin": round(float(margin), 3), "cloud_clearance": round(float(dist[i]), 3),
                         "direction": "+n" if s[i + 1] > s[i] else "-n", "sample": int(i)})
    return rows


def evaluate(points, scene):
    P = np.asarray(points, float)[:, :3]
    cloud, gates = gate_cloud(scene)
    dense, seg = densify(P)
    dist = torch.cdist(torch.tensor(dense, dtype=torch.float32), cloud).min(1).values.numpy()
    i = int(dist.argmin())
    return {"minclr": float(dist[i]), "minclr_xyz": dense[i].tolist(), "minclr_seg": int(seg[i]),
            "under": float((dist < BODY_R).sum() * STEP), "crossings": crossings(dense, dist, gates),
            "dense": dense, "dist": dist, "seg": seg}


def summary(ev):
    xs = "; ".join(f"{x['gate']} {'in' if x['inside'] else 'OUT'} margin {x['edge_margin']:.2f} clr {x['cloud_clearance']:.2f}"
                   for x in ev["crossings"])
    return f"minclr {ev['minclr']:.3f} (seg {ev['minclr_seg']+1}->{ev['minclr_seg']+2}) under {ev['under']:.2f} m | {xs}"
