"""Post-hoc gate CLEARANCE audit — the collision check the transit judge lacks.

Min distance from each trajectory point to the gate Gaussian cloud (mocap frame),
i.e. the same geometry falsify's COLLISION_GATE fires on. The transit judge
(gate_success.py) latches plane crossings inside the aperture AABB but never checks
rim contact — 2026-08-05 video review (Denis): every compound oracle "success"
clipped the hoop (min clearance 0.001-0.005 m). Scoring rule going forward:
strict success = transit judge + clearance + human video.

Drone body half-extents [0.175, 0.175, 0.075] (safety YAML): point-to-cloud
clearance below ~0.18 m ~= body contact; report <0.18 and <0.10 step counts.

Usage (tv env):
  gate_clearance.py --scene left_and_center|right_and_center|center|left|right \
                    --traj T1.npy [T2.npy ...]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsplat_scene_edit import FALSIFY

EXPORT = "/home/dfliu/code/falsify/data/gate_scenes_export"
LEFT_CK = (f"{EXPORT}/left_scene/mocap_outputs/sagesplat_mocap/sagesplat/"
           "2026-05-11_153901/nerfstudio_models/step-000029999.ckpt")
LEFT_TW2G = np.array([
    [0.12614431661544656, 2.138646801849853e-06, -0.00025306576654559085, -0.15671883492487332],
    [-2.138646801849853e-06, -0.1261265572041315, -0.0021319289354524646, -0.08013551648879384],
    [-0.00025306576654559085, 0.0021319289354524646, -0.12612630156484925, -0.18772133850562778],
    [0, 0, 0, 1.0]])
BODY_R = 0.18   # ~ drone body reach (half-extents [0.175,0.175,0.075])


def _find_right():
    # same ckpt + ICP-composed M as gate_video_scene.py (Tw2g = M @ diag(1,-1,-1,1))
    ck = (f"{EXPORT}/right_scene/mocap_outputs/sagesplat_mocap/sagesplat/"
          "2026-05-11_144353/nerfstudio_models/step-000029999.ckpt")
    M = np.array([[0.136708, -0.001053, 0.006031, -0.111938],
                  [0.00108, 0.13684, -0.000588, 0.030456],
                  [-0.006027, 0.000635, 0.136711, -0.201447],
                  [0, 0, 0, 1.0]])
    return ck, M @ np.diag([1.0, -1, -1, 1])


def gauss_means_mocap(ck, tw2g):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["pipeline"]
    for p in ("_model.gauss_params.means", "_model.means"):
        if p in sd:
            means = sd[p].numpy(); break
    A = tw2g[:3, :3] @ np.diag([1, -1, -1]); t = tw2g[:3, 3]
    return (np.linalg.inv(A) @ (means.T - t[:, None])).T


def aabb_mask(X, ed):
    m = np.all((X >= np.asarray(ed["target_aabb_min"])) &
               (X <= np.asarray(ed["target_aabb_max"])), axis=1)
    for box in ed.get("include_aabbs", []) or []:
        m |= np.all((X >= np.asarray(box["min"])) & (X <= np.asarray(box["max"])), axis=1)
    for box in ed.get("exclude_aabbs", []) or []:
        m &= ~np.all((X >= np.asarray(box["min"])) & (X <= np.asarray(box["max"])), axis=1)
    return m


def gate_cloud(scene):
    """Gate points in mocap for the requested scene, edits applied."""
    if scene in ("right",):                     # native right scene, no edits
        ck, tw2g = _find_right()
        X = gauss_means_mocap(ck, tw2g)
        cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/right_gate.yaml"))
        reg = cfg["gate_region"]
        m = np.all((X >= np.asarray(reg["aabb_min"])) & (X <= np.asarray(reg["aabb_max"])), axis=1)
        return X[m]
    if scene == "right_and_center":
        # defined on the RIGHT splat (its YAML gsplat_path) — falling through to the left splat
        # here masked a gate AABB where no gate exists (same bug as the rollout renderer's
        # scene selection, fixed 2026-08-12)
        ck, tw2g = _find_right()
        X = gauss_means_mocap(ck, tw2g)
    else:
        X = gauss_means_mocap(LEFT_CK, LEFT_TW2G)
    if scene == "left":
        cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/left_gate.yaml"))
        reg = cfg["gate_region"]
        m = np.all((X >= np.asarray(reg["aabb_min"])) & (X <= np.asarray(reg["aabb_max"])), axis=1)
        return X[m]
    if scene == "center":                       # move_gate edit
        cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/center_gate.yaml"))
        (ed,) = [e for e in cfg["scene_edits"] if e["name"] == "move_gate"]
        G = X[aabb_mask(X, ed)]
        T = _anchor_transform(ed["transform"])
        return (T[:3, :3] @ G.T).T + T[:3, 3]
    cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/{scene}.yaml"))
    (ed,) = [e for e in cfg["scene_edits"] if e["type"] == "duplicate_aabb"]
    G1 = X[aabb_mask(X, ed)]
    T = _anchor_transform(ed["transform"])
    G2 = (T[:3, :3] @ G1.T).T + T[:3, 3]
    return np.concatenate([G1, G2], 0)


def _anchor_transform(tr):
    sa = np.asarray(tr["source_anchor"], float); ta = np.asarray(tr["target_anchor"], float)
    sn = np.asarray(tr["source_normal"], float); tn = np.asarray(tr["target_normal"], float)
    th = np.arctan2(tn[1], tn[0]) - np.arctan2(sn[1], sn[0])
    c, s = np.cos(th), np.sin(th)
    T = np.eye(4); T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T[:3, 3] = ta - T[:3, :3] @ sa
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True,
                    choices=["left", "right", "center", "left_and_center", "right_and_center"])
    ap.add_argument("--traj", nargs="+", required=True)
    a = ap.parse_args()
    G = gate_cloud(a.scene)
    G = G[::max(1, len(G) // 25000)]
    gt = torch.tensor(G, dtype=torch.float32)
    print(f"scene={a.scene} gate cloud {len(G)} pts; body-contact threshold ~{BODY_R} m")
    n_clean = 0
    for f in a.traj:
        P = torch.tensor(np.load(f)[:, :3], dtype=torch.float32)
        d = torch.cdist(P, gt).min(1).values.numpy()
        i = int(d.argmin()); clean = d.min() >= BODY_R
        n_clean += clean
        print("%-28s min-clearance %.3f m @step %3d %s  steps<%.2f: %2d  CLEAN=%s" % (
            os.path.basename(f), d.min(), i, np.round(P[i].numpy(), 2),
            BODY_R, int((d < BODY_R).sum()), clean))
    print("== %d/%d clearance-clean (%s)" % (n_clean, len(a.traj), a.scene))


if __name__ == "__main__":
    main()
