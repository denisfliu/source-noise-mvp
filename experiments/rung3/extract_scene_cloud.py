"""Export a whole-scene Gaussian point cloud (positions + colours, mocap frame) for the
trajectory viewers used in review pages.

Reuses the gate-cloud machinery in gate_clearance.py so the geometry and the scene edits are
identical to the ones the clearance scorer measures against: for `center` the gate AABB is moved
by the YAML `move_gate` transform, for the compound scenes the gate is duplicated. Colours come
from the splat's SH DC term.

This is a VISUALISATION/diagnostic tool. It reads scene YAMLs, which is allowed for building,
scoring and clearly-labelled diagnostics — never as a source of training supervision.

  python extract_scene_cloud.py --scene center --out <dir> [--max-pts 60000]
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_clearance import (LEFT_CK, LEFT_TW2G, _anchor_transform, _find_right, aabb_mask,
                            gauss_means_mocap)
from gsplat_scene_edit import FALSIFY

SH_C0 = 0.28209479177387814


def _colours(ck, sd=None):
    sd = sd if sd is not None else torch.load(ck, map_location="cpu", weights_only=False)["pipeline"]
    for p in ("_model.gauss_params.features_dc", "_model.features_dc", "_model.gauss_params.colors"):
        if p in sd:
            dc = sd[p].numpy().reshape(len(sd[p]), -1)[:, :3]
            break
    else:
        raise SystemExit(f"no colour field in {ck}; keys e.g. {list(sd)[:8]}")
    return np.clip(SH_C0 * dc + 0.5, 0, 1)


def _opacity(sd):
    return torch.sigmoid(sd["_model.gauss_params.opacities"]).numpy().ravel()


# gate regions in the mocap frame, for the gate-priority sampling below (viz only)
GATE_REGION = {
    "left": (np.array([0.36, 0.12, 0.05]), np.array([1.36, 1.27, 2.05])),
    "right": (np.array([-0.06, -1.55, 0.05]), np.array([1.15, -0.75, 2.05])),
}


def scene_cloud(scene):
    """(pts, rgb, gate_mask) for the whole scene with the scene's edits applied. gate_mask marks
    the gate Gaussians so decimation can keep them at full density — a uniform voxel thin reduces
    the thin gate structure to a few dozen points and it disappears from the viewers."""
    # right_and_center is defined on the RIGHT splat (its YAML gsplat_path)
    if scene in ("right", "right_and_center"):
        ck, tw2g = _find_right()
    else:
        ck, tw2g = LEFT_CK, LEFT_TW2G
    sd = torch.load(ck, map_location="cpu", weights_only=False)["pipeline"]
    X = gauss_means_mocap(ck, tw2g)
    rgb, op = _colours(ck, sd), _opacity(sd)
    if scene in ("left", "right"):
        lo, hi = GATE_REGION[scene]
        return X, rgb, np.all((X >= lo) & (X <= hi), axis=1), op
    if scene == "center":
        cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/center_gate.yaml"))
        (ed,) = [e for e in cfg["scene_edits"] if e["name"] == "move_gate"]
        m = aabb_mask(X, ed)
        T = _anchor_transform(ed["transform"])
        Y = X.copy()
        Y[m] = (T[:3, :3] @ X[m].T).T + T[:3, 3]
        return Y, rgb, m, op
    cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/{scene}.yaml"))
    (ed,) = [e for e in cfg["scene_edits"] if e["type"] == "duplicate_aabb"]
    m = aabb_mask(X, ed)
    T = _anchor_transform(ed["transform"])
    dup = (T[:3, :3] @ X[m].T).T + T[:3, 3]
    gm = np.concatenate([m, np.ones(len(dup), bool)])
    return np.concatenate([X, dup], 0), np.concatenate([rgb, rgb[m]], 0), gm, np.concatenate([op, op[m]])


CROP = (np.array([-6.0, -6.0, -0.4]), np.array([6.0, 6.0, 4.0]))  # flight volume; drops far field
# The goal table (the low wire table carrying the penguin, goal box centre (1.525, -0.615, 1.0)) and the
# tub beside it: kept at full density like the gates (Denis, 2026-09-22: "put the point cloud of the
# table in all of these artifacts"); a uniform thin left it ~120 points and invisible.
TABLE_REGION = (np.array([0.95, -1.35, 0.03]), np.array([2.2, 0.05, 1.05]))


# Floaters (Denis, 2026-09-25: "remove the random things in the air but have more of the ground/table"): drop
# near-transparent Gaussians, and above the floor drop Gaussians in sparsely populated 10 cm voxels -- real
# structure (gates, table, walls, the frame around the flight area) is dense, floaters are isolated.
MIN_OPACITY, AIR_Z, VOXEL, MIN_VOXEL_COUNT = 0.5, 0.2, 0.10, 40
FLOOR_SHARE = 0.5   # share of the background budget given to the floor (z < AIR_Z)


def drop_floaters(pts, rgb, gate_mask, op):
    keep = op >= MIN_OPACITY
    pts, rgb, gate_mask = pts[keep], rgb[keep], gate_mask[keep]
    _, inv, cnt = np.unique(np.floor(pts / VOXEL).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    dense = cnt[inv.ravel()] >= MIN_VOXEL_COUNT
    keep = gate_mask | (pts[:, 2] < AIR_Z) | dense
    print(f"floaters dropped: {int((~keep).sum())} of {len(pts)} above-opacity Gaussians")
    return pts[keep], rgb[keep], gate_mask[keep]


def _voxel_thin(p, c, k, rng):
    """One point per voxel, with the voxel size shrunk until there are at least k occupied voxels (a volume-based
    size leaves thin layers like the floor with far fewer), then k of them at random."""
    if len(p) <= k:
        return p, c
    v = (np.prod(p.max(0) - p.min(0) + 1e-6) / k) ** (1 / 3)
    while True:
        _, idx = np.unique(np.floor(p / v).astype(np.int64), axis=0, return_index=True)
        if len(idx) >= k or v < 1e-3:
            break
        v *= 0.7
    idx = rng.permutation(idx)[:k]
    return p[idx], c[idx]


def decimate(pts, rgb, keep, gate_mask, gate_budget=9000, seed=0):
    """Crop to the flight volume, voxel-thin the BACKGROUND to ~keep points (FLOOR_SHARE of them on the floor),
    and keep gate Gaussians and the goal table at full density (subsampled only above gate_budget)."""
    lo, hi = CROP
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts, rgb, gate_mask = pts[inside], rgb[inside], gate_mask[inside]
    tlo, thi = TABLE_REGION
    table = np.all((pts >= tlo) & (pts <= thi), axis=1)
    print(f"table-region points at full density: {int(table.sum())}")
    gate_mask = gate_mask | table
    rng = np.random.default_rng(seed)
    gp, gc = pts[gate_mask], rgb[gate_mask]
    if len(gp) > gate_budget:
        k = rng.permutation(len(gp))[:gate_budget]
        gp, gc = gp[k], gc[k]
    bp, bc = pts[~gate_mask], rgb[~gate_mask]
    kb = max(keep - len(gp), 1)
    fl = bp[:, 2] < AIR_Z
    fp, fc = _voxel_thin(bp[fl], bc[fl], int(kb * FLOOR_SHARE), rng)
    ap_, ac = _voxel_thin(bp[~fl], bc[~fl], kb - len(fp), rng)
    return np.concatenate([fp, ap_, gp], 0), np.concatenate([fc, ac, gc], 0), len(gp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--max-pts", type=int, default=60000)
    a = ap.parse_args()
    pts, rgb, gm, op = scene_cloud(a.scene)
    pts, rgb, gm = drop_floaters(pts, rgb, gm, op)
    pts, rgb, ngate = decimate(pts, rgb, a.max_pts, gm)
    print(f"gate points kept: {ngate}")
    f = f"{a.out}/scene_cloud_{a.scene}.npz"
    np.savez_compressed(f, pts=pts.astype(np.float32), rgb=(rgb * 255).astype(np.uint8))
    print(f"{f}: {len(pts)} pts, extent {np.round(pts.min(0), 2)} .. {np.round(pts.max(0), 2)}")


if __name__ == "__main__":
    main()
