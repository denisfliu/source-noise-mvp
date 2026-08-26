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


def _colours(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["pipeline"]
    for p in ("_model.gauss_params.features_dc", "_model.features_dc", "_model.gauss_params.colors"):
        if p in sd:
            dc = sd[p].numpy().reshape(len(sd[p]), -1)[:, :3]
            break
    else:
        raise SystemExit(f"no colour field in {ck}; keys e.g. {list(sd)[:8]}")
    return np.clip(SH_C0 * dc + 0.5, 0, 1)


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
    X = gauss_means_mocap(ck, tw2g)
    rgb = _colours(ck)
    if scene in ("left", "right"):
        lo, hi = GATE_REGION[scene]
        return X, rgb, np.all((X >= lo) & (X <= hi), axis=1)
    if scene == "center":
        cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/center_gate.yaml"))
        (ed,) = [e for e in cfg["scene_edits"] if e["name"] == "move_gate"]
        m = aabb_mask(X, ed)
        T = _anchor_transform(ed["transform"])
        Y = X.copy()
        Y[m] = (T[:3, :3] @ X[m].T).T + T[:3, 3]
        return Y, rgb, m
    cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/{scene}.yaml"))
    (ed,) = [e for e in cfg["scene_edits"] if e["type"] == "duplicate_aabb"]
    m = aabb_mask(X, ed)
    T = _anchor_transform(ed["transform"])
    dup = (T[:3, :3] @ X[m].T).T + T[:3, 3]
    gm = np.concatenate([m, np.ones(len(dup), bool)])
    return np.concatenate([X, dup], 0), np.concatenate([rgb, rgb[m]], 0), gm


CROP = (np.array([-6.0, -6.0, -0.4]), np.array([6.0, 6.0, 4.0]))  # flight volume; drops far field


def decimate(pts, rgb, keep, gate_mask, gate_budget=3000, seed=0):
    """Crop to the flight volume, voxel-thin the BACKGROUND to ~keep points, and keep gate
    Gaussians at full density (subsampled only above gate_budget)."""
    lo, hi = CROP
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts, rgb, gate_mask = pts[inside], rgb[inside], gate_mask[inside]
    rng = np.random.default_rng(seed)
    gp, gc = pts[gate_mask], rgb[gate_mask]
    if len(gp) > gate_budget:
        k = rng.permutation(len(gp))[:gate_budget]
        gp, gc = gp[k], gc[k]
    bp, bc = pts[~gate_mask], rgb[~gate_mask]
    kb = max(keep - len(gp), 1)
    if len(bp) > kb:
        lo, hi = bp.min(0), bp.max(0)
        v = (np.prod(hi - lo + 1e-6) / max(kb, 1)) ** (1 / 3)
        _, idx = np.unique(np.floor(bp / v).astype(np.int64), axis=0, return_index=True)
        if len(idx) > kb:
            idx = rng.permutation(idx)[:kb]
        bp, bc = bp[idx], bc[idx]
    return np.concatenate([bp, gp], 0), np.concatenate([bc, gc], 0), len(gp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--max-pts", type=int, default=60000)
    a = ap.parse_args()
    pts, rgb, gm = scene_cloud(a.scene)
    pts, rgb, ngate = decimate(pts, rgb, a.max_pts, gm)
    print(f"gate points kept: {ngate}")
    f = f"{a.out}/scene_cloud_{a.scene}.npz"
    np.savez_compressed(f, pts=pts.astype(np.float32), rgb=(rgb * 255).astype(np.uint8))
    print(f"{f}: {len(pts)} pts, extent {np.round(pts.min(0), 2)} .. {np.round(pts.max(0), 2)}")


if __name__ == "__main__":
    main()
