"""Apply the center_gate `move_gate` scene edit to raw gsplat tensors (tv env).

Replicates falsify.sim.scene_edits.RigidTransformAABB for our standalone
renderer: yaw-only rotation solved from source/target normal xy + anchor
translation, authored in MOCAP; selection by target AABB + include strips −
exclude boxes (all MOCAP). Params read from the published
configs/scenes/center_gate.yaml — never hardcoded here.

The splat's means/quats live in the scene's native (NS) frame; MOCAP points map
via p_ns = A @ p_mocap + b with A = Tw2g[:3,:3] @ diag(1,-1,-1), b = Tw2g[:3,3]
(the same convention as the render chain's to_ns()). The edit is conjugated:
T_ns = M @ T_mocap @ M^{-1}; quats rotate by the pure-rotation part
R_q = R_A @ Rz(theta) @ R_A^T (A = s*R_A, uniform scale).
Means and quats only, per the Splat-MOVER precedent falsify follows.
"""
import math
import os

import numpy as np
import torch
import yaml

FALSIFY = os.path.expanduser("~/code/falsify-pi")


def load_move_gate_edit():
    cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/center_gate.yaml"))
    (edit,) = [e for e in cfg["scene_edits"] if e["name"] == "move_gate"]
    assert edit["type"] == "rigid_transform_aabb" and edit["transform_frame"] == "mocap"
    return edit


def load_duplicate_edit(scene):
    """duplicate_aabb edit from left_and_center / right_and_center scene configs."""
    cfg = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/{scene}.yaml"))
    (edit,) = [e for e in cfg["scene_edits"] if e["type"] == "duplicate_aabb"]
    assert edit["transform_frame"] == "mocap"
    return edit


def _mask_mocap(P, edit):
    mn = np.asarray(edit["target_aabb_min"]); mx = np.asarray(edit["target_aabb_max"])
    m = np.all((P >= mn) & (P <= mx), axis=1)
    for box in edit.get("include_aabbs", []) or []:
        m |= np.all((P >= np.asarray(box["min"])) & (P <= np.asarray(box["max"])), axis=1)
    for box in edit.get("exclude_aabbs", []) or []:
        m &= ~np.all((P >= np.asarray(box["min"])) & (P <= np.asarray(box["max"])), axis=1)
    return m


def _quat_mul(q, r):
    w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w2, x2, y2, z2 = r[0], r[1], r[2], r[3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=1)


def apply_move_gate(means, quats, tw2g):
    """In-place-style edit; returns (means', quats', n_moved). Tensors on any device."""
    edit = load_move_gate_edit()
    A = tw2g[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    b = tw2g[:3, 3]
    Ainv = np.linalg.inv(A)
    dev = means.device
    P_ns = means.detach().cpu().numpy().astype(np.float64)
    P_mocap = (P_ns - b) @ Ainv.T
    mask = _mask_mocap(P_mocap, edit)

    s_xy = np.asarray(edit["transform"]["source_normal"], float)[:2]
    t_xy = np.asarray(edit["transform"]["target_normal"], float)[:2]
    theta = math.atan2(t_xy[1], t_xy[0]) - math.atan2(s_xy[1], s_xy[0])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    src_a = np.asarray(edit["transform"]["source_anchor"], float)
    tgt_a = np.asarray(edit["transform"]["target_anchor"], float)
    t_vec = tgt_a - R @ src_a

    # move means: mocap-frame edit, then back to NS
    moved = (R @ P_mocap[mask].T).T + t_vec
    P_ns_new = moved @ A.T + b
    means = means.clone()
    means[torch.as_tensor(mask, device=dev)] = torch.as_tensor(P_ns_new, dtype=means.dtype, device=dev)

    # rotate quats by the NS-frame pure rotation
    scale = np.cbrt(abs(np.linalg.det(A)))
    R_A = A / scale
    R_ns = R_A @ R @ R_A.T
    tr = np.trace(R_ns)
    qw = math.sqrt(max(0.0, 1.0 + tr)) / 2.0
    qx = (R_ns[2, 1] - R_ns[1, 2]) / (4 * qw)
    qy = (R_ns[0, 2] - R_ns[2, 0]) / (4 * qw)
    qz = (R_ns[1, 0] - R_ns[0, 1]) / (4 * qw)
    q_R = torch.tensor([qw, qx, qy, qz], dtype=quats.dtype, device=dev)
    quats = quats.clone()
    mt = torch.as_tensor(mask, device=dev)
    quats[mt] = _quat_mul(quats[mt], q_R)
    return means, quats, int(mask.sum())


def apply_duplicate_gate(means, quats, scales, opac, colors, tw2g, scene="left_and_center"):
    """COPY the gate Gaussians to the target pose (original stays) — falsify
    duplicate_aabb replication. Returns extended tensors + n_copied."""
    import torch
    edit = load_duplicate_edit(scene)
    A = tw2g[:3, :3] @ np.diag([1.0, -1.0, -1.0]); b = tw2g[:3, 3]
    Ainv = np.linalg.inv(A)
    P_mocap = (means.detach().cpu().numpy().astype(np.float64) - b) @ Ainv.T
    mask = _mask_mocap(P_mocap, edit)
    s_xy = np.asarray(edit["transform"]["source_normal"], float)[:2]
    t_xy = np.asarray(edit["transform"]["target_normal"], float)[:2]
    theta = math.atan2(t_xy[1], t_xy[0]) - math.atan2(s_xy[1], s_xy[0])
    c_, s_ = math.cos(theta), math.sin(theta)
    R = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
    t_vec = np.asarray(edit["transform"]["target_anchor"], float) - R @ np.asarray(edit["transform"]["source_anchor"], float)
    moved = (R @ P_mocap[mask].T).T + t_vec
    P_new = moved @ A.T + b
    dev = means.device
    mt = torch.as_tensor(mask, device=dev)
    scale = np.cbrt(abs(np.linalg.det(A)))
    R_A = A / scale
    R_ns = R_A @ R @ R_A.T
    tr = np.trace(R_ns)
    qw = math.sqrt(max(0.0, 1.0 + tr)) / 2.0
    q_R = torch.tensor([qw, (R_ns[2, 1] - R_ns[1, 2]) / (4 * qw),
                        (R_ns[0, 2] - R_ns[2, 0]) / (4 * qw),
                        (R_ns[1, 0] - R_ns[0, 1]) / (4 * qw)], dtype=quats.dtype, device=dev)
    new_means = torch.as_tensor(P_new, dtype=means.dtype, device=dev)
    new_quats = _quat_mul(quats[mt], q_R)
    means2 = torch.cat([means, new_means], 0)
    quats2 = torch.cat([quats, new_quats], 0)
    scales2 = torch.cat([scales, scales[mt]], 0)
    opac2 = torch.cat([opac, opac[mt]], 0)
    colors2 = torch.cat([colors, colors[mt]], 0)
    return means2, quats2, scales2, opac2, colors2, int(mask.sum())
