"""Agent-authored commands (2026-09-18): one movement primitive -> one pinned chunk.

A closed-loop agent (Claude Code on the drone workstation) decides one chunk at a time. Each
decision is a *move*: a net displacement in the drone's own frame plus a yaw change, executed
over the 50-step chunk with a smooth (cosine) speed profile. The move becomes a per-step
displacement track in the mocap frame, is normalized like the sketch pipeline's chunks, and is
projected through U into the 16-dim command the source noise carries. The flow supplies the
residual; the head and the sketch tracker are bypassed for that replan.

Move spec (all optional, metres / degrees, per chunk):
    {"forward": 1.0, "left": 0.0, "up": 0.0, "yaw_deg": 0.0, "sigma": 0.0}
"forward"/"left" are along / across the current mocap yaw ("forward" = +x rotated by yaw; the
pilot's demos fly with heading == mocap yaw, median offset -0.5 deg, 2026-09-18). Magnitudes are
clamped to the range the command distribution covers; a request beyond the clamp is executed at
the clamp and reported back, never refused silently.
"""
import math
import numpy as np

H, AD = 50, 32
MAX_XY, MAX_Z, MAX_YAW_DEG, MAX_SIGMA = 2.0, 1.0, 45.0, 1.5      # per chunk (5 s)
Z_MIN, Z_MAX = 0.4, 2.3                                          # absolute mocap z the move may target


def _profile(n=H, n_exec=None):
    """Cosine speed profile over the EXECUTED part of the chunk, zero afterwards.

    The move describes a displacement the caller wants to see happen before the next decision. When only
    the first `n_exec` of the chunk's `n` steps execute (short replan intervals), spreading the profile
    over all n steps delivers only a fraction of it -- at n_exec 15 of 50 with this profile, about a
    fifth. Shaping the whole displacement into the executed prefix fixes that (2026-09-21).
    """
    m = int(n if n_exec is None else max(2, min(n, n_exec)))
    w = 1.0 - np.cos(np.linspace(0, 2 * math.pi, m + 2)[1:-1])
    w = w / w.sum()
    out = np.zeros(n, np.float64)
    out[:m] = w
    return out


def clamp_move(move, pose):
    """-> (clamped move dict, list of clamp notes). pose = [x, y, z, yaw]."""
    m = {k: float(move.get(k, 0.0) or 0.0) for k in ("forward", "left", "up", "yaw_deg", "sigma")}
    notes = []
    xy = math.hypot(m["forward"], m["left"])
    if xy > MAX_XY:
        s = MAX_XY / xy; m["forward"] *= s; m["left"] *= s; notes.append(f"xy clamped to {MAX_XY} m")
    if abs(m["up"]) > MAX_Z:
        m["up"] = math.copysign(MAX_Z, m["up"]); notes.append(f"up clamped to {MAX_Z} m")
    z_t = float(pose[2]) + m["up"]
    if z_t < Z_MIN or z_t > Z_MAX:
        m["up"] = float(np.clip(z_t, Z_MIN, Z_MAX)) - float(pose[2]); notes.append(f"target z held inside [{Z_MIN}, {Z_MAX}]")
    if abs(m["yaw_deg"]) > MAX_YAW_DEG:
        m["yaw_deg"] = math.copysign(MAX_YAW_DEG, m["yaw_deg"]); notes.append(f"yaw clamped to {MAX_YAW_DEG} deg")
    if not (0.0 <= m["sigma"] <= MAX_SIGMA):
        m["sigma"] = float(np.clip(m["sigma"], 0.0, MAX_SIGMA)); notes.append(f"sigma clamped to [0, {MAX_SIGMA}]")
    return m, notes


def move_track(move, pose, n_exec=None):
    """Per-step [dx, dy, dz, dyaw] in the mocap frame (H, 4), metres and radians.

    `n_exec` is how many steps of the chunk will actually be executed before the next decision; the
    requested displacement is delivered within those steps.
    """
    yaw = float(pose[3])
    c, s = math.cos(yaw), math.sin(yaw)
    dx = c * move["forward"] - s * move["left"]
    dy = s * move["forward"] + c * move["left"]
    w = _profile(H, n_exec)
    seg = np.zeros((H, 4), np.float32)
    seg[:, 0], seg[:, 1], seg[:, 2] = dx * w, dy * w, move["up"] * w
    seg[:, 3] = math.radians(move["yaw_deg"]) * w
    return seg


class AgentMove:
    """Turns a move into the flow's normalized chunk and the pin command."""

    def __init__(self, amean, astd, U):
        self.amean, self.astd, self.U = np.asarray(amean, np.float32), np.asarray(astd, np.float32), U

    def chunk(self, move, pose, n_exec=None):
        m, notes = clamp_move(move, pose)
        seg = move_track(m, pose, n_exec)
        ch = np.zeros((H, AD), np.float32)
        ch[:, :4] = (seg - self.amean[:4]) / (self.astd[:4] + 1e-6)
        return ch, m, notes

    def command(self, move, pose, n_exec=None):
        """-> (c (16,), clamped move, notes)."""
        ch, m, notes = self.chunk(move, pose, n_exec)
        return (ch.reshape(-1) @ self.U).astype(np.float32), m, notes
