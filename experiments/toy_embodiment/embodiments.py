"""Embodiments for the cross-embodiment toy (Rung 1), Option-1 design.

Denis-approved 2026-07-17: ALL bodies act in task-space (tip position-delta)
coordinates, so the shared invariant (tip-trajectory phase/magnitude in the
canonical task frame) stays LINEAR in every body's action space and the
source-noise pin is carried exactly (preservation property intact — see
docs/REPRODUCTION_GUIDE.md pitfall #2). The embodiment difference is therefore
NOT the action parameterization but the body's ability to REALIZE a commanded
tip motion: reachable workspace + loss of radial authority near full extension.

Each body maps a planned tip path (T+1 positions in the world frame) to the
tip path it actually ACHIEVES. The stored demo is the achieved tip-delta chunk.

Model (leading-order planar kinematics, NOT full IK — deliberately simple and
robust for a fast toy; full Jacobian IK is a possible refinement):
  - PointRobot: identity (holonomic; the drone analog — no body constraint).
  - PlanarArm(reach): base at origin; per-step tip motion decomposed into
    radial (w.r.t. base) + tangential; the OUTWARD-radial component is
    attenuated -> 0 as the tip radius approaches `reach` (an arm near full
    extension cannot push its tip further out), then the tip is hard-clipped
    to [0, reach]. Tangential motion is unconstrained. Larger `reach` = more
    capable arm; the point robot is the reach=inf limit.
"""

import numpy as np

EPS = 1e-9


class Body:
    name = "body"

    def realize(self, planned_pos):
        raise NotImplementedError


class PointRobot(Body):
    name = "point"
    reach = np.inf

    def realize(self, planned_pos):
        return np.asarray(planned_pos, dtype=float).copy()


class PlanarArm(Body):
    def __init__(self, name, reach, atten_power=3.0):
        self.name = name
        self.reach = float(reach)
        self.atten_power = float(atten_power)

    def realize(self, planned_pos):
        P = np.asarray(planned_pos, dtype=float)
        T = P.shape[0]
        out = np.empty_like(P)
        # start: clip the planned start into the annulus [0, reach]
        r0 = np.linalg.norm(P[0])
        out[0] = P[0] * (min(r0, self.reach) / (r0 + EPS)) if r0 > EPS else P[0]
        for i in range(1, T):
            cur = out[i - 1]
            desired = P[i] - P[i - 1]
            cur_r = np.linalg.norm(cur)
            if cur_r < 1e-6:
                new = cur + desired            # full authority near the base
            else:
                r_hat = cur / cur_r
                radial_mag = float(desired @ r_hat)
                radial = radial_mag * r_hat
                tang = desired - radial
                if radial_mag > 0.0:           # pushing OUTWARD: attenuate
                    atten = max(0.0, 1.0 - (cur_r / self.reach) ** self.atten_power)
                    radial = radial * atten
                new = cur + radial + tang
            nr = np.linalg.norm(new)
            if nr > self.reach:                # hard reach clip
                new = new * (self.reach / (nr + EPS))
            out[i] = new
        return out


class PointDrag(Body):
    """Holonomic point robot with motion inertia: tip deltas are EMA-smoothed
    (it cannot turn sharply), then rescaled per axis so the endpoint is exact.
    A divergence mechanism DISTINCT from the arms' radial constraint and from a
    clean point robot -> gives the coherence axis a genuinely different point."""
    name = "point_drag"
    reach = np.inf

    def __init__(self, beta=0.6):
        self.beta = float(beta)

    def realize(self, planned_pos):
        P = np.asarray(planned_pos, dtype=float)
        d = np.diff(P, axis=0)
        f = np.empty_like(d)
        f[0] = d[0]
        for i in range(1, len(d)):
            f[i] = self.beta * f[i - 1] + (1 - self.beta) * d[i]
        s_d, s_f = d.sum(0), f.sum(0)               # endpoint-preserving rescale
        f = f * np.where(np.abs(s_f) > 1e-9, s_d / (s_f + 1e-12), 1.0)
        return np.concatenate([P[:1], P[:1] + np.cumsum(f, axis=0)], axis=0)


class PointPhase(Body):
    """Point robot whose LATERAL detour phase (low-freq bins 1,2 of the
    perpendicular-to-target component) is rotated by a fixed angle theta, with
    the along-target (progress/endpoint) component left exact. A CONTROLLED knob
    on coherence-with-the-arms: theta=0 is a clean point (high coherence);
    larger theta shifts the pinned lateral phase (lower coherence). Used to test
    G-predict (does coherence predict transfer gain?) with a real coherence
    spread, which realistic bodies don't give (the obstacle detour is physically
    forced, hence shared)."""

    def __init__(self, name, theta_deg, bins=(1, 2)):
        self.name = name
        self.theta = np.radians(theta_deg)
        self.bins = bins

    def realize(self, planned_pos):
        P = np.asarray(planned_pos, dtype=float)
        p0 = P[0]
        rel = P - p0
        d = P[-1] - p0
        nd = np.linalg.norm(d)
        if nd < 1e-9:
            return P.copy()
        u = d / nd
        v = np.array([-u[1], u[0]])
        prog = rel @ u
        lat = rel @ v
        spec = np.fft.rfft(lat)
        for om in self.bins:
            if om < len(spec):
                spec[om] = spec[om] * np.exp(1j * self.theta)
        lat2 = np.fft.irfft(spec, n=len(lat))
        return p0 + np.outer(prog, u) + np.outer(lat2, v)


def make_bodies():
    """Set A = {arm2,arm3,arm4}. Held-out ladder spans the coherence axis:
    arm5 (biggest reach, closest to the arm family -> high coherence), arm_short
    (same family but tightly constrained -> mid), point (unconstrained drone ->
    low), point_drag (distinct inertia mechanism -> low/other). Reaches graded;
    targets at radius <= ~1.7 so all arms reach the goal and divergence shows on
    the detour."""
    return {
        "arm2": PlanarArm("arm2", reach=1.8),
        "arm3": PlanarArm("arm3", reach=2.0),
        "arm4": PlanarArm("arm4", reach=2.2),
        "arm5": PlanarArm("arm5", reach=2.5),
        "arm_short": PlanarArm("arm_short", reach=1.7),
        "point": PointRobot(),
        "point_drag": PointDrag(),
        "point_phase0": PointPhase("point_phase0", 0.0),
        "point_phase15": PointPhase("point_phase15", 15.0),
        "point_phase30": PointPhase("point_phase30", 30.0),
        "point_phase45": PointPhase("point_phase45", 45.0),
    }
