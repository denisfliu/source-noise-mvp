"""How far a flight is from the minimum-snap trajectory through its own path (2026-09-24, Denis: "check how closely
it looks like a min snap trajectory"). Minimum-snap (Mellinger & Kumar 2011) is the standard smooth quadrotor
reference: for fixed waypoints and times, the curve minimizing the integral of squared snap is the degree-7 natural
spline (4th-6th derivatives zero at the ends). For every flight we take its own positions every KNOT seconds as
waypoints, fit that spline, and report the RMS distance between the flight and the spline: how much the flight
wiggles or staircases between points it visits anyway. 0 = the flight IS a minimum-snap trajectory.
Rates: sim and demos at 10 Hz; hardware logs update at 2 Hz, so every source is evaluated at 2 Hz as well."""
import glob, sys, json, numpy as np
from scipy.interpolate import make_interp_spline
sys.path.insert(0, "/home/dfliu/code/source-noise-mvp/experiments/rung3")
import realism as R
KNOT = 1.0
BC = ([(4, 0.0), (5, 0.0), (6, 0.0)], [(4, 0.0), (5, 0.0), (6, 0.0)])

def dev(P, dt):
    sp = np.linalg.norm(np.diff(P, axis=0), axis=1) / dt
    mv = np.where(sp > 0.05)[0]
    if len(mv) < 3: return np.nan
    P = P[mv[0]:mv[-1] + 2]; t = np.arange(len(P)) * dt
    step = max(1, int(round(KNOT / dt))); k = np.arange(0, len(P), step)
    if k[-1] != len(P) - 1: k = np.r_[k, len(P) - 1]
    if len(k) < 9: return np.nan
    S = make_interp_spline(t[k], P[k], k=7, bc_type=BC)
    return float(np.sqrt(np.mean(np.sum((S(t) - P) ** 2, axis=1))))

def cell(Ps, dt):
    d = np.array([dev(P, dt) for P in Ps]); d = d[np.isfinite(d)]
    return dict(n=len(d), median_cm=float(np.median(d) * 100), p90_cm=float(np.percentile(d, 90) * 100))

demos = R.load_demos(); pilot = [P for ps in demos["real"].values() for P in ps]
load = lambda pre: [np.load(f)[:, :3].astype(np.float64) for f in sorted(glob.glob(f"/home/dfliu/ctxrun/traj_{pre}_[0-9]*.npy"))]
d5 = lambda Ps: [P[::5] for P in Ps]
out = {}
out["10 Hz: pilot demos"] = cell(pilot, 0.1)
for lab, pre in [("ours, left", "armrealonly_apc25_left"), ("ours, right", "armrealonly_apc25_right"), ("pi0, left", "armscrreal_apc25_left"), ("pi0, right", "armscrreal_apc25_right"),
                 ("ours, relocated gates", "ro25mgall"), ("ours, orbit 1.3 m", "ro25_orbit_wide"), ("ours, figure-eight A", "ro25_fig8_denis3")]:
    out["10 Hz sim: " + lab] = cell(load(pre), 0.1)
out["2 Hz: pilot demos"] = cell(d5(pilot), 0.5)
out["2 Hz sim: ours, left"] = cell(d5(load("armrealonly_apc25_left")), 0.5)
out["2 Hz sim: pi0, left"] = cell(d5(load("armscrreal_apc25_left")), 0.5)
out["2 Hz hardware: ours, left"] = cell(d5(load("hwro_left")), 0.5)
out["2 Hz hardware: pi0, left"] = cell(d5(load("hwpi0_left")), 0.5)
for k, v in out.items(): print(f"{k:36s} n={v['n']:3d}  RMS distance to min-snap {v['median_cm']:.1f} cm (p90 {v['p90_cm']:.1f})")
json.dump(out, open("/home/dfliu/ctxrun/minsnap_realism.json", "w"), indent=1)
