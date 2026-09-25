"""Realism on flown trajectories (2026-09-24): pilot demos vs hardware flights of ours and pi0, at the rate the
hardware log actually updates (the mocap pose changes every 5th executed step -> 2 Hz). Everything, demos and sim
included, is subsampled to 2 Hz so the comparison is like for like. Features per 3 s window: mean and std of speed,
mean and max |accel|, mean |jerk|, mean turn rate of the horizontal velocity. ROC-AUC of a logistic regression,
5-fold CV grouped by flight (0.5 = indistinguishable from the pilot)."""
import glob, os, sys, json
import numpy as np
RD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, RD)
import realism as R
DT = 0.5

def dedupe2hz(P):
    """hardware: keep one row per pose update (every 5th executed step); sim/demos: every 5th row."""
    return P[::5]

def active(P):
    sp = np.linalg.norm(np.diff(P, axis=0), axis=1) / DT
    mv = np.where(sp > 0.05)[0]
    return P[:mv[-1] + 2] if len(mv) else P

def windows(P, w=6):
    v = np.diff(P, axis=0) / DT; a = np.diff(v, axis=0) / DT; j = np.diff(a, axis=0) / DT
    sp = np.linalg.norm(v, axis=1); am = np.linalg.norm(a, axis=1); jm = np.linalg.norm(j, axis=1)
    hd = np.unwrap(np.arctan2(v[:, 1], v[:, 0])); tr = np.abs(np.diff(hd)) / DT
    out = []
    for s in range(0, len(jm) - w + 1, w // 2):
        out.append([sp[s:s+w].mean(), sp[s:s+w].std(), am[s:s+w].mean(), am[s:s+w].max(), jm[s:s+w].mean(), np.median(tr[s:s+w])])
    return np.array(out)

def auc(A, B, folds=5, seed=0):
    fa = [windows(P) for P in A]; fb = [windows(P) for P in B]
    ga = [i for i, f in enumerate(fa) for _ in f]; gb = [i for i, f in enumerate(fb) for _ in f]
    X = np.vstack([np.vstack([f for f in fa if len(f)]), np.vstack([f for f in fb if len(f)])])
    y = np.r_[np.ones(len(ga)), np.zeros(len(gb))]
    g = np.r_[np.array(ga), 10000 + np.array(gb)]
    rng = np.random.default_rng(seed); ug = np.unique(g); rng.shuffle(ug); fold = {u: k % folds for k, u in enumerate(ug)}
    s = np.zeros(len(y))
    for k in range(folds):
        te = np.array([fold[u] == k for u in g]); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = R.logreg_fit((X[tr] - mu) / sd, y[tr]); s[te] = ((X[te] - mu) / sd) @ w[:-1] + w[-1]
    return R.auc_score(y, s)

def stats(Ps):
    v = [np.linalg.norm(np.diff(P, axis=0), axis=1) / DT for P in Ps]; a = [np.linalg.norm(np.diff(np.diff(P, axis=0), axis=0), axis=1) / DT**2 for P in Ps]
    return dict(n=len(Ps), v95=float(np.median([np.percentile(x, 95) for x in v])), vmean=float(np.median([x[x > 0.05].mean() for x in v])),
                a95=float(np.median([np.percentile(x, 95) for x in a])), T=float(np.median([len(P) * DT for P in Ps])))

demos = R.load_demos()
pilot = [active(dedupe2hz(P)) for ps in demos["real"].values() for P in ps]
pl_left = [active(dedupe2hz(P)) for P in demos["real"].get(0, [])]
rng = np.random.default_rng(0); idx = rng.permutation(len(pilot)); h1 = [pilot[i] for i in idx[:50]]; h2 = [pilot[i] for i in idx[50:]]
load = lambda pre: [active(dedupe2hz(np.load(f)[:, :3].astype(np.float64))) for f in sorted(glob.glob(f"/home/dfliu/ctxrun/traj_{pre}_[0-9]*.npy"))]
cells = {"hardware, ours (left gate, 5 flights)": load("hwro_left"), "hardware, pi0 (left gate, 5 flights)": load("hwpi0_left"),
         "sim, ours (left gate, 100)": load("armrealonly_apc25_left"), "sim, pi0 (left gate, 100)": load("armscrreal_apc25_left")}
res = {"pilot vs pilot": dict(auc=auc(h1, h2), **stats(pilot))}
for k, Ps in cells.items():
    res[k] = dict(auc=auc(Ps, pilot), **stats(Ps))
for k, r in res.items():
    print(f"{k:40s} n={r['n']:3d} AUC vs pilot {r['auc']:.3f}   v95 {r['v95']:.2f} m/s  moving speed {r['vmean']:.2f}  a95 {r['a95']:.2f} m/s^2  T {r['T']:.0f} s")
json.dump(res, open("/home/dfliu/ctxrun/realism_hw.json", "w"), indent=1)
