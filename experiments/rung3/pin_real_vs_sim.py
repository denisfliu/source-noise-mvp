"""Pin properties on REAL vs SIM chunks (Denis, 2026-08-22): does the current pin transfer?

Four chunk-space measurements (no model, CPU), all on the zero-pad train convention, real
(data_gate_real: 50 L + 50 R teleop) vs synth L/R (data_gate_synth eps 100-199):

  1. CAPTURE: within-task variance fraction in span(U) by phase — is the pin as EXPRESSIVE on
     real flying as on sim? (mh16 is hand-built displacement functionals — domain-free by
     construction, but the real action distribution could still live off-span.)
  2. c-DISTRIBUTION ALIGNMENT: per-component real-vs-synth standardized mean shift and std
     ratio, split early/late — which command channels mean the same thing across domains.
  3. SUBSPACE: principal angles between real-fit and synth-fit within-task PCA(16) (the
     data-driven counterpart of the pin span), plus how much of each domain's variance the
     OTHER domain's PCA span captures — the symmetric transfer number. Box K=5 reference:
     RRR real-vs-synth shared only ~1-2 directions, yet the synth basis served real best.
  4. MATCHED-STATE ORACLE GAP BY BAND: at position+heading-matched real/sim states (the
     sim_real_c_probe matching rule), how far apart are the ORACLE commands per mh16 horizon
     band (h6/h12/h25/h50)? The box measured the gap is behavior-dominated at K=5; the band
     decomposition says WHICH timescales of behavior diverge — short bands transferring while
     long bands diverge would argue for band-weighted trust in any real deployment.

  python3 pin_real_vs_sim.py [--u pin_U_mh16.npy]
"""
import argparse
import json
import os

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
H, AD = 50, 32
STRIDE = 4
NS = json.load(open(os.path.expanduser(
    "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
AMEAN, ASTD = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
SYNTH = {"left": range(100, 150), "right": range(150, 200)}
REAL = {"left": range(0, 50), "right": range(50, 100)}
BANDS = {"h6": range(0, 4), "h12": range(4, 8), "h25": range(8, 12), "h50": range(12, 16)}


def seg_to_Y(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - AMEAN[:r]) / (ASTD[:r] + 1e-6)
    return ch.reshape(-1)


def load(domain_dir, eps):
    Ys, frac, stop, pos, hdg, ei = [], [], [], [], [], []
    for e in eps:
        d = np.load(f"{RD}/{domain_dir}/ep_{e:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32)
        st = d["state"][:, :3].astype(np.float32)
        T = len(ac)
        for t in range(0, T, STRIDE):
            Ys.append(seg_to_Y(ac[t:]))
            frac.append(t / T)
            stop.append(t > T - H)
            pos.append(st[t])
            v = ac[t:t + 10, :3].sum(0)
            hdg.append(v / (np.linalg.norm(v) + 1e-9))
            ei.append(e)
    return (np.stack(Ys), np.array(frac), np.array(stop), np.stack(pos), np.stack(hdg),
            np.array(ei))


def capture(Y, U):
    Yc = Y - Y.mean(0)
    return float(((Yc @ U) ** 2).sum() / ((Yc ** 2).sum() + 1e-9))


def pca(Y, k):
    Yc = Y - Y.mean(0)
    w, V = np.linalg.eigh(np.cov(Yc.T))
    return V[:, ::-1][:, :k].astype(np.float32)


def pangles(A, B):
    s = np.linalg.svd(np.linalg.qr(A)[0].T @ np.linalg.qr(B)[0], compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", default=f"{RD}/pin_U_mh16.npy")
    a = ap.parse_args()
    U = np.load(a.u).astype(np.float32)
    K = U.shape[1]
    Uf = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32)

    D = {}
    for dom, dird, tasks in (("synth", "data_gate_synth", SYNTH), ("real", "data_gate_real", REAL)):
        for task, eps in tasks.items():
            D[dom, task] = load(dird, eps)

    print(f"== 1. CAPTURE by phase, {os.path.basename(a.u)} (and flat K=5) ==")
    print(f"{'domain/task':14s} {'n':>5s}  {'[0,.5)':>7s} {'[.5,.75)':>8s} {'[.75,1]':>8s} "
          f"{'stop':>7s}   {'flatK5 stop':>11s}")
    for (dom, task), (Y, frac, stop, *_ ) in D.items():
        cells = [capture(Y[(frac >= lo) & (frac < hi)], U) for lo, hi in
                 ((0, .5), (.5, .75), (.75, 1.01))]
        print(f"{dom+'/'+task:14s} {len(Y):5d}  " + " ".join(f"{c:7.3f}" for c in cells)
              + f" {capture(Y[stop], U):7.3f}   {capture(Y[stop], Uf):11.3f}")

    print(f"\n== 2. c-distribution alignment (real vs synth, per component; early frac<0.5 / "
          f"late frac>0.7) ==")
    for task in ("left", "right"):
        Ys, fs = D['synth', task][0], D['synth', task][1]
        Yr, fr = D['real', task][0], D['real', task][1]
        for label, ms, mr in (("early", fs < 0.5, fr < 0.5), ("late", fs > 0.7, fr > 0.7)):
            cs, cr = Ys[ms] @ U, Yr[mr] @ U
            shift = np.abs(cr.mean(0) - cs.mean(0)) / (cs.std(0) + 1e-6)
            ratio = cr.std(0) / (cs.std(0) + 1e-6)
            bs = {b: (float(shift[list(ix)].mean()), float(ratio[list(ix)].mean()))
                  for b, ix in BANDS.items()}
            print(f"  {task:5s} {label:5s} |mean shift|/std per band: "
                  + "  ".join(f"{b}={v[0]:.2f}({v[1]:.2f}x)" for b, v in bs.items()))

    print(f"\n== 3. Subspace transfer, within-task PCA(16) ==")
    Ysl = np.vstack([D['synth', t][0] - D['synth', t][0].mean(0) for t in ('left', 'right')])
    Yrl = np.vstack([D['real', t][0] - D['real', t][0].mean(0) for t in ('left', 'right')])
    Ps, Pr = pca(Ysl, 16), pca(Yrl, 16)
    ang = pangles(Ps, Pr)
    print(f"  principal angles synthPCA16 vs realPCA16 (deg): {np.round(ang, 1)}")
    print(f"  synth var captured by: own PCA16 {capture(Ysl, Ps):.3f} | real PCA16 "
          f"{capture(Ysl, Pr):.3f} | mh16 {capture(Ysl, U):.3f}")
    print(f"  real  var captured by: own PCA16 {capture(Yrl, Pr):.3f} | synth PCA16 "
          f"{capture(Yrl, Ps):.3f} | mh16 {capture(Yrl, U):.3f}")

    print(f"\n== 4. Matched-state oracle gap by mh16 band (pos<0.35m + heading dot>0.3) ==")
    rng = np.random.default_rng(0)
    for task in ("left", "right"):
        Ys, _, _, ps, hs, _ = D['synth', task]
        Yr, _, _, pr, hr, _ = D['real', task]
        cs_all, cr_all = Ys @ U, Yr @ U
        idx = rng.permutation(len(Yr))[:400]
        gaps, coss, n = {b: [] for b in BANDS}, [], 0
        for i in idx:
            d2 = np.linalg.norm(ps - pr[i], axis=1)
            ok = (d2 < 0.35) & ((hs @ hr[i]) > 0.3)
            if not ok.any():
                continue
            j = np.argmin(np.where(ok, d2, np.inf))
            n += 1
            dc = cr_all[i] - cs_all[j]
            coss.append(float(np.dot(cr_all[i], cs_all[j]) /
                              (np.linalg.norm(cr_all[i]) * np.linalg.norm(cs_all[j]) + 1e-9)))
            for b, ix in BANDS.items():
                sd = cs_all[:, list(ix)].std(0).mean()
                gaps[b].append(float(np.abs(dc[list(ix)]).mean() / (sd + 1e-6)))
        print(f"  {task}: matched {n}/400  oracle cos {np.mean(coss):.2f}  gap/std by band: "
              + "  ".join(f"{b}={np.mean(g):.2f}" for b, g in gaps.items()))


if __name__ == "__main__":
    main()
