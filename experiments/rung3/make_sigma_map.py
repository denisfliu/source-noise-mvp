"""Build the sigma*-to-sigma_serve calibration map for a sigma-conditioned MDN flow.

Input: sigma_phase_probe --save rows (per-task arrays of [frac, is_stop, err, sig]).
Output json for SNMVP_SIGMA_MAP: piecewise-linear sig_star -> sig_serve where sig_serve is the
median command error in TRAIN-NOISE UNITS (fractions of ||c-std||: the flow was trained with
c + sigma*cstd*eps, so E||corruption|| ~= sigma*||cstd||_2 -> sigma ~= err/||cstd||_2), capped
at the training PIN_NOISE so serve stays in-distribution. cstd is computed from the demo oracle
c distribution on the same stride-8 rows as everything else. Stamped with basis sha.

  python3 make_sigma_map.py --rows sigrows_gmsig.npz --pin-u pin_U_mh16.npy \
      --cap 1.5 --out sigma_map_gmsig.json
"""
import argparse
import hashlib
import json
import os

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
H, AD = 50, 32
STRIDE = 8
NS = json.load(open(os.path.expanduser(
    "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
AMEAN, ASTD = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)


def seg_to_Y(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32)
    ch[:, :r] = (seg - AMEAN[:r]) / (ASTD[:r] + 1e-6)
    return ch.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--cap", type=float, default=1.5)
    ap.add_argument("--nbins", type=int, default=8)
    ap.add_argument("--data-dir", default="data_gate_synth",
                    help="demo dir for the c-std normalizer — use the CHECKPOINT'S OWN training "
                         "mirrors (data_gate_synth3 for gmsig3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    U = np.load(a.pin_u).astype(np.float32)
    Cs = []
    for e in range(200):
        d = np.load(f"{RD}/{a.data_dir}/ep_{e:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32)
        for t in range(0, len(ac), STRIDE):
            Cs.append(seg_to_Y(ac[t:]) @ U)
    cstd_norm = float(np.linalg.norm(np.std(np.stack(Cs), axis=0)))

    z = np.load(a.rows)
    rr = np.concatenate([z[k] for k in z.files], 0)      # (n, 4): frac, is_stop, err, sig
    sig, err = rr[:, 3], rr[:, 2]
    qs = np.quantile(sig, np.linspace(0, 1, a.nbins + 1))
    xs, ys = [], []
    for i in range(a.nbins):
        m = (sig >= qs[i]) & (sig <= qs[i + 1])
        if m.sum() < 5:
            continue
        xs.append(float(np.median(sig[m])))
        ys.append(float(np.clip(np.median(err[m]) / cstd_norm, 0.0, a.cap)))
    ys = np.maximum.accumulate(ys).tolist()              # enforce monotone
    out = {"sig_star": xs, "sig_serve": ys, "cap": a.cap, "cstd_norm": cstd_norm,
           "pin_u": os.path.abspath(a.pin_u),
           "pin_u_sha": hashlib.sha256(open(a.pin_u, "rb").read()).hexdigest(),
           "rows": os.path.abspath(a.rows), "n_rows": int(len(rr))}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: cstd_norm={cstd_norm:.3f}")
    for x, y in zip(xs, ys):
        print(f"  sig*={x:6.2f} -> sigma_serve={y:5.3f}")


if __name__ == "__main__":
    main()
