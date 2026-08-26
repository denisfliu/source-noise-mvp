"""Zero-shot feasibility diagnostic: is the pinned instruction coordinate c domain-invariant?
If c for a given instruction is the same in sim and real (in the shared gate U-space / gate norm),
then a language-only prior fit on SIM predicts real c with no real data -> zero-shot transfer. This
measures, per shared instruction (left/right gate), the sim-vs-real gap in mean c against the
left-vs-right separation we must preserve, and reports the R^2 of a language-only prior FIT ON SIM
when evaluated on REAL chunks (the zero-shot number, at the prior level, before the flow) versus the
same prior fit on real (in-domain upper bound) and the null (overall-mean) baseline."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32
import openpi.shared.normalize as NZ
ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
U = np.load(os.path.join(RD, "pin_U_gate_k5.npy")).astype(np.float32)


def seg_to_c(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1) @ U


def load_chunks(raw):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    C, lang = [], []
    for k in sorted(meta):
        d = np.load(os.path.join(raw, k + ".npz"))
        acts = d["action"].astype(np.float32); T = len(acts)
        for t in range(0, T, 3):
            C.append(seg_to_c(acts[t:])); lang.append(meta[k]["lang"])
    return np.asarray(C, np.float32), np.asarray(lang)


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def main():
    Cr, Lr = load_chunks(os.path.join(RD, "data_gate_real"))
    Cs, Ls = load_chunks(os.path.join(RD, "data_gate_synth"))
    shared = sorted(set(Lr) & set(Ls))
    print(f"real chunks={len(Cr)} synth chunks={len(Cs)}")
    print(f"shared instructions ({len(shared)}):")
    for s in shared:
        print("   -", s)
    print()

    # per-instruction mean c in each domain
    sim_mean = {s: Cs[Ls == s].mean(0) for s in shared}
    real_mean = {s: Cr[Lr == s].mean(0) for s in shared}
    np.set_printoptions(precision=3, suppress=True)
    for s in shared:
        gap = np.linalg.norm(sim_mean[s] - real_mean[s])
        print(f"[{s[:32]:32s}] |c_sim-c_real|={gap:.3f}")
        print(f"    c_sim ={sim_mean[s]}")
        print(f"    c_real={real_mean[s]}")
    # separation we must preserve (left vs right), per domain
    if len(shared) >= 2:
        a, b = shared[0], shared[1]
        sep_sim = np.linalg.norm(sim_mean[a] - sim_mean[b])
        sep_real = np.linalg.norm(real_mean[a] - real_mean[b])
        dom_gap = 0.5 * (np.linalg.norm(sim_mean[a] - real_mean[a]) + np.linalg.norm(sim_mean[b] - real_mean[b]))
        print(f"\nleft-vs-right separation: sim={sep_sim:.3f} real={sep_real:.3f}")
        print(f"mean sim-vs-real domain gap: {dom_gap:.3f}")
        print(f"RATIO domain_gap/separation = {dom_gap / (sep_real + 1e-9):.3f}  (<<1 => zero-shot feasible)")

    # language-only prior R^2 on REAL chunks
    mask = np.isin(Lr, shared)
    Cr_s, Lr_s = Cr[mask], Lr[mask]
    pred_zs = np.stack([sim_mean[l] for l in Lr_s])    # fit on SIM  -> zero-shot
    pred_id = np.stack([real_mean[l] for l in Lr_s])   # fit on REAL -> in-domain upper bound
    print(f"\nlanguage-only prior, evaluated on REAL chunks (n={len(Cr_s)}):")
    print(f"  fit on SIM  (ZERO-SHOT)      subspace R^2 = {r2(pred_zs, Cr_s):.3f}")
    print(f"  fit on REAL (in-domain UB)   subspace R^2 = {r2(pred_id, Cr_s):.3f}")
    print(f"  null (overall real mean)     subspace R^2 = 0.000  (by definition)")
    print("ZS_DIAG_DONE")


if __name__ == "__main__":
    main()
