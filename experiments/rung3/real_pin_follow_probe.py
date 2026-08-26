"""Cross-domain pin-follow probe (2026-08-26, 'will the human-placed pin work in real?'):
the decisive offline check. For frames from REAL episodes (data_gate_real) and synth
controls (data_gate_synth3), command the flow through the pin at sigma=0 with (a) the
frame's own oracle c (in-distribution) and (b) contradictory commands (oracle +/- 1 cstd
along basis dims), then measure error-to-command ||U^T a_hat - c|| of the generated chunk.

If real-frame follow error ~= synth-frame follow error, the noise channel survives real
perception and sketch prompting transfers; per-dim errors identify which basis directions
are domain-bound. This is the CLAUDE.md wrong-invariant probe applied across domains —
whole-chunk metric, normalized units throughout.

  SNMVP_HEAD=1 ... python real_pin_follow_probe.py --ckpt <ck> --pin-u <U>
"""
import argparse
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import joint_head

H, AD = 50, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--frames", type=int, default=60, help="frames per domain")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    ns = _nz.load(a.norm)
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt, norm_stats=ns)
    U = np.load(a.pin_u).astype(np.float32)
    K = U.shape[1]
    NS = json.load(open(os.path.expanduser(
        "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
    amean, astd = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)

    def chunk_c(ac, t):
        ch = np.zeros((H, AD), np.float32)
        m = min(H, len(ac) - t)
        ch[:m, :7] = (ac[t:t + m] - amean) / (astd + 1e-6)
        return ch.reshape(-1) @ U

    rng = np.random.default_rng(0)
    # cstd from synth demo chunks (the command scale the flow was trained against)
    Cs = []
    for e in range(0, 200, 10):
        d = np.load(f"{RD}/data_gate_synth3/ep_{e:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32)
        Cs.extend(chunk_c(ac, t) for t in range(0, len(ac) - H, 25))
    cstd_dims = np.std(np.stack(Cs), axis=0)

    def serve(obs, c):
        g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
        out = policy.infer(obs, noise=noise, snmvp_sigma=0.0)
        act = np.asarray(out["actions"], np.float32)[:H]
        ch = np.zeros((H, AD), np.float32)
        ch[:, :7] = (act[:, :7] - amean) / (astd + 1e-6)
        return ch.reshape(-1) @ U

    results = {}
    for dom, ddir, eps in [("real", "data_gate_real", range(0, 100)),
                           ("synth", "data_gate_synth3", range(0, 200))]:
        errs_o, errs_p, perdim = [], [], []
        picked = rng.choice(list(eps), a.frames // 3 + 1, replace=False)
        n = 0
        for e in picked:
            d = np.load(f"{RD}/{ddir}/ep_{int(e):04d}.npz", allow_pickle=True)
            ac, st = d["action"].astype(np.float32), d["state"].astype(np.float32)
            T = len(ac)
            for frac in (0.15, 0.45, 0.75):
                if n >= a.frames:
                    break
                t = int(frac * (T - H - 1))
                obs = {"observation/image": r224(d["image"][t]),
                       "observation/wrist_image": r224(d["wrist"][t]),
                       "observation/state": st[t],
                       "prompt": "fly through the gate and hover over the stuffed animal"}
                c0 = chunk_c(ac, t)
                co = serve(obs, c0)
                errs_o.append(np.abs(co - c0))
                # contradictory: +/- 1 cstd on one random dim per trial (2 dims probed)
                for k in rng.choice(K, 2, replace=False):
                    cp = c0.copy()
                    cp[k] += cstd_dims[k] * (1 if rng.random() < 0.5 else -1)
                    cr = serve(obs, cp)
                    errs_p.append(np.abs(cr - cp))
                    perdim.append((k, float(np.abs(cr - cp)[k])))
                n += 1
        eo, ep_ = np.stack(errs_o), np.stack(errs_p)
        results[dom] = (eo, ep_, perdim)
        print(f"[{dom}] frames={n}  oracle-c follow |err| L2={np.linalg.norm(eo.mean(0)):.2f} "
              f"(/cstd {np.linalg.norm(eo.mean(0))/np.linalg.norm(cstd_dims):.3f})   "
              f"perturbed follow L2={np.linalg.norm(ep_.mean(0)):.2f} "
              f"(/cstd {np.linalg.norm(ep_.mean(0))/np.linalg.norm(cstd_dims):.3f})", flush=True)

    print("\nper-dim perturbed-command follow error (err on the pushed dim / push size), by dim:")
    print(f"{'dim':>4s} {'real':>8s} {'synth':>8s}   (low = command honored)")
    for k in range(K):
        row = []
        for dom in ("real", "synth"):
            v = [e for kk, e in results[dom][2] if kk == k]
            row.append(np.mean(v) / cstd_dims[k] if v else float("nan"))
        print(f"{k:4d} {row[0]:8.3f} {row[1]:8.3f}")
    if a.out:
        np.savez(a.out, real_o=results["real"][0], real_p=results["real"][1],
                 synth_o=results["synth"][0], synth_p=results["synth"][1],
                 cstd=cstd_dims)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
