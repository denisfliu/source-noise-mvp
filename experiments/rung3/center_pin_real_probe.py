"""The CENTER-gate pin on REAL observations (Denis, 2026-08-22): no real center demos exist —
prompt the head with the center task on real frames from real-episode STARTS/early flight
(before any task commitment; the scene observation is task-agnostic there) and compare the
commanded c against the sim reference. Zero-shot task-command synthesis across the domain gap.

Per frame (prompts CFL and CFR): argmax-component (mu*, sigma*, pi), and mu* DECODED to its
implied 50-step path from the frame's own position (chunk = U mu* -> denorm -> cumsum) — the
literal shape of the commanded coarse movement, viewable over the scene cloud.

Groups: real early frames (real L+R eps, frac<0.25) | synth center-demo early frames (with
their oracle c). Saved to center_pin_real.npz for the viewer; prints cos/err vs the matched
sim command, sigma* levels, and the decoded initial heading vs the demo heading fan.

  SNMVP_HEAD_GMM=1 ... python center_pin_real_probe.py --ckpt <gmsig ckpt> --pin-u pin_U_mh16.npy
"""
import argparse
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import joint_head
from sigma_phase_probe import gmm_params

H, AD = 50, 32
PROMPTS = {"cfl": joint_head.PROMPTS["center_from_left"],
           "cfr": joint_head.PROMPTS["center_from_right"]}
NS = json.load(open(os.path.expanduser(
    "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
AMEAN, ASTD = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)


def decode_path(U, c, pos):
    ch = (U @ c).reshape(H, AD)[:, :7]
    raw = ch * (ASTD + 1e-6) + AMEAN
    return pos[None, :] + np.cumsum(raw[:, :3], 0)


def frames(domain_dir, eps, frac_max, stride, rng, per_ep):
    out = []
    for e in eps:
        d = np.load(f"{RD}/{domain_dir}/ep_{e:04d}.npz", allow_pickle=True)
        T = len(d["state"])
        ts = [t for t in range(0, T - 5, stride) if t / T < frac_max][:per_ep]
        for t in ts:
            out.append((d, e, t))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--n", type=int, default=60, help="frames per group")
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.training.config as C
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt)
    U = np.load(a.pin_u).astype(np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(0)

    groups = {
        "real": frames("data_gate_real", range(0, 100), 0.25, 8, rng, 3)[:a.n],
        "synth": frames("data_gate_synth", range(0, 100), 0.25, 8, rng, 2)[:a.n],
    }

    def oracle(d, t):
        ac = d["action"].astype(np.float32)
        ch = np.zeros((H, AD), np.float32)
        m = min(H, len(ac) - t)
        ch[:m, :7] = (ac[t:t + m] - AMEAN) / (ASTD + 1e-6)
        return ch.reshape(-1) @ U

    res = {}
    for gname, fs in groups.items():
        for pk, prompt in PROMPTS.items():
            rows = []
            for i in range(0, len(fs), 12):
                batch = fs[i:i + 12]
                raws = [{"observation/image": r224(d["image"][t]),
                         "observation/wrist_image": r224(d["wrist"][t]),
                         "observation/state": d["state"][t].astype(np.float32),
                         "prompt": prompt} for d, e, t in batch]
                w, mu, logsig = gmm_params(policy, raws)
                for k, (d, e, t) in enumerate(batch):
                    j = int(w[k].argmax())
                    pos = d["state"][t, :3].astype(np.float32)
                    rows.append({"pos": pos, "c": mu[k, j],
                                 "sig": float(np.linalg.norm(np.exp(logsig[k, j]))),
                                 "pi": w[k], "ep": e, "t": t,
                                 "oracle": oracle(d, t) if gname == "synth" else None})
            res[gname, pk] = rows
            print(f"[{gname}/{pk}] {len(rows)} frames", flush=True)

    # numbers: match each real frame to nearest synth frame (same prompt), compare commands
    print(f"\n{'prompt':6s} {'cos(real,sim)':>13s} {'|dc|/std':>9s} {'sig* real':>10s} "
          f"{'sig* synth':>10s} {'cos(sim,orac)':>13s}")
    save = {}
    for pk in PROMPTS:
        Rr, Rs = res["real", pk], res["synth", pk]
        cs = np.stack([r["c"] for r in Rs])
        cstd = cs.std(0)
        ps = np.stack([r["pos"] for r in Rs])
        coss, gaps = [], []
        for r in Rr:
            j = int(np.argmin(np.linalg.norm(ps - r["pos"], axis=1)))
            cr, csim = r["c"], Rs[j]["c"]
            coss.append(np.dot(cr, csim) / (np.linalg.norm(cr) * np.linalg.norm(csim) + 1e-9))
            gaps.append(np.abs(cr - csim).mean() / (cstd.mean() + 1e-6))
        oc = [np.dot(r["c"], r["oracle"]) /
              (np.linalg.norm(r["c"]) * np.linalg.norm(r["oracle"]) + 1e-9) for r in Rs]
        print(f"{pk:6s} {np.mean(coss):13.2f} {np.mean(gaps):9.2f} "
              f"{np.mean([r['sig'] for r in Rr]):10.2f} {np.mean([r['sig'] for r in Rs]):10.2f} "
              f"{np.mean(oc):13.2f}")
        for gname in ("real", "synth"):
            rows = res[gname, pk]
            save[f"{gname}_{pk}_paths"] = np.stack(
                [decode_path(U, r["c"], r["pos"]) for r in rows])
            save[f"{gname}_{pk}_sig"] = np.array([r["sig"] for r in rows])
    np.savez_compressed(f"{RD}/center_pin_real.npz", **save)
    print(f"saved decoded paths -> center_pin_real.npz")


if __name__ == "__main__":
    main()
