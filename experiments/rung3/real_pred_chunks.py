"""Predicted-from-real chunk fans (2026-08-27): at anchors along REAL episodes, run the
full serve path on the real observation (GMM head argmax-c -> pinned noise -> flow at the
calibrated sigma), integrate the generated 50-step chunk from the anchor, and save
everything for the cloud viewer: real paths, predicted chunks, sigma* per anchor, and the
real continuation is the path itself. Episodes are classified left/right by which aperture
the real flight crosses.

  SNMVP_HEAD=1 ... python real_pred_chunks.py --ckpt <ck> --pin-u <U> --out <npz>
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
PROMPTS = {"left": "go through the gate on the left and hover over the stuffed animal",
           "right": "go through the gate on the right and hover over the stuffed animal"}
APER = {"left": (np.array([0.65, 1.05]), np.array([1.18, 0.45]), +1),
        "right": (np.array([0.195, -1.348]), np.array([0.924, -0.952]), -1)}


def classify(P):
    for side, (a, b, sign) in APER.items():
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([t[1], -t[0]])
        d = (P[:, :2] - a) @ n
        s = (P[:, :2] - a) @ t
        ins = (s > 0) & (s < np.linalg.norm(b - a))
        cross = np.where((np.sign(d[1:]) != np.sign(d[:-1])) & ins[1:])[0]
        if len(cross):
            return side
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--sigma-map", default=f"{RD}/sigma_map_gmsig3.json")
    ap.add_argument("--eps-per-side", type=int, default=8)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt,
                                      norm_stats=_nz.load(a.norm))
    U = np.load(a.pin_u).astype(np.float32)
    sm = json.load(open(a.sigma_map))
    xs, ys, cap = np.asarray(sm["sig_star"]), np.asarray(sm["sig_serve"]), float(sm["cap"])
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(0)

    out = {"left": {"paths": [], "chunks": [], "sig": []},
           "right": {"paths": [], "chunks": [], "sig": []}}
    counts = {"left": 0, "right": 0}
    for e in range(100):
        if all(counts[s] >= a.eps_per_side for s in counts):
            break
        d = np.load(f"{RD}/data_gate_real/ep_{e:04d}.npz", allow_pickle=True)
        st = d["state"].astype(np.float32)
        side = classify(st)
        if side is None or counts[side] >= a.eps_per_side:
            continue
        counts[side] += 1
        out[side]["paths"].append(st[:, :3].copy())
        for t in range(10, len(st) - 5, a.stride):
            obs = {"observation/image": r224(d["image"][t]),
                   "observation/wrist_image": r224(d["wrist"][t]),
                   "observation/state": st[t], "prompt": PROMPTS[side]}
            w, mu, sig = gmm_params(policy, [obs])
            j = int(w[0].argmax())
            c = mu[0, j]
            sstar = float(np.linalg.norm(sig[0, j]))
            sig_serve = float(np.clip(np.interp(sstar, xs, ys), 0.0, cap))
            g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
            noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
            res = policy.infer(obs, noise=noise, snmvp_sigma=sig_serve)
            act = np.asarray(res["actions"], np.float32)[:H, :3]
            traj = st[t, :3] + np.cumsum(act, axis=0)
            out[side]["chunks"].append(np.concatenate([st[t, :3][None], traj]))
            out[side]["sig"].append(sstar)
        print(f"ep{e:03d} [{side}] anchors done", flush=True)

    np.savez(a.out,
             **{f"{s}_path_{i}": p for s in out for i, p in enumerate(out[s]["paths"])},
             **{f"{s}_chunk_{i}": c for s in out for i, c in enumerate(out[s]["chunks"])},
             left_sig=np.array(out["left"]["sig"]), right_sig=np.array(out["right"]["sig"]))
    print(f"saved -> {a.out}  (left {counts['left']} eps/{len(out['left']['chunks'])} chunks, "
          f"right {counts['right']}/{len(out['right']['chunks'])})")


if __name__ == "__main__":
    main()
