"""Angle-bias attribution + pin-correction test on real anchors (2026-08-27, Denis:
'every real trajectory is off by degrees -> colliding; understand why; compare scratch;
then correct the approach angle THROUGH THE PIN').

--mode scratch : plain scratch3 chunks at the same real anchors (no pin, model noise).
                 Same heading bias as gmsig3 -> shared data/perception cause; none -> pin path.
--mode correct : gmsig3; at each anchor measure the fan's heading error dtheta vs the real
                 continuation, then re-serve with the pin command ROTATED by -dtheta
                 (and fixed +/-15 deg doses) at sigma=0. Reports correction gain
                 (achieved rotation / commanded) and residual heading error — the
                 'your angle of approach is incorrect, try xyz' channel, quantified.

  python real_angle_fix.py --mode scratch --ckpt <scratch> --out <npz>
  SNMVP_HEAD=1 ... python real_angle_fix.py --mode correct --ckpt <gmsig3> --pin-u <U> --out <npz>
"""
import argparse
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)

H, AD = 50, 32
PROMPTS = {"left": "go through the gate on the left and hover over the stuffed animal",
           "right": "go through the gate on the right and hover over the stuffed animal"}
APER = {"left": (np.array([0.65, 1.05]), np.array([1.18, 0.45])),
        "right": (np.array([0.195, -1.348]), np.array([0.924, -0.952]))}


def classify(P):
    for side, (a, b) in APER.items():
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([t[1], -t[0]])
        d = (P[:, :2] - a) @ n
        s = (P[:, :2] - a) @ t
        ins = (s > 0) & (s < np.linalg.norm(b - a))
        if np.any((np.sign(d[1:]) != np.sign(d[:-1])) & ins[1:]):
            return side
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "correct"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--eps-per-side", type=int, default=8)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "correct":
        import joint_head
        joint_head.enable_head(a.pin_u)
        from sigma_phase_probe import gmm_params
    from PIL import Image
    import gate_ctx_common as gc
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    cfg = C.get_config("pi0_gate")
    ns = gc.pad_norm_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = PC.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    U = np.load(a.pin_u).astype(np.float32) if a.mode == "correct" else None
    NS = json.load(open(os.path.expanduser(
        "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
    amean, astd = np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(0)

    def proj(acts):
        ch = np.zeros((H, AD), np.float32)
        ch[:len(acts), :7] = (acts - amean) / (astd + 1e-6)
        return ch.reshape(-1) @ U

    def gen(obs, c=None):
        kw = {}
        if c is not None:
            g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
            kw = dict(noise=(g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32),
                      snmvp_sigma=0.0)
        return np.asarray(policy.infer(obs, **kw)["actions"], np.float32)[:H, :7]

    def heading(v):
        return np.arctan2(v[1], v[0])

    def rotz(acts, th):
        o = acts.copy()
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
        o[:, :2] = acts[:, :2] @ R.T
        return o

    out = []
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
        for t in range(10, len(st) - 30, a.stride):
            obs = {"observation/image": r224(d["image"][t]),
                   "observation/wrist_image": r224(d["wrist"][t]),
                   "observation/state": st[t], "prompt": PROMPTS[side]}
            m = min(25, len(st) - 1 - t)
            hr = heading(st[t + m, :2] - st[t, :2])
            rec = {"side": side, "e": e, "t": t, "frac": t / len(st), "hr": float(hr),
                   "anchor": st[t, :3].copy()}
            if a.mode == "scratch":
                acts = gen(obs)
                rec["h0"] = float(heading(np.sum(acts[:m, :2], axis=0)))
                rec["traj"] = st[t, :3] + np.cumsum(acts[:, :3], axis=0)
            else:
                from sigma_phase_probe import gmm_params
                w, mu, _ = gmm_params(policy, [obs])
                c0 = mu[0, int(w[0].argmax())]
                acts0 = gen(obs, c0)
                h0 = heading(np.sum(acts0[:m, :2], axis=0))
                dth = float(np.angle(np.exp(1j * (h0 - hr))))
                rec["h0"], rec["dth"] = float(h0), dth
                # corrected command: the model's own actions rotated by -dth, reprojected
                acts_fix = gen(obs, proj(rotz(acts0, -dth)))
                rec["hfix"] = float(heading(np.sum(acts_fix[:m, :2], axis=0)))
                rec["traj0"] = st[t, :3] + np.cumsum(acts0[:, :3], axis=0)
                rec["trajfix"] = st[t, :3] + np.cumsum(acts_fix[:, :3], axis=0)
                # dose-response: command +/-15 deg rotations of the model's own plan,
                # both xy-only (original verb) and with a nose ramp (theta spread over the
                # first 10 steps of dyaw) — testing whether turning the nose closes the
                # 0.76 gain gap (2026-09-01)
                for tag, th in [("p15", np.radians(15)), ("m15", np.radians(-15))]:
                    av = gen(obs, proj(rotz(acts0, th)))
                    rec[f"h{tag}"] = float(heading(np.sum(av[:m, :2], axis=0)))
                    ar = rotz(acts0, th)
                    ar[:10, 3] += th / 10.0
                    av2 = gen(obs, proj(ar))
                    rec[f"h{tag}n"] = float(heading(np.sum(av2[:m, :2], axis=0)))
            out.append(rec)
        print(f"ep{e:03d} [{side}] done", flush=True)

    np.savez(a.out, meta=json.dumps([{k: v for k, v in r.items()
                                      if not isinstance(v, np.ndarray)} for r in out]),
             **{f"arr_{i}_{k}": v for i, r in enumerate(out)
                for k, v in r.items() if isinstance(v, np.ndarray)})
    print(f"saved -> {a.out}  ({len(out)} anchors)")


if __name__ == "__main__":
    main()
