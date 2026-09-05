"""SDEdit cross-domain probe (2026-09-03): the real_pin_follow_probe.py protocol applied to the
SDEdit baseline. Same frames (REAL episodes vs synth controls, seed 0), same metrics, but the
unpinned policy is guided the SDEdit way: x_{t0} = t0 z + (1-t0) a_guide, guide = the frame's
own demo continuation (in-distribution) or that chunk pushed +/- 1 cstd along one basis dim
(contradictory). Reports, per domain and t0: follow error ||U^T a_hat - c|| / cstd, how far the
output moved from the guide (normalized L2 per step), and per-dim honoring of the push.
t0 = 1.0 is the plain policy (no guide) for reference.

  python sdedit_real_probe.py --ckpt <scratch ckpt> [--t0 0.3 0.5 0.7 1.0] [--frames 60]
"""
import argparse
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import gate_ctx_common as gc

H, AD = 50, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--t0", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from PIL import Image
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    cfg = C.get_config("pi0_gate")
    raw = _nz.load(a.norm)
    policy = PC.create_trained_policy(cfg, a.ckpt, norm_stats=gc.pad_norm_stats(raw, cfg.model.action_dim))
    U = np.load(a.pin_u).astype(np.float32); K = U.shape[1]
    amean = np.asarray(raw["actions"].mean[:7], np.float32); astd = np.asarray(raw["actions"].std[:7], np.float32)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)

    def chunk(ac, t):
        ch = np.zeros((H, AD), np.float32); m = min(H, len(ac) - t)
        ch[:m, :4] = (ac[t:t + m, :4] - amean[:4]) / (astd[:4] + 1e-6)
        return ch

    rng = np.random.default_rng(0)
    Cs = []
    for e in range(0, 200, 10):
        d = np.load(f"{RD}/data_gate_synth3/ep_{e:04d}.npz", allow_pickle=True)
        ac = d["action"].astype(np.float32)
        Cs.extend(chunk(ac, t).reshape(-1) @ U for t in range(0, len(ac) - H, 25))
    cstd = np.std(np.stack(Cs), axis=0)

    def serve(obs, guide, t0):
        z = rng.standard_normal((H, AD)).astype(np.float32)
        if t0 >= 1.0:
            out = policy.infer(obs, noise=z)
        else:
            out = policy.infer(obs, noise=(t0 * z + (1 - t0) * guide).astype(np.float32), snmvp_t_start=float(t0))
        act = np.asarray(out["actions"], np.float32)[:H]
        ch = np.zeros((H, AD), np.float32)
        ch[:, :4] = (act[:, :4] - amean[:4]) / (astd[:4] + 1e-6)
        return ch

    frames = {}
    for dom, ddir, eps in [("real", "data_gate_real", range(0, 100)), ("synth", "data_gate_synth3", range(0, 200))]:
        picked = rng.choice(list(eps), a.frames // 3 + 1, replace=False); rows = []
        for e in picked:
            d = np.load(f"{RD}/{ddir}/ep_{int(e):04d}.npz", allow_pickle=True)
            ac, st = d["action"].astype(np.float32), d["state"].astype(np.float32); T = len(ac)
            for frac in (0.15, 0.45, 0.75):
                if len(rows) >= a.frames:
                    break
                t = int(frac * (T - H - 1))
                obs = {"observation/image": r224(d["image"][t]), "observation/wrist_image": r224(d["wrist"][t]),
                       "observation/state": st[t], "prompt": "fly through the gate and hover over the stuffed animal"}
                rows.append((obs, chunk(ac, t)))
        frames[dom] = rows
    print(f"frames: real {len(frames['real'])}, synth {len(frames['synth'])}; cstd L2 {np.linalg.norm(cstd):.2f}")
    print(f"{'t0':>4s} {'dom':>5s} {'follow/cstd':>11s} {'dev from guide':>14s} {'push honored (err/push)':>24s}")
    summary = {}
    for t0 in a.t0:
        for dom in ("real", "synth"):
            eo, dev, pushed = [], [], []
            for obs, g in frames[dom]:
                c0 = g.reshape(-1) @ U
                out = serve(obs, g, t0)
                eo.append(np.abs(out.reshape(-1) @ U - c0))
                dev.append(np.linalg.norm((out - g)[:, :4], axis=1).mean())
                if t0 < 1.0:
                    k = int(rng.integers(K)); sgn = 1 if rng.random() < 0.5 else -1
                    gp = g + (sgn * cstd[k] * U[:, k]).reshape(H, AD)
                    cp = gp.reshape(-1) @ U
                    outp = serve(obs, gp, t0)
                    pushed.append(abs((outp.reshape(-1) @ U - cp)[k]) / cstd[k])
            eo = np.stack(eo)
            fo = np.linalg.norm(eo.mean(0)) / np.linalg.norm(cstd)
            summary[(t0, dom)] = (float(fo), float(np.mean(dev)), float(np.mean(pushed)) if pushed else float("nan"))
            print(f"{t0:4.1f} {dom:>5s} {fo:11.3f} {np.mean(dev):14.3f} {summary[(t0, dom)][2]:24.3f}", flush=True)
    if a.out:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in summary.items()}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
