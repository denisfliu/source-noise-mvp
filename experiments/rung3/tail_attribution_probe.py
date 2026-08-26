"""Three-link attribution of the mh16-family ENDGAME failure (Denis, 2026-08-13: the ending is
still unsolved; measure where the chain breaks instead of reaching for CFG).

Chain: (1) can the head's sampler pick the STOP mode at demo tail states (on-manifold
calibration)? (2) does it pick stop at closed-loop goal-region states (off-manifold calibration)?
(3) when a stop-mode command IS issued closed-loop, does the flight settle (execution)?

Mode references come from the demos themselves: per task, the c-space centroids (under U_mh16) of
"cruise" (frac 0.3-0.6), "decel" (0.75-0.9), "stop" (>0.9). A draw/command is assigned to the
nearest centroid. Link 1 needs the checkpoint on GPU (samples at rendered demo frames); links 2-3
are CPU (clog + trajectories).

  Link 1:  --ckpt <gen ckpt>  (16 samples per frame, tail + transit frames, left+right tasks)
  Links 2-3: always run (clog_<arm>.npy + traj_arm<arm>_*.npy)
"""
import argparse
import json
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
H, AD = 50, 32
TASKS = {"left": 2, "right": 3}
GOAL = np.array([1.525, -0.615, 1.0])


def normalizer():
    import openpi.transforms as T
    from openpi.shared.normalize import NormStats, load as load_ns
    ns = load_ns("/home/ubuntu/hf_bundle/gate-drone-pi0/assets/gate_nav")
    o = {}
    for k, s in ns.items():
        n = len(s.mean)
        if n >= AD:
            o[k] = s; continue
        p = AD - n
        ext = lambda a, f: None if a is None else np.concatenate(
            [np.asarray(a, np.float32), np.full(p, f, np.float32)])
        o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
    return T.Normalize(o, use_quantiles=False)


def mode_centroids(task, U, nrm):
    meta = json.load(open(f"{RD}/data_gate_synth/meta.json"))
    segs = {"cruise": (0.3, 0.6), "decel": (0.75, 0.9), "stop": (0.9, 1.01)}
    cs = {k: [] for k in segs}
    for k in sorted(meta):
        if meta[k]["task"] != task:
            continue
        d = np.load(f"{RD}/data_gate_synth/{k}.npz")
        ac = d["action"].astype(np.float32); T_ = len(ac)
        for t in range(0, T_ - 2, 3):
            f = t / (T_ - 1)
            for sname, (lo, hi) in segs.items():
                if lo <= f < hi:
                    ch = np.zeros((H, AD), np.float32); m = min(H, T_ - t); ch[:m, :7] = ac[t:t + m]
                    cs[sname].append(nrm({"actions": ch})["actions"].reshape(-1) @ U)
    return {k: np.mean(v, 0) for k, v in cs.items()}


def assign(c, cents):
    ds = {k: np.linalg.norm(c - v) for k, v in cents.items()}
    best = min(ds, key=ds.get)
    second = sorted(ds.values())[1]
    return best if ds[best] < 0.8 * second or ds[best] < 3.0 else "neither"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gen16")
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--ckpt", default="", help="run link 1 (GPU) if given")
    ap.add_argument("--sides", default="left,right")
    a = ap.parse_args()
    nrm = normalizer()
    U = np.load(a.pin_u).astype(np.float32)
    cents = {s: mode_centroids(TASKS[s], U, nrm) for s in a.sides.split(",")}

    # ---- link 1: on-manifold sampler calibration (GPU) ----
    if a.ckpt:
        import joint_head
        joint_head.enable_head(a.pin_u)
        os.environ["SNMVP_GEN_SAMPLES"] = "1"
        from PIL import Image
        import openpi.policies.policy_config as PC
        import openpi.training.config as C
        policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt)
        r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
        meta = json.load(open(f"{RD}/data_gate_synth/meta.json"))
        rng = np.random.default_rng(0)
        print("== link 1: on-manifold sampler mode fractions (16 draws x 8 frames per cell)")
        for side in a.sides.split(","):
            keys = [k for k in sorted(meta) if meta[k]["task"] == TASKS[side]]
            for seg, (lo, hi) in (("cruise", (0.35, 0.55)), ("stop", (0.92, 0.99))):
                frames = []
                for k in rng.choice(keys, 8, replace=False):
                    d = np.load(f"{RD}/data_gate_synth/{k}.npz")
                    T_ = len(d["state"])
                    t = int(rng.integers(int(lo * T_), max(int(lo * T_) + 1, int(hi * T_))))
                    frames.append({"observation/image": r224(d["image"][t]),
                                   "observation/wrist_image": r224(d["wrist"][t]),
                                   "observation/state": d["state"][t].astype(np.float32),
                                   "prompt": meta[k]["lang"]})
                votes = {"cruise": 0, "decel": 0, "stop": 0, "neither": 0}
                for _ in range(16):
                    for c in joint_head.head_c(policy, frames):
                        votes[assign(c, cents[side])] += 1
                tot = sum(votes.values())
                print(f"   {side:5s} {seg:6s} frames -> " +
                      " ".join(f"{k}:{v / tot:.2f}" for k, v in votes.items()))

    # ---- links 2+3: closed-loop draws + execution (CPU) ----
    clog = np.load(f"/home/ubuntu/ctxrun/clog_{a.arm}.npy")
    print(f"\n== links 2+3: closed-loop ({a.arm}, {len(clog)} replans; rows attributed by side)")
    for side in a.sides.split(","):
        own = cents[side]
        rows = []
        for r in clog:
            d_goal = np.linalg.norm(r[:3] - GOAL)
            other = "right" if side == "left" else "left"
            if other in cents:
                import numpy as _np
                # attribute row to this side by y-sign region (left route +y, right -y)
                if side == "left" and r[1] < -0.15:
                    continue
                if side == "right" and r[1] > 0.15:
                    continue
            rows.append((d_goal, r[3:]))
        near = [(d, c) for d, c in rows if d < 1.0]
        if not near:
            print(f"   {side}: no goal-region replans attributed")
            continue
        votes = {"cruise": 0, "decel": 0, "stop": 0, "neither": 0}
        for _, c in near:
            votes[assign(c, own)] += 1
        tot = sum(votes.values())
        print(f"   {side:5s} goal-region draws (d<1.0m, n={tot}): " +
              " ".join(f"{k}:{v / tot:.2f}" for k, v in votes.items()))
    print("\n(link 3 executed-settle check: run tail_execution_check in this file next — needs "
          "per-rollout alignment of clog rows to trajectories, only valid for sequential-eval logs)")
    print("ATTRIB_DONE")


if __name__ == "__main__":
    main()
