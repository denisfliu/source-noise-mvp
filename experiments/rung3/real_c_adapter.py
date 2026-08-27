"""Learned real-sim reconciliation in COMMAND SPACE (2026-08-27, Denis: 'rotating fixes
trajectories -> there is probably a learned reconciliation; find a general way').

The rotation verb is ~a linear operator in c-space, so the general first rung is an AFFINE
adapter c' = A c_pred + b (K^2+K params) fit on real episodes' own oracle targets.

--stage collect : sweep real episodes, save (c_pred, c_oracle, side, frac, e, t) per anchor.
--stage fit     : train/held-out split by EPISODE; least squares (ridge); report held-out
                  c-error raw vs adapted, pooled + per-side + rotation-structure readout.
--stage geneval : generation ladder on HELD-OUT anchors, all sigma=0:
                  head-c | adapted-c | oracle-c -> heading error vs real continuation.
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
TRAIN_EPS = 80   # episodes 0..79 train, 80..99 held out


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


def load_norm():
    NS = json.load(open(os.path.expanduser(
        "~/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json")))["norm_stats"]["actions"]
    return np.asarray(NS["mean"], np.float32), np.asarray(NS["std"], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect", "fit", "geneval"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--rows", default="/home/dfliu/ctxrun/cadapt_rows.npz")
    ap.add_argument("--adapter", default="/home/dfliu/ctxrun/cadapt_affine.npz")
    ap.add_argument("--stride", type=int, default=25)
    a = ap.parse_args()
    U = np.load(a.pin_u).astype(np.float32)
    K = U.shape[1]
    amean, astd = load_norm()

    def chunk_c(ac, t):
        ch = np.zeros((H, AD), np.float32)
        m = min(H, len(ac) - t)
        ch[:m, :7] = (ac[t:t + m] - amean) / (astd + 1e-6)
        return ch.reshape(-1) @ U

    if a.stage == "collect":
        import joint_head
        joint_head.enable_head(a.pin_u)
        from sigma_phase_probe import gmm_params
        from PIL import Image
        import gate_ctx_common as gc
        import openpi.policies.policy_config as PC
        import openpi.shared.normalize as _nz
        import openpi.training.config as C
        cfg = C.get_config("pi0_gate")
        policy = PC.create_trained_policy(cfg, a.ckpt,
                                          norm_stats=gc.pad_norm_stats(_nz.load(a.norm),
                                                                       cfg.model.action_dim))
        r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
        rows = []
        for e in range(100):
            d = np.load(f"{RD}/data_gate_real/ep_{e:04d}.npz", allow_pickle=True)
            st, ac = d["state"].astype(np.float32), d["action"].astype(np.float32)
            side = classify(st)
            if side is None:
                continue
            obs_list, metas = [], []
            for t in range(10, len(st) - 30, a.stride):
                obs_list.append({"observation/image": r224(d["image"][t]),
                                 "observation/wrist_image": r224(d["wrist"][t]),
                                 "observation/state": st[t], "prompt": PROMPTS[side]})
                metas.append(t)
            for i in range(0, len(obs_list), 8):
                batch = obs_list[i:i + 8]
                w, mu, _ = gmm_params(policy, batch)
                for j, ob in enumerate(batch):
                    t = metas[i + j]
                    rows.append(dict(e=e, t=t, frac=t / len(st),
                                     side=0 if side == "left" else 1,
                                     c_pred=mu[j, int(w[j].argmax())],
                                     c_orc=chunk_c(ac, t)))
            print(f"ep{e:03d} [{side}] {len(metas)} anchors", flush=True)
        np.savez(a.rows,
                 e=np.array([r["e"] for r in rows]), t=np.array([r["t"] for r in rows]),
                 frac=np.array([r["frac"] for r in rows]),
                 side=np.array([r["side"] for r in rows]),
                 c_pred=np.stack([r["c_pred"] for r in rows]),
                 c_orc=np.stack([r["c_orc"] for r in rows]))
        print(f"saved {len(rows)} rows -> {a.rows}")
        return

    Z = np.load(a.rows)
    cstd = float(np.linalg.norm(np.std(Z["c_orc"], axis=0)))
    tr = Z["e"] < TRAIN_EPS
    te = ~tr

    if a.stage == "fit":
        X, Y = Z["c_pred"][tr], Z["c_orc"][tr]
        Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
        lam = 1.0
        W = np.linalg.solve(Xa.T @ Xa + lam * np.eye(K + 1), Xa.T @ Y)   # (K+1, K)
        A, b = W[:K].T, W[K]
        def err(X_, Y_, adapt):
            P = X_ @ A.T + b if adapt else X_
            return np.linalg.norm(P - Y_, axis=1) / cstd
        print(f"rows: train {tr.sum()} held-out {te.sum()}  cstd={cstd:.1f}")
        for name, m in [("train", tr), ("HELD-OUT", te)]:
            e0 = err(Z["c_pred"][m], Z["c_orc"][m], False)
            e1 = err(Z["c_pred"][m], Z["c_orc"][m], True)
            print(f"  {name:9s}: raw c-err median {np.median(e0):.3f} -> adapted {np.median(e1):.3f}")
        for s, nm in [(0, "left"), (1, "right")]:
            m = te & (Z["side"] == s)
            e0 = err(Z["c_pred"][m], Z["c_orc"][m], False)
            e1 = err(Z["c_pred"][m], Z["c_orc"][m], True)
            print(f"  held-out {nm:5s}: {np.median(e0):.3f} -> {np.median(e1):.3f} (n={m.sum()})")
        # structure readout: how far is A from identity / how rotation-like
        sv = np.linalg.svd(A, compute_uv=False)
        print(f"  A: ||A-I||_F={np.linalg.norm(A - np.eye(K)):.2f}  singvals [{sv.min():.2f},{sv.max():.2f}]  ||b||/cstd={np.linalg.norm(b)/cstd:.3f}")
        np.savez(a.adapter, A=A, b=b)
        print(f"saved adapter -> {a.adapter}")
        return

    # geneval
    import joint_head
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import gate_ctx_common as gc
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    cfg = C.get_config("pi0_gate")
    policy = PC.create_trained_policy(cfg, a.ckpt,
                                      norm_stats=gc.pad_norm_stats(_nz.load(a.norm),
                                                                   cfg.model.action_dim))
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    AB = np.load(a.adapter)
    A, b = AB["A"].astype(np.float32), AB["b"].astype(np.float32)
    rng = np.random.default_rng(0)

    def gen(obs, c):
        g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(H, AD).astype(np.float32)
        return np.asarray(policy.infer(obs, noise=noise, snmvp_sigma=0.0)["actions"], np.float32)[:H]

    idx = np.where(te)[0]
    rng.shuffle(idx)
    idx = idx[:60]
    errs = {"head": [], "adapted": [], "oracle": []}
    sides = []
    cache = {}
    for i in idx:
        e, t = int(Z["e"][i]), int(Z["t"][i])
        if e not in cache:
            cache = {e: np.load(f"{RD}/data_gate_real/ep_{e:04d}.npz", allow_pickle=True)}
        d = cache[e]
        st = d["state"].astype(np.float32)
        side = "left" if Z["side"][i] == 0 else "right"
        sides.append(side)
        obs = {"observation/image": r224(d["image"][t]),
               "observation/wrist_image": r224(d["wrist"][t]),
               "observation/state": st[t], "prompt": PROMPTS[side]}
        m = min(25, len(st) - 1 - t)
        hr = np.arctan2(*(st[t + m, :2] - st[t, :2])[::-1])
        for tag, c in [("head", Z["c_pred"][i]),
                       ("adapted", A @ Z["c_pred"][i] + b),
                       ("oracle", Z["c_orc"][i])]:
            acts = gen(obs, c.astype(np.float32))
            h = np.arctan2(*np.sum(acts[:m, :2], axis=0)[::-1])
            errs[tag].append(abs(np.degrees(np.angle(np.exp(1j * (h - hr))))))
    print(f"GENERATION LADDER on {len(idx)} held-out anchors (all sigma=0), |heading err| deg:")
    for tag in ("head", "adapted", "oracle"):
        v = np.array(errs[tag])
        print(f"  {tag:8s}: median {np.median(v):5.1f}  p75 {np.percentile(v, 75):5.1f}")
        for s in ("left", "right"):
            vv = v[[i for i, sd in enumerate(sides) if sd == s]]
            if len(vv):
                print(f"      {s:5s}: median {np.median(vv):5.1f} (n={len(vv)})")


if __name__ == "__main__":
    main()
