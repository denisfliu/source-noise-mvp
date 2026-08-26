"""Does the grounded prior see a different feature at train time than at serve time?

The prior's language-embedding input is far outside its training distribution from the FIRST
inference of an episode, with the drone at the origin where every demo also starts, so the skew is
not novel viewpoints. The prompt is identical, so the remaining input is the image — and the cache
was built from the STORED demo frames while the server embeds LIVE renders of the same scene.

For a sample of demo timesteps this computes, at the very same state:
  e64_stored  from the frame the cache used            (training convention)
  e64_live    from a live render through the serving path
and reports how far each sits from the cached training distribution, plus what each implies for the
commanded chunk displacement in metres against the demo's own command.

Two stages, because the renderer (gsplat, /tmp/tv) and the policy (openpi venv) live in different
environments:
  /tmp/tv/bin/python feature_skew_probe.py --render --n 16 --out live.npz
  openpi/.venv/bin/python feature_skew_probe.py --frames live.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
H = 50
TASK_EPS = {"left": range(100, 150), "right": range(150, 200)}
PROMPT = {"left": "go through the gate on the left and hover over the stuffed animal",
          "right": "go through the gate on the right and hover over the stuffed animal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="frames per task")
    ap.add_argument("--prior", default=f"{RD}/langprior_zeropad.pt")
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_gate_rrr_k5.npy")
    ap.add_argument("--norm", default="/home/ubuntu/hf_bundle/gate-drone-pi0/assets/gate_nav")
    ap.add_argument("--ckpt", default="/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_pin_zeropad/4999")
    ap.add_argument("--render", action="store_true", help="stage 1: dump live renders (gsplat env)")
    ap.add_argument("--out", default="/home/ubuntu/ctxrun/skew_live.npz")
    ap.add_argument("--frames", default="/home/ubuntu/ctxrun/skew_live.npz")
    a = ap.parse_args()

    if a.render:                                  # stage 1: renderer env only
        import render_skew_probe as RS
        rng0 = np.random.default_rng(0)
        keys, imgs = [], []
        for task, eps in TASK_EPS.items():
            for e in rng0.choice(list(eps), a.n, replace=True):
                dat = np.load(f"{RD}/data_gate_synth/ep_{int(e):04d}.npz", allow_pickle=True)
                st = dat["state"].astype(np.float32)
                t = int(rng0.integers(0, min(200, len(st) - 6)))
                imgs.append(RS.to(RS.to(RS.rend(st[t, :3], -float(st[t, 3])), 256, Image.BILINEAR),
                                  224, Image.BICUBIC))
                keys.append((task, int(e), t))
        np.savez_compressed(a.out, img=np.stack(imgs),
                            task=np.array([k[0] for k in keys]),
                            ep=np.array([k[1] for k in keys]), t=np.array([k[2] for k in keys]))
        print(f"rendered {len(imgs)} live frames -> {a.out}")
        return

    import openpi.training.config as C
    import openpi.policies.policy_config as PC
    import gate_ctx_common as gc
    from openpi.shared.normalize import NormStats, load as load_ns
    import openpi.transforms as T

    cfg = C.get_config("pi0_gate")
    AD = cfg.model.action_dim
    policy = PC.create_trained_policy(cfg, a.ckpt, default_prompt="")
    U = np.load(a.pin_u).astype(np.float32)
    K = U.shape[1]
    d = torch.load(a.prior, map_location="cpu", weights_only=False)
    net = torch.nn.Sequential(torch.nn.Linear(d["in_dim"], 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, K))
    net.load_state_dict(d["state_dict"])
    net.eval()

    def pads(ns, dim):
        o = {}
        for k, s in ns.items():
            n = len(s.mean)
            if n >= dim:
                o[k] = s
                continue
            p = dim - n
            ext = lambda x, f: None if x is None else np.concatenate(
                [np.asarray(x, np.float32), np.full(p, f, np.float32)])
            o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
        return o

    PS = pads(load_ns(a.norm), AD)
    nrm = T.Normalize(PS, use_quantiles=False)
    amean = np.asarray(PS["actions"].mean, np.float32)
    astd = np.asarray(PS["actions"].std, np.float32)

    z = np.load(f"{RD}/langprior_feats.npz")
    E64_cache = (z["E"] - d["Em"]) @ d["P"]
    ep_cache = z["ep"]

    def metres(c):
        ch = (U @ np.atleast_2d(c).T).T.reshape(-1, H, AD)
        return (ch * astd[None, None, :] + amean[None, None, :])[:, :, :3].sum(1)

    r224 = lambda img: np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC), np.uint8)
    Z = np.load(a.frames)
    LIVE, LTASK, LEP, LT = Z["img"], Z["task"], Z["ep"], Z["t"]
    for task, eps in TASK_EPS.items():
        m = np.isin(ep_cache, list(eps))
        mu, sd = E64_cache[m].mean(0), E64_cache[m].std(0) + 1e-9
        sel = np.where(LTASK == task)[0]
        rows = []
        for row in sel:
            e, t = int(LEP[row]), int(LT[row])
            f = f"{RD}/data_gate_synth/ep_{e:04d}.npz"
            dat = np.load(f, allow_pickle=True)
            st, ac = dat["state"].astype(np.float32), dat["action"].astype(np.float32)
            if t >= len(st) - 5:
                continue
            obs_stored = {"observation/image": r224(dat["image"][t]),
                          "observation/wrist_image": r224(dat["wrist"][t]),
                          "observation/state": st[t], "prompt": PROMPT[task]}
            live_f = LIVE[row]
            obs_live = dict(obs_stored, **{"observation/image": live_f,
                                           "observation/wrist_image": live_f})
            es, el = gc.lang_pool(policy, [obs_stored])[0], gc.lang_pool(policy, [obs_live])[0]
            ms = np.asarray(policy._input_transform(dict(obs_stored))["state"]).reshape(-1)
            out = []
            for ee in (es, el):
                e64 = (ee - d["Em"]) @ d["P"]
                x = np.concatenate([ms, e64]).astype(np.float32)
                with torch.no_grad():
                    out.append((e64, net(torch.tensor(((x - d["mu"]) / d["sd"])[None]))[0].numpy()))
            ch = np.zeros((H, AD), np.float32)
            k = min(H, len(ac) - t)
            ch[:k, :7] = ac[t:t + k]
            ctrue = (nrm({"actions": ch})["actions"].reshape(-1)) @ U
            rows.append((out[0][0], out[1][0], out[0][1], out[1][1], ctrue))
        Es = np.array([r[0] for r in rows]); El = np.array([r[1] for r in rows])
        Cs = np.array([r[2] for r in rows]); Cl = np.array([r[3] for r in rows])
        Ct = np.array([r[4] for r in rows])
        zs, zl = np.abs((Es - mu) / sd), np.abs((El - mu) / sd)
        ds, dl, dt = metres(Cs), metres(Cl), metres(Ct)
        print(f"\n=== {task}: {len(rows)} demo timesteps, identical state and prompt, image path differs")
        print(f"  feature vs training distribution   stored: mean|z| {zs.mean():5.2f}  "
              f"dims|z|>3 {100 * (zs > 3).mean():5.1f}%      live: mean|z| {zl.mean():5.2f}  "
              f"dims|z|>3 {100 * (zl > 3).mean():5.1f}%")
        print(f"  command error vs the demo's own    stored: {np.linalg.norm(ds - dt, axis=1).mean():.3f} m"
              f"      live: {np.linalg.norm(dl - dt, axis=1).mean():.3f} m")
        print(f"  stored vs live command differ by   {np.linalg.norm(ds - dl, axis=1).mean():.3f} m "
              f"(max {np.linalg.norm(ds - dl, axis=1).max():.3f})")


if __name__ == "__main__":
    main()
