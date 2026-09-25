"""Command carry (2026-09-24): how much of a commanded coarse motion does the flow actually deliver?

For many frames of sim-rendered (data_gate_synth3) and real (data_gate_real) demonstrations, write a set of
commands into the source (the agent's own move primitives via AgentMove, the demo's own command, and the
head's), denoise once at a given sigma, and compare the delivered chunk with the command:
  carry  = <realized net displacement, commanded net displacement> / |commanded|^2   (whole 50-step chunk)
  ortho  = |realized - carry * commanded| / |commanded|                              (sideways error)
  c_rel  = |U^T a_hat - c| / |c|                                                     (command-space error)
  yaw_carry for turn commands, same form on net yaw.

  SNMVP_HEAD=1 ... python carry_probe.py --ckpt <ck> --out carry.npz [--frames 60] [--sigma 0]
"""
import argparse, json, os, sys
import numpy as np
RD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, RD)
import joint_head
from sigma_phase_probe import gmm_params
from agent_prompt import AgentMove, H, AD

MOVES = {"forward 1 m": {"forward": 1.0}, "forward 2 m": {"forward": 2.0}, "left 1 m": {"left": 1.0},
         "right 1 m": {"left": -1.0}, "back 0.5 m": {"forward": -0.5}, "up 0.5 m": {"up": 0.5},
         "turn left 30 + forward 1 m": {"forward": 1.0, "yaw_deg": 30.0}, "turn right 45": {"yaw_deg": -45.0}}
PROMPT = {"left": "go through the gate on the left and hover over the stuffed animal",
          "right": "go through the gate on the right and hover over the stuffed animal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--norm", default=os.path.expanduser("~/hf_bundle/gate-drone-pi0/assets/gate_nav"))
    ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    joint_head.enable_head(a.pin_u)
    from PIL import Image
    import gate_ctx_common as gc
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as _nz
    import openpi.training.config as C
    cfg = C.get_config("pi0_gate")
    policy = PC.create_trained_policy(cfg, a.ckpt, norm_stats=gc.pad_norm_stats(_nz.load(a.norm), cfg.model.action_dim))
    U = np.load(a.pin_u).astype(np.float32)
    NS = json.load(open(os.path.join(a.norm, "norm_stats.json")))["norm_stats"]["actions"]
    amean, astd = np.asarray(NS["mean"], np.float32)[:7], np.asarray(NS["std"], np.float32)[:7]
    mover = AgentMove(amean, astd, U)
    r224 = lambda im: np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC), np.uint8)
    rng = np.random.default_rng(a.seed)

    def norm_chunk(act7):
        ch = np.zeros((H, AD), np.float32); m = min(H, len(act7)); ch[:m, :7] = (act7[:m] - amean) / (astd + 1e-6); return ch

    def phys(ch):   # normalized (H, AD) -> physical per-step [dx, dy, dz, dyaw]
        return ch[:, :4] * (astd[:4] + 1e-6) + amean[:4]

    def gen(obs, c):
        g = rng.standard_normal(H * AD).astype(np.float32)
        noise = (g - (g @ U) @ U.T + c @ U.T).reshape(H, AD).astype(np.float32)
        return np.asarray(policy.infer(obs, noise=noise, snmvp_sigma=a.sigma)["actions"], np.float32)[:H, :7]

    frames = []
    for dom, ddir, eps in (("sim", "data_gate_synth3", range(100, 200)), ("real", "data_gate_real", range(0, 100))):
        pool = []
        for e in eps:
            p = f"{RD}/{ddir}/ep_{e:04d}.npz"
            if not os.path.exists(p): continue
            n = len(np.load(p, allow_pickle=True)["state"])
            for t in range(5, n - H, 15): pool.append((e, t))
        pick = rng.choice(len(pool), size=min(a.frames, len(pool)), replace=False)
        frames += [(dom, ddir) + pool[i] for i in sorted(pick)]
    print(f"{len(frames)} frames", flush=True)

    rows = []
    cache = {}
    for k, (dom, ddir, e, t) in enumerate(frames):
        if (ddir, e) not in cache: cache = {(ddir, e): np.load(f"{RD}/{ddir}/ep_{e:04d}.npz", allow_pickle=True)}
        d = cache[(ddir, e)]; st = d["state"].astype(np.float32); ac = d["action"].astype(np.float32)
        side = "left" if np.mean(st[:, 1]) > -0.2 else "right"
        obs = {"observation/image": r224(d["image"][t]), "observation/wrist_image": r224(d["wrist"][t]),
               "observation/state": st[t], "prompt": PROMPT[side]}
        w, mu, _ = gmm_params(policy, [obs]); c_head = mu[0, int(w[0].argmax())].astype(np.float32)
        cmds = {"demo's own command": norm_chunk(ac[t:t + H]), "head's own command": None}
        for name, mv in MOVES.items():
            cmds[name] = mover.chunk(dict(mv), st[t, :4])[0]
        for name, ch in cmds.items():
            c = c_head if ch is None else ch.reshape(-1) @ U
            want = phys((U @ c).reshape(H, AD))          # the command's own coarse track, decoded
            act = gen(obs, c); got_n = norm_chunk(act)
            got = phys(got_n)
            W, G = want[:, :3].sum(0), got[:, :3].sum(0)
            wy, gy = want[:, 3].sum(), got[:, 3].sum()
            nw = float(W @ W)
            rows.append({"dom": dom, "e": int(e), "t": int(t), "cmd": name,
                         "carry": float(G @ W / nw) if nw > 1e-4 else np.nan,
                         "ortho": float(np.linalg.norm(G - (G @ W / nw) * W) / np.sqrt(nw)) if nw > 1e-4 else np.nan,
                         "cmd_m": float(np.sqrt(nw)), "got_m": float(np.linalg.norm(G)),
                         "yaw_carry": float(gy / wy) if abs(wy) > 0.05 else np.nan,
                         "c_rel": float(np.linalg.norm(got_n.reshape(-1) @ U - c) / (np.linalg.norm(c) + 1e-6))})
        if k % 10 == 0: print(f"frame {k}/{len(frames)} {dom} ep{e} t{t}", flush=True)
    json.dump(rows, open(a.out, "w"))
    print("saved", a.out, len(rows), "rows")
    import collections
    for dom in ("sim", "real"):
        print(f"\n== {dom}, sigma {a.sigma}: median over frames (p25-p75)")
        by = collections.defaultdict(list)
        for r in rows:
            if r["dom"] == dom: by[r["cmd"]].append(r)
        for name, rs in by.items():
            f = lambda key: np.array([r[key] for r in rs if np.isfinite(r[key])])
            cr, orr, yc, cm = f("carry"), f("ortho"), f("yaw_carry"), f("c_rel")
            q = lambda x: f"{np.median(x):.2f} ({np.percentile(x,25):.2f}-{np.percentile(x,75):.2f})" if len(x) else "  -"
            print(f"  {name:28s} carry {q(cr):20s} sideways {q(orr):20s} yaw carry {q(yc):20s} |dc|/|c| {q(cm)}")


if __name__ == "__main__":
    main()
