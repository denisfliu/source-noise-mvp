"""Train-once + tiny-per-embodiment-adapter, tested faithfully with an
EXECUTION LOOP (the drone-navigation architecture Denis wants).

The earlier gate battery scored the policy's OUTPUT directly (no dynamics), which
is unfaithful: it hid the drone's real difficulty (inertia between command and
outcome). Here we close that loop:

    policy -> command chunk -> body.realize() [dynamics] -> realized path -> score

Split (the fix for the earlier wrong-split failure): FREEZE the shared task
INTENT (trajectory that achieves the goal, trained once on non-drone bodies),
and make the tiny per-embodiment adapter the DYNAMICS/inverse-model, composed as
a RESIDUAL so zero drone data = command-the-intent and minimal data = learn-the
body's pre-compensation:

    a_cmd = a_shared(obs)  +  g_body(obs)        # g_body: small, few drone demos

Expert commands (BC targets) are produced per body by ITERATIVE LEARNING CONTROL
(ILC): find the command whose realized path matches the ideal gate-passing path.
An arm needs almost none; the inertia drone (point_drag) needs a lot — that gap
is exactly what the adapter must learn from minimal data.

Headline metric: realized gate-passage success vs number of drone demos, for
  zero-shot (frozen intent, 0 drone demos) / intent+residual-adapter / scratch.
Your architecture wins if the adapter reaches high success at n far below scratch.
No OAT, no VLA^2 -- the simplest version of "train one thing, tiny drone adapter".
"""

import json
import os
import sys
import time

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
import gate_dataset as gd          # noqa: E402
import embodiments as emb          # noqa: E402
import dataset as tfd              # noqa: E402

H = gd.H
ACT_SCALE = gd.ACT_SCALE
OBS = gd.OBS_DIM
HID = 128
SET_A = ["arm2", "arm3", "arm4"]
NG = [1, 2]
BODIES_B = ["point_drag", "point"]
SEEDS = [0, 1, 2]
NS = [1, 3, 10, 30]
ITERS = 6000
N_EVAL = 100
OUT = os.path.join(HERE, "results", "gate_adapter")


# ------------------- inverse dynamics (ILC) expert commands ------------------

def ilc_command(body, ideal_pos, iters=30, lr=1.0):
    """Command (world positions) whose realized path matches ideal_pos, via
    iterative learning control: cmd <- cmd + (ideal - realize(cmd)). Converges
    for the toy's mild dynamics; where a body physically can't track (arm past
    reach), it settles at the closest feasible command (its ceiling)."""
    cmd = np.asarray(ideal_pos, dtype=float).copy()
    for _ in range(iters):
        achieved = body.realize(cmd)
        cmd = cmd + lr * (ideal_pos - achieved)
    return cmd


def expert_chunk(body, scene, rng):
    """Canonical expert command chunk (H,2) for `body` on `scene`."""
    P_ideal = gd.planned_positions(scene, rng)                 # world, gate-passing
    cmd_world = ilc_command(body, P_ideal)
    world_chunk = np.diff(cmd_world, axis=0) * ACT_SCALE       # (H,2)
    return tfd.to_canonical(world_chunk[None], np.array([scene["angle"]]))[0]


def make_expert_data(bodies_subset, n_scenes, rng, n_gates, per_scene=1):
    """-> obs (M,OBS), angles (M,), and per-body canonical expert chunks (M,H,2)
    (averaged over per_scene draws to denoise style)."""
    scenes, obs, ang = [], [], []
    experts = {b: [] for b in bodies_subset}
    for _ in range(n_scenes):
        sc = gd.make_scene(rng, n_gates)
        scenes.append(sc); obs.append(gd.scene_obs(sc)); ang.append(sc["angle"])
        for b, body in bodies_subset.items():
            experts[b].append(np.mean([expert_chunk(body, sc, rng)
                                       for _ in range(per_scene)], axis=0))
    return (scenes, np.array(obs), np.array(ang),
            {b: np.array(v) for b, v in experts.items()})


# ------------------------------ policy (BC) ----------------------------------

def mlp_init(dims, rng):
    return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
            for a, b in zip(dims[:-1], dims[1:])]


def mlp(params, x):
    h = x
    for w, b in params[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = params[-1]
    return h @ w + b


def train_bc(obs, target_chunks, seed, iters, base=None):
    """Regress obs -> canonical command chunk (flattened). If `base` (M,H*2) is
    given, learn only the RESIDUAL on top of it (the adapter)."""
    rng = np.random.default_rng(seed)
    M = obs.shape[0]
    y = target_chunks.reshape(M, -1)
    hid = 64 if base is not None else HID          # adapter is small
    params = mlp_init([OBS, hid, hid, H * 2], rng)
    b0 = np.zeros((M, H * 2)) if base is None else base

    def loss(p, it):
        r = np.random.default_rng(it)
        idx = r.integers(0, M, size=min(128, M))
        pred = mlp(p, obs[idx]) + b0[idx]
        return anp.mean((pred - y[idx]) ** 2)

    return adam(grad(loss), params, num_iters=iters, step_size=2e-3)


# --------------------------- execution-loop eval -----------------------------

def eval_realized(cmd_chunks, scenes, angles, body):
    """cmd_chunks (M,H,2) canonical -> rotate to world -> body.realize ->
    realized path -> score gate passage on the REALIZED trajectory."""
    world = tfd.to_canonical(cmd_chunks, -np.asarray(angles))          # canon->world
    succ = []
    for i, sc in enumerate(scenes):
        pos = np.concatenate([[[0.0, 0.0]], np.cumsum(world[i] / ACT_SCALE, axis=0)])
        achieved = body.realize(pos)
        realized_chunk = np.diff(achieved, axis=0) * ACT_SCALE
        succ.append(gd.success(sc, realized_chunk))
    return float(np.mean(succ))


def predict(params, obs, base=None):
    out = np.asarray(mlp(params, obs))
    if base is not None:
        out = out + base
    return out.reshape(-1, H, 2)


# ------------------------------- battery -------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "rows.jsonl")
    open(rows_path, "w").close()
    t0 = time.time()
    bodies = emb.make_bodies()

    rows = []
    for ng in NG:
        # shared intent: BC on set-A arms' expert commands (train once, frozen)
        setA = {b: bodies[b] for b in SET_A}
        _, A_obs, A_ang, A_exp = make_expert_data(setA, 200, np.random.default_rng(7), ng)
        A_obs_all = np.concatenate([A_obs] * len(SET_A))
        A_tgt_all = np.concatenate([A_exp[b] for b in SET_A])
        shared = train_bc(A_obs_all, A_tgt_all, seed=0, iters=8000)
        print(f"[{time.time()-t0:.0f}s] n_gates={ng}: shared intent trained on "
              f"{A_tgt_all.shape[0]} arm expert demos", flush=True)

        he_s, he_o, he_a, _ = make_expert_data({}, N_EVAL, np.random.default_rng(7777), ng)
        # zero-shot ceiling: does commanding the shared intent pass, per body?
        base_he = np.asarray(mlp(shared, he_o))
        shared_chunks = base_he.reshape(-1, H, 2)

        for B in BODIES_B:
            bodyB = bodies[B]
            zshot = eval_realized(shared_chunks, he_s, he_a, bodyB)
            for s in SEEDS:
                for n in NS:
                    _, ad_o, ad_a, ad_exp = make_expert_data(
                        {B: bodyB}, n, np.random.default_rng(1234 + s), ng)
                    base_ad = np.asarray(mlp(shared, ad_o))          # frozen intent on drone obs
                    # adapter: residual on the frozen shared intent
                    adapter = train_bc(ad_o, ad_exp[B], seed=s, iters=ITERS, base=base_ad)
                    # scratch: full policy from n drone demos
                    scratch = train_bc(ad_o, ad_exp[B], seed=s, iters=ITERS, base=None)

                    a_chunks = predict(adapter, he_o, base=base_he)
                    s_chunks = predict(scratch, he_o, base=None)
                    row = {"n_gates": ng, "B": B, "seed": s, "n": n,
                           "zeroshot": round(zshot, 4),
                           "adapter": round(eval_realized(a_chunks, he_s, he_a, bodyB), 4),
                           "scratch": round(eval_realized(s_chunks, he_s, he_a, bodyB), 4)}
                    rows.append(row)
                    with open(rows_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[{time.time()-t0:.0f}s] ng{ng} {B} s{s} n{n}: "
                          f"zeroshot={row['zeroshot']} adapter={row['adapter']} "
                          f"scratch={row['scratch']}", flush=True)

    verdict = summarize(rows)
    json.dump({"rows": rows, "verdict": verdict},
              open(os.path.join(OUT, "battery.json"), "w"), indent=2)
    print("VERDICT:", json.dumps(verdict, indent=2))
    print(f"GATE_ADAPTER_DONE=ok in {time.time()-t0:.0f}s")


def summarize(rows):
    out = {}
    for ng in NG:
        for B in BODIES_B:
            zs = [x["zeroshot"] for x in rows if x["n_gates"] == ng and x["B"] == B]
            d = {"zeroshot": round(float(np.mean(zs)), 3)} if zs else {}
            for n in NS:
                r = [x for x in rows if x["n_gates"] == ng and x["B"] == B and x["n"] == n]
                if r:
                    d[f"adapter_n{n}"] = round(float(np.mean([x["adapter"] for x in r])), 3)
                    d[f"scratch_n{n}"] = round(float(np.mean([x["scratch"] for x in r])), 3)
            out[f"ng{ng}_{B}"] = d
    out["_reading"] = ("adapter = frozen shared intent + tiny residual on n drone "
                       "demos; scratch = full policy on n drone demos; zeroshot = "
                       "shared intent, 0 drone demos. Architecture wins if adapter "
                       "reaches high realized gate-passage at n far below scratch.")
    return out


if __name__ == "__main__":
    main()
