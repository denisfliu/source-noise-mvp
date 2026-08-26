"""Distill the (now 4/5-successful) compound waypoint oracle into a LEARNED prior.

The oracle is closed-form in position, so label the workspace densely: for the CFL
task, c* = oracle chunk toward {east-clearance waypoint | gate-2 carry-through |
goal} (exact logic mirrored from serve_gate_pin_oracle). The other three tasks keep
their demo-derived labels (same rows the no-clock prior trains on), so the result
is a drop-in 4-task prior in the standard schema — serveable by
serve_gate_pin_prog4 / serve_gate_pin_splice with zero server changes.

If this prior reproduces the oracle's compound completions closed-loop, the route
knowledge is LEARNABLE from (state, task) — the remaining step to a grounded
command source is swapping the input for VLM features, not inventing new machinery.
CPU-only (JAX_PLATFORMS=cpu).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc

RD = gc.RD
OUT = os.path.join(RD, "oracle_distill_prior.pt")
U = np.load(os.path.join(RD, "pin_U_gate_rrr_k5.npy"))
ns, amean, astd = gc.load_norm()
H = gc.H
TASKS4 = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]
GATE2_CENTER = np.array([2.756, -0.3275, 1.0])
GATE2_PLANE_Y = -0.3275
GOAL = np.array([1.525, -0.615, 1.0])

torch.manual_seed(0)
rng = np.random.default_rng(0)
policy = gc.make_policy()
_D = np.zeros((224, 224, 3), np.uint8)


def mstate(raw, lang):
    return np.asarray(policy._input_transform(
        {"observation/image": _D, "observation/wrist_image": _D,
         "observation/state": raw, "prompt": lang})["state"]).reshape(-1)


def oracle_c(pos):
    if pos[1] < GATE2_PLANE_Y - 0.20:
        target = GOAL
    elif pos[0] < 1.9:
        target = np.array([2.05, 0.85, 1.15])
    else:
        target = GATE2_CENTER + np.array([0.0, -0.40, 0.0])
    delta = target - pos
    dist = np.linalg.norm(delta)
    net = delta / (dist + 1e-9) * min(dist, 1.0)
    chunk = np.zeros((H, 7), np.float32)
    chunk[:, :3] = net / H
    return (gc.segY(chunk, amean, astd) @ U).astype(np.float32)


X, Y = [], []
# oracle rows: dense position sweep over the compound workspace, yaw-marginalized.
# v2: the endgame and the phase-boundary band get dedicated oversampling — the v1
# distillate reproduced the route (gates 2/2 in 4/5) but parked ~0.3 m outside the
# goal box: near the goal the oracle command magnitude -> 0 and across the
# through-latch boundary the target jumps, so uniform sampling under-trains exactly
# where precision is needed (2026-08-05 closed-loop).
N_U, N_G, N_B = 6000, 3000, 2000
pos_u = np.stack([rng.uniform(-0.5, 3.6, N_U), rng.uniform(-2.0, 1.5, N_U),
                  rng.uniform(0.4, 2.0, N_U)], 1)
pos_g = GOAL + rng.normal(0, [0.45, 0.45, 0.3], (N_G, 3))          # endgame ball
pos_b = np.stack([rng.uniform(1.8, 3.4, N_B),                       # through-boundary band
                  GATE2_PLANE_Y - 0.20 + rng.normal(0, 0.35, N_B),
                  rng.uniform(0.5, 1.6, N_B)], 1)
pos = np.concatenate([pos_u, pos_g, pos_b], 0)
N_OR = len(pos)
yaw = rng.uniform(-1.2, 1.2, N_OR)
oh_cfl = np.zeros(4, np.float32); oh_cfl[0] = 1.0
for i in range(N_OR):
    raw = np.array([*pos[i], yaw[i], 0, 0, 0], np.float32)
    X.append(np.concatenate([mstate(raw, gc.PROMPT_CFL), oh_cfl]).astype(np.float32))
    Y.append(oracle_c(pos[i]))
print("oracle rows:", N_OR, flush=True)

# demo rows for the other three tasks (authoritative labels, frozen split not needed:
# this prior's CFL behavior is the object under test, the rest is passthrough)
eps = gc.load_eps(with_images=False)
for e in eps:
    if e["lang"] == gc.PROMPT_CFL or e["lang"] not in TASKS4:
        continue
    oh = np.zeros(4, np.float32); oh[TASKS4.index(e["lang"])] = 1.0
    st = e["state"].astype(np.float32); ac = e["action"].astype(np.float32)
    for t in range(0, len(st) - H, 8):
        X.append(np.concatenate([mstate(st[t], e["lang"]), oh]).astype(np.float32))
        Y.append((gc.segY(ac[t:], amean, astd) @ U).astype(np.float32))
X = np.stack(X); Y = np.stack(Y)
print("total rows:", len(X), flush=True)

mu, sd = X.mean(0), X.std(0) + 1e-6
Xn = torch.tensor((X - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Y)
net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256),
                    nn.SiLU(), nn.Linear(256, 5))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
idx = np.arange(len(X))
for ep in range(90):
    rng.shuffle(idx)
    tot = 0.0
    for i in range(0, len(idx), 1024):
        b = idx[i:i + 1024]; opt.zero_grad()
        loss = ((net(Xn[b]) - Yt[b]) ** 2).mean()
        loss.backward(); opt.step(); tot += float(loss) * len(b)
    if ep % 10 == 0 or ep == 59:
        print(f"epoch {ep} mse {tot/len(idx):.4f}", flush=True)
net.eval()
with torch.no_grad():
    P = net(Xn).numpy()
is_or = np.zeros(len(X), bool); is_or[:N_OR] = True
for name, m in (("oracle-CFL", is_or), ("demo-tasks", ~is_or)):
    r2 = 1 - ((Y[m] - P[m]) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum()
    print(f"train-fit R2 {name}: {r2:.3f}", flush=True)
torch.save({"in_dim": X.shape[1], "hidden": [256, 256], "H": H, "AD": gc.AD, "K": 5,
            "state_dict": net.state_dict(), "mu": mu.astype(np.float32),
            "sd": sd.astype(np.float32), "tasks": TASKS4}, OUT)
print("saved", OUT); print("ORACLE_DISTILL_DONE", flush=True)
