"""Right-gate aiming-bias attribution (command-source-independent diagnostic).

The record config misses the right aperture with a consistent ~1 m +x overshoot
(0/10, gate_success 2026-08-05). Two suspects, separated teacher-forced on
held-out RIGHT demos at approach-phase frames:

  arm TRUE-C  — flow pinned with the demo chunk's own c
                -> bias here means the FLOW learned an asymmetric right-gate
                   behavior (fix is data/flow-side; no command source helps)
  arm PRIOR-C — flow pinned with the progress-prior's c (input construction
                mirrors serve_gate_pin_prog: [transformed_state, onehot, t/T],
                standardized with the prior's own mu/sd)
                -> extra bias here means the PRIOR mispredicts right-side
                   commands (fix is command-side; the grounded source inherits
                   the job)

Metrics per arm (whole-trajectory, B1 rule): 25-step ADE [m], and the aim —
executed net displacement error, x-component reported separately (the observed
overshoot axis). LEFT demos run as the control (where closed-loop succeeds).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc

RRRCK = os.path.expanduser("~/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999")
PRIOR = os.path.join(gc.RD, "prog_prior_rrr.pt")
N_EP, N_T, EVAL_T = 10, 5, 25

ns, amean, astd = gc.load_norm()
eps = gc.load_eps(with_images=True)
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
rng = np.random.default_rng(0)
idx = rng.permutation(len(eps)); heldout = set(idx[int(0.8 * len(eps)):].tolist())

d = torch.load(PRIOR, map_location="cpu", weights_only=False)
layers, din = [], d["in_dim"]
import torch.nn as nn
for h in d["hidden"]:
    layers += [nn.Linear(din, h), nn.SiLU()]; din = h
layers += [nn.Linear(din, 5)]
prior = nn.Sequential(*layers); prior.load_state_dict(d["state_dict"]); prior.eval()
mu, sd = d["mu"] if "mu" in d else d.get("mn"), d["sd"]
TASKS = d["tasks"]

policy = gc.make_policy(RRRCK)


def prior_c(obs, prompt, prog):
    ms = np.asarray(policy._input_transform(dict(obs))["state"]).reshape(-1)
    oh = np.zeros(len(TASKS), np.float32); oh[TASKS.index(prompt)] = 1.0
    x = np.concatenate([ms, oh, [np.float32(prog)]]).astype(np.float32)
    xn = (x - mu) / sd
    with torch.no_grad():
        return prior(torch.tensor(xn[None]))[0].numpy().astype(np.float32)


def run_arm(side, use_prior):
    prompt = gc.PROMPT_L if side == "left" else gc.PROMPT_R
    rows = []
    cand = [i for i in heldout if (eps[i]["lang"] == prompt)][:N_EP]
    for ei in cand:
        e = eps[ei]; n = min(len(e["action"]), len(e["state"]) - 1)
        for t in np.linspace(0, max(0, n - gc.H), N_T).astype(int):
            obs = gc.mkobs(e, t)
            if use_prior:
                c = prior_c(obs, prompt, t / max(1, n))
            else:
                c = gc.segY(e["action"][t:], amean, astd) @ U
            g = np.random.default_rng(int(t) + 1).standard_normal((gc.H, gc.AD)).astype(np.float32).reshape(-1)
            noise = (g - (g @ U) @ U.T + (c @ U.T)).reshape(gc.H, gc.AD).astype(np.float32)
            pred = np.asarray(policy.infer(obs, noise=noise)["actions"])[:, :4]
            true = e["action"][t:t + EVAL_T, :4]
            m = len(true)
            ade = float(np.linalg.norm(np.cumsum(pred[:m, :3], 0) - np.cumsum(true[:m, :3], 0), axis=1).mean())
            aim = pred[:m, :3].sum(0) - true[:m, :3].sum(0)
            rows.append((ade, aim[0], aim[1], aim[2]))
    a = np.array(rows)
    print("%-5s %-8s n=%3d  ADE %.3f±%.3f   aim-err x %+.3f±%.3f  y %+.3f  z %+.3f"
          % (side, "PRIOR-C" if use_prior else "TRUE-C", len(rows),
             a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(),
             a[:, 2].mean(), a[:, 3].mean()), flush=True)


for side in ("right", "left"):
    for use_prior in (False, True):
        run_arm(side, use_prior)
print("ATTRIBUTION_DONE", flush=True)
