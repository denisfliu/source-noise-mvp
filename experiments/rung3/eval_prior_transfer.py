"""Sim->real prior transfer evaluation (Denis's inferred-real-center hypothesis).

Evaluates four no-clock priors on the SAME held-out REAL episodes, per task:
  sim-only        — trained on synth 4-task only (noprog_prior_rrr4.pt)
  real-only       — trained on real 2-task only (noprog_prior_real.pt)
  sim->FT both    — warm-start sim, fine-tune on real both gate tasks
  sim->FT LEFT    — warm-start sim, fine-tune on real LEFT ONLY
The key cell: FT-LEFT's held R2 on real RIGHT — does the sim->real correction
learned on one task transfer to the other? If yes (clean-label retest of the old
negative), the correction is task-independent and an inferred real CENTER c is
plausible. Also reports each prior's held R2 on SYNTH CENTER rows (forgetting).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc

RD = gc.RD
U = np.load(os.path.join(RD, "pin_U_gate_rrr_k5.npy"))
ns, amean, astd = gc.load_norm()
H, AD = gc.H, gc.AD
TASKS4 = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]

policy = gc.make_policy()
_D = np.zeros((224, 224, 3), np.uint8)
def mstate(raw, lang):
    return np.asarray(policy._input_transform(
        {"observation/image": _D, "observation/wrist_image": _D,
         "observation/state": raw, "prompt": lang})["state"]).reshape(-1)

def rows_from(files, langs):
    X, Y, T = [], [], []
    for f, lang in zip(files, langs):
        d = np.load(f, allow_pickle=True)
        st = d["state"].astype(np.float32); ac = d["action"].astype(np.float32)
        oh = np.zeros(4, np.float32); oh[TASKS4.index(lang)] = 1.0
        for t in range(0, len(st) - H, 8):
            X.append(np.concatenate([mstate(st[t], lang), oh]).astype(np.float32))
            Y.append(gc.segY(ac[t:], amean, astd) @ U)
            T.append(lang)
    return np.stack(X), np.stack(Y).astype(np.float32), np.array(T)

REAL = os.path.join(RD, "data_gate_real")
meta = json.load(open(os.path.join(REAL, "meta.json")))
keys = sorted(meta)
rng = np.random.default_rng(0)
perm = rng.permutation(len(keys))
held_keys = [keys[i] for i in perm[int(0.8 * len(keys)):]]
Xr, Yr, Tr = rows_from([f"{REAL}/{k}.npz" for k in held_keys],
                       [meta[k]["lang"] for k in held_keys])
print("held real rows:", len(Xr), flush=True)

eps = gc.load_eps(with_images=False)
sidx = rng.permutation(len(eps))
held_center = [i for i in sidx[160:] if eps[i]["lang"] in (gc.PROMPT_CFL, gc.PROMPT_CFR)][:8]
files_c = [f"{gc.DD}/ep_{i:04d}.npz" for i in held_center]
Xc, Yc, Tc = rows_from(files_c, [eps[i]["lang"] for i in held_center])
print("held synth-center rows:", len(Xc), flush=True)

def load_prior(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    layers, din = [], d["in_dim"]
    for h in d["hidden"]:
        layers += [nn.Linear(din, h), nn.SiLU()]; din = h
    layers += [nn.Linear(din, 5)]
    net = nn.Sequential(*layers); net.load_state_dict(d["state_dict"]); net.eval()
    return net, d["mu"], d["sd"]

def r2(net, mu, sd, X, Y, mask=None):
    if mask is not None:
        X, Y = X[mask], Y[mask]
    with torch.no_grad():
        P = net(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy()
    return float(1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9))

print("%-14s %12s %12s %14s" % ("prior", "real-LEFT", "real-RIGHT", "synth-CENTER"))
for name, path in (("sim-only", "noprog_prior_rrr4.pt"), ("real-only", "noprog_prior_real.pt"),
                   ("sim>FT-both", "noprog_prior_simft.pt"), ("sim>FT-LEFT", "noprog_prior_ftleft.pt"), ("mixed", "noprog_prior_mixed.pt")):
    net, mu, sd = load_prior(os.path.join(RD, path))
    print("%-14s %12.3f %12.3f %14.3f" % (
        name, r2(net, mu, sd, Xr, Yr, Tr == gc.PROMPT_L),
        r2(net, mu, sd, Xr, Yr, Tr == gc.PROMPT_R),
        r2(net, mu, sd, Xc, Yc)), flush=True)
print("TRANSFER_EVAL_DONE", flush=True)
