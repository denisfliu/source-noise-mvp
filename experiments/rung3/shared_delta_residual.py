"""Shared task-independent residual for sim->real (leave-one-task-out).

Freeze the sim prior (noprog_prior_rrr4.pt). Fit Δ(state) — NO task one-hot, so
task-independent by construction — on real LEFT train rows only, target
R = c_true - prior(x). Test: held real LEFT (sanity) and held real RIGHT
(the transfer cell: Δ never saw any right data). Also synth-center rows with Δ
applied (does the correction poison sim tasks?).

Decision comparisons (2026-08-05 transfer table, same held split):
  sim-only  real-L -0.32  real-R 0.00 ; mixed (deploy candidate) 0.66/0.75.
If prior+Δ_left ~= mixed on real RIGHT, the sim->real correction is
task-independent and an inferred real CENTER c is plausible. FT-on-left's
-1e8 pathology is avoided because the prior is frozen and Δ is ridge-linear.
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
H = gc.H
TASKS4 = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]

policy = gc.make_policy()
_D = np.zeros((224, 224, 3), np.uint8)


def mstate(raw, lang):
    return np.asarray(policy._input_transform(
        {"observation/image": _D, "observation/wrist_image": _D,
         "observation/state": raw, "prompt": lang})["state"]).reshape(-1)


def rows_from(files, langs):
    X, Y = [], []
    for f, lang in zip(files, langs):
        d = np.load(f, allow_pickle=True)
        st = d["state"].astype(np.float32); ac = d["action"].astype(np.float32)
        oh = np.zeros(4, np.float32); oh[TASKS4.index(lang)] = 1.0
        for t in range(0, len(st) - H, 8):
            X.append(np.concatenate([mstate(st[t], lang), oh]).astype(np.float32))
            Y.append(gc.segY(ac[t:], amean, astd) @ U)
    return np.stack(X), np.stack(Y).astype(np.float32)


d = torch.load(os.path.join(RD, "noprog_prior_rrr4.pt"), map_location="cpu", weights_only=False)
layers, din = [], d["in_dim"]
for h in d["hidden"]:
    layers += [nn.Linear(din, h), nn.SiLU()]; din = h
layers += [nn.Linear(din, 5)]
prior = nn.Sequential(*layers); prior.load_state_dict(d["state_dict"]); prior.eval()
pmu, psd = d["mu"], d["sd"]


def prior_pred(X):
    with torch.no_grad():
        return prior(torch.tensor((X - pmu) / psd, dtype=torch.float32)).numpy()


REAL = os.path.join(RD, "data_gate_real")
meta = json.load(open(os.path.join(REAL, "meta.json")))
keys = sorted(meta)
rng = np.random.default_rng(0)
perm = rng.permutation(len(keys))
ntr = int(0.8 * len(keys))
tr_keys = [keys[i] for i in perm[:ntr]]
he_keys = [keys[i] for i in perm[ntr:]]


def per_task(klist, task):
    ks = [k for k in klist if meta[k]["lang"] == task]
    return rows_from([f"{REAL}/{k}.npz" for k in ks], [meta[k]["lang"] for k in ks])


Xl_tr, Yl_tr = per_task(tr_keys, gc.PROMPT_L)
Xl_he, Yl_he = per_task(he_keys, gc.PROMPT_L)
Xr_he, Yr_he = per_task(he_keys, gc.PROMPT_R)
print("rows: left-train %d  left-held %d  right-held %d" % (len(Xl_tr), len(Xl_he), len(Xr_he)), flush=True)

# synth-center held rows (forgetting check), same frozen split as eval_prior_transfer
eps = gc.load_eps(with_images=False)
sidx = rng.permutation(len(eps))
held_center = [i for i in sidx[160:] if eps[i]["lang"] in (gc.PROMPT_CFL, gc.PROMPT_CFR)][:8]
Xc, Yc = rows_from([f"{gc.DD}/ep_{i:04d}.npz" for i in held_center],
                   [eps[i]["lang"] for i in held_center])

# Δ inputs: transformed state only (drop the trailing one-hot block)
S = lambda X: X[:, :-4]
smu, ssd = S(Xl_tr).mean(0), S(Xl_tr).std(0) + 1e-6
Rtr = Yl_tr - prior_pred(Xl_tr)


def fit_ridge(lam):
    Z = np.concatenate([(S(Xl_tr) - smu) / ssd, np.ones((len(Xl_tr), 1))], 1)
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Rtr)
    return lambda X: np.concatenate([(S(X) - smu) / ssd, np.ones((len(X), 1))], 1) @ W


def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9))


# lam chosen by 5-fold CV on LEFT train rows ONLY (right stays untouched)
folds = np.arange(len(Xl_tr)) % 5
best, blam = -1e9, None
for lam in (1.0, 10.0, 100.0, 1000.0):
    sc = []
    for f in range(5):
        m = folds != f
        Z = np.concatenate([(S(Xl_tr[m]) - smu) / ssd, np.ones((m.sum(), 1))], 1)
        W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Rtr[m])
        Zv = np.concatenate([(S(Xl_tr[~m]) - smu) / ssd, np.ones(((~m).sum(), 1))], 1)
        sc.append(r2(Rtr[~m], Zv @ W))
    if np.mean(sc) > best:
        best, blam = np.mean(sc), lam
print("Δ ridge lam=%g (CV residual-R2 %.3f on left-train)" % (blam, best), flush=True)
delta = fit_ridge(blam)

print("%-22s %10s %10s %12s" % ("model", "real-L", "real-R", "synth-CENTER"))
for name, fn in (("sim prior", prior_pred),
                 ("prior + Δ_left(state)", lambda X: prior_pred(X) + delta(X))):
    print("%-22s %10.3f %10.3f %12.3f" % (
        name, r2(Yl_he, fn(Xl_he)), r2(Yr_he, fn(Xr_he)), r2(Yc, fn(Xc))), flush=True)
print("reference: mixed co-trained prior 0.66 / 0.75 / 0.98 (2026-08-05 table)")
print("SHARED_DELTA_DONE", flush=True)
