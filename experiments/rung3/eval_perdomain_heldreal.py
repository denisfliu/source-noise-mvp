"""Per-domain U stage A follow-up: held-out REAL c-R^2 for each (prior, basis) pair.
The builder prints aggregate held R^2; for mixed-data priors that pools domains, and the
per-domain-U decision needs the real-rows-only number. Replicates make_progress_prior4's
frozen split and row construction (stride 4, NOPROG inputs). CPU-only (JAX_PLATFORMS=cpu);
the policy is loaded solely for _input_transform.
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
DD = f"{RD}/data_gate_synth"
HFB = "/home/ubuntu/hf_bundle/gate-drone-pi0"
H, AD, K = 50, 32, 5
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = LEFT.replace("left", "right")
CFL = "go through the center gate from the left and hover over the stuffed animal"
CFR = "go through the center gate from the right and hover over the stuffed animal"
TASKS4 = [CFL, CFR, LEFT, RIGHT]

import openpi.shared.normalize as NZ
from openpi import transforms as T
from openpi.transforms import NormStats
import openpi.training.config as C
import openpi.policies.policy_config as PC

ns = NZ.load(f"{HFB}/assets/gate_nav")

def pads(nsd, dim):
    out = {}
    for k, s in nsd.items():
        n = np.asarray(s.mean).shape[-1]
        if n >= dim:
            out[k] = s; continue
        p = dim - n
        ext = lambda a, f: None if a is None else np.concatenate([np.asarray(a, np.float32), np.full(p, f, np.float32)])
        out[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
    return out

cfg = C.get_config("pi0_gate")
nsp = pads(ns, cfg.model.action_dim)
policy = PC.create_trained_policy(cfg, f"{HFB}/checkpoints/gate_both_pin", norm_stats=nsp)
nrm = T.Normalize(nsp, use_quantiles=False)
_D = np.zeros((224, 224, 3), np.uint8)

def chunk_norm(chunk7):
    L = len(chunk7)
    ch = np.zeros((H, AD), np.float32)
    m = min(L, H)
    ch[:m, :7] = chunk7[:m]
    if m < H:
        ch[m:, :7] = chunk7[m - 1]
    return nrm({"actions": ch})["actions"].reshape(-1)

real_meta = json.load(open(f"{RD}/data_gate_real/meta.json"))

def files_for(data):
    files = []
    if data in ("synth", "synth+real"):
        files += [("synth", f) for f in sorted(glob.glob(f"{DD}/ep_*.npz"))]
    if data in ("real", "synth+real"):
        files += [("real", f"{RD}/data_gate_real/{k}.npz") for k in sorted(real_meta)]
    return files

def held_real_rows(data):
    """(model_state+onehot inputs, normalized 1600-d chunks) for held REAL episodes."""
    files = files_for(data)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(files))
    held = set(perm[int(0.8 * len(files)):].tolist())
    X, Ych = [], []
    for ei, (kind, f) in enumerate(files):
        if ei not in held or kind != "real":
            continue
        d = np.load(f, allow_pickle=True)
        st = d["state"].astype(np.float32); ac = d["action"].astype(np.float32)
        lang = real_meta[os.path.basename(f)[:-4]]["lang"]
        oh = np.zeros(4, np.float32); oh[TASKS4.index(lang)] = 1.0
        Tn = len(st)
        for t in range(0, Tn - H, 4):
            ms = np.asarray(policy._input_transform(
                {"observation/image": _D, "observation/wrist_image": _D,
                 "observation/state": st[t], "prompt": lang})["state"]).reshape(-1)
            X.append(np.concatenate([ms, oh]).astype(np.float32))
            Ych.append(chunk_norm(ac[t:t + H]))
    return np.array(X, np.float32), np.array(Ych, np.float32)

rows = {d: held_real_rows(d) for d in ("real", "synth+real")}
print({d: rows[d][0].shape for d in rows}, flush=True)

def build(indim):
    return nn.Sequential(nn.Linear(indim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, K))

CASES = [
    ("noprog_prior_realU.pt",          "pin_U_gate_rrr_real_k5.npy",   "real"),
    ("noprog_prior_realdata_synthU.pt", "pin_U_gate_rrr_k5.npy",        "real"),
    ("noprog_prior_pooledU.pt",         "pin_U_gate_rrr_pooled_k5.npy", "synth+real"),
    ("noprog_prior_mixed.pt",           "pin_U_gate_rrr_k5.npy",        "synth+real"),
]
for pt, upath, data in CASES:
    ck = torch.load(f"{RD}/{pt}", map_location="cpu", weights_only=False)
    U = np.load(f"{RD}/{upath}").astype(np.float32)
    X, Ych = rows[data]
    Yc = Ych @ U
    net = build(ck["in_dim"]); net.load_state_dict(ck["state_dict"]); net.eval()
    with torch.no_grad():
        pr = net(torch.tensor((X - ck["mu"]) / ck["sd"])).numpy()
    r2 = 1 - ((Yc - pr) ** 2).sum() / (((Yc - Yc.mean(0)) ** 2).sum() + 1e-9)
    per = 1 - ((Yc - pr) ** 2).sum(0) / (((Yc - Yc.mean(0)) ** 2).sum(0) + 1e-9)
    print(f"{pt:36s} U={upath:30s} held-REAL c-R2 {r2:+.3f}  per-dim {np.round(per, 2)}", flush=True)
print("HELDREAL_DONE", flush=True)
