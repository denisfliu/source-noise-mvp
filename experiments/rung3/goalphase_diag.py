"""Goal-phase command diagnosis (queue #1, 2026-08-08): the component battery showed the
flow can execute the goal phase under demo-oracle commands (right full 3/5) but not under
the MLP prior (0). Compare prior c vs true demo c along each demo, split by trajectory
phase (early / transit / tail), on the gate tasks. CPU-only.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RD = os.path.dirname(os.path.abspath(__file__))
HFB = "/home/ubuntu/hf_bundle/gate-drone-pi0"
H, AD = 50, 32
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = LEFT.replace("left", "right")
CFL = "go through the center gate from the left and hover over the stuffed animal"
CFR = "go through the center gate from the right and hover over the stuffed animal"
TASKS4 = [CFL, CFR, LEFT, RIGHT]

import openpi.shared.normalize as NZ
import openpi.training.config as C
import openpi.policies.policy_config as PC
from eval_perdomain_heldreal import pads

ns = NZ.load(f"{HFB}/assets/gate_nav")
cfg = C.get_config("pi0_gate")
nsp = pads(ns, cfg.model.action_dim)
policy = PC.create_trained_policy(cfg, f"{HFB}/checkpoints/gate_both_pin", norm_stats=nsp)
amean = np.asarray(nsp["actions"].mean); astd = np.asarray(nsp["actions"].std)
U = np.load(f"{RD}/pin_U_gate_rrr_k5.npy").astype(np.float32)

d = torch.load(os.environ.get("PRIOR", f"{RD}/noprog_prior_rrr4.pt"), map_location="cpu", weights_only=False)
layers, din = [], d["in_dim"]
for hdim in d["hidden"]:
    layers += [nn.Linear(din, hdim), nn.SiLU()]; din = hdim
layers += [nn.Linear(din, d["K"])]
prior = nn.Sequential(*layers); prior.load_state_dict(d["state_dict"]); prior.eval()
mu, sd = d["mu"].astype(np.float32), d["sd"].astype(np.float32)

def c_of(chunk7):
    L = len(chunk7)
    ch = np.zeros((H, AD), np.float32); m = min(L, H)
    ch[:m, :7] = (chunk7[:m] - amean[:7]) / (astd[:7] + 1e-6)
    if m < H: ch[m:, :7] = ch[m - 1, :7]
    return ch.reshape(-1) @ U

_D = np.zeros((224, 224, 3), np.uint8)
def prior_c(state, lang):
    ms = np.asarray(policy._input_transform(
        {"observation/image": _D, "observation/wrist_image": _D,
         "observation/state": state, "prompt": lang})["state"]).reshape(-1)
    oh = np.zeros(4, np.float32); oh[TASKS4.index(lang)] = 1.0
    x = np.concatenate([ms, oh]).astype(np.float32)
    with torch.no_grad():
        return prior(torch.tensor(((x - mu) / sd)[None]))[0].numpy()

# gate-task synth demos: eps 100-149 LEFT, 150-199 RIGHT
files = sorted(glob.glob(f"{RD}/data_gate_synth/ep_*.npz"))
phases = {"early (0-50%)": (0.0, 0.5), "transit (50-75%)": (0.5, 0.75), "tail (75-100%)": (0.75, 1.0)}
stats = {lang: {ph: {"err": [], "cn_true": [], "cn_prior": []} for ph in phases} for lang in (LEFT, RIGHT)}
for i in range(100, 200, 4):  # every 4th demo for speed
    dd = np.load(files[i], allow_pickle=True)
    st = dd["state"].astype(np.float32); ac = dd["action"].astype(np.float32)
    lang = LEFT if i < 150 else RIGHT
    T = len(st)
    for t in range(0, T - 5, 8):
        frac = t / (T - 1)
        ph = next(k for k, (a, b) in phases.items() if a <= frac < b or (b == 1.0 and frac >= a))
        ct = c_of(ac[t:t + H]); cp = prior_c(st[t], lang)
        s = stats[lang][ph]
        s["err"].append(np.abs(cp - ct)); s["cn_true"].append(np.linalg.norm(ct)); s["cn_prior"].append(np.linalg.norm(cp))

for lang in (LEFT, RIGHT):
    print(f"\n== {('LEFT' if lang == LEFT else 'RIGHT')}")
    for ph in phases:
        s = stats[lang][ph]
        err = np.mean(s["err"], 0)
        print(f"  {ph:18s} n={len(s['err']):4d}  |c|_true {np.mean(s['cn_true']):6.2f}  |c|_prior {np.mean(s['cn_prior']):6.2f}"
              f"  per-dim |err| {np.round(err, 2)}  rel {np.round(err / (np.std([*s['cn_true']]) + 1e-6), 2) if False else ''}")
print("\nGOALPHASE_DIAG_DONE", flush=True)
