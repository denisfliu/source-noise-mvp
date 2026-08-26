"""Combined c-map: MLP on RENDERED features + cfground L/R counterfactual rows.

Ingredients (each the antidote to a measured deficiency):
- rendered-domain features         -> serving-domain validity (G4r: first THROUGH)
- MLP capacity                     -> pointwise prompt disambiguation (ridge flat at 0.68 m)
- cfground L/R rows                -> left/right identifiability (scene-confounded in true prompts)
- (fwd/back grounding is already real, from reversal rows)

Training rows: [X_rend true prompts -> demo c] + [X_rendlr (orig rows, L<->R swapped
prompt) -> phase-matched counterfactual c from the opposite side].

Exports rung3/vlmc_mlp_rend.npz with mu,sg,W1,b1,W2,b2,W3,b3,clo,chi (GELU MLP,
numpy-forward at serving — serve_gate_pin_vlmc detects the schema). Prints the
full offline gate panel: held R2, endpoint disambiguation errs, fwd/back + L/R
grounding alignment. CPU-only.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import gate_traj_algebra as ta

RUN = os.path.expanduser("~/ctxrun")
STRIDE = 12
torch.manual_seed(0); np.random.seed(0)

ns, amean, astd = gc.load_norm()
src = gc.load_eps(with_images=False)
rng = np.random.default_rng(0)
idx = rng.permutation(len(src)); trep = set(idx[:160].tolist())

groups = []
for si, e in enumerate(src):
    groups.append((si, "orig", e)); groups.append((si, "reverse", ta.reverse(e)))
    for nm, f in (("crop_to", ta.crop_to_gate), ("crop_from", ta.crop_from_gate)):
        a = f(e)
        if a is not None:
            groups.append((si, nm, a))
    groups.append((si, "hover", ta.hover(e, len(e["action"]) // 2)))
recs = []
for si, v, e in groups:
    n = min(len(e["action"]), len(e["state"]) - 1)
    for t in range(0, n, STRIDE):
        recs.append(dict(si=si, v=v, t=t, n=n, prog=t / max(1, n), lang=e["lang"],
                         side_L=e["lang"] == gc.PROMPT_L,
                         Y=gc.segY(e["action"][t:], amean, astd)))
import glob as _glob
X = np.concatenate([np.load(f) for f in sorted(_glob.glob(f"{RUN}/Xrendshard_*.npy"))], 0)
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
C = np.stack([r["Y"] for r in recs]).astype(np.float32) @ U
tr = np.array([r["si"] in trep for r in recs]); te = ~tr
assert len(X) == len(recs)

# cfground rows: swapped-prompt features of ORIG recs, targets = opposite side's
# phase-binned mean c (train episodes only)
Xs = np.concatenate([np.load(f) for f in sorted(_glob.glob(f"{RUN}/Xrendlrshard_*.npy"))], 0)
# Counterfactual rows are FAMILY-MATCHED (2026-08-05): the swap cache holds every
# orig row with its within-family swapped prompt (L<->R, CFL<->CFR) — never across
# families. cf targets therefore come from the SWAPPED task's own rows.
gate_langs = (gc.PROMPT_L, gc.PROMPT_R)
SWAP = {gc.PROMPT_L: gc.PROMPT_R, gc.PROMPT_R: gc.PROMPT_L,
        gc.PROMPT_CFL: gc.PROMPT_CFR, gc.PROMPT_CFR: gc.PROMPT_CFL}
orig_ix = [i for i, r in enumerate(recs) if r["v"] == "orig"]
assert len(Xs) == len(orig_ix), (len(Xs), len(orig_ix))
NPH = 20
bins = np.array([min(NPH - 1, int(r["prog"] * NPH)) for r in recs])
sideL = np.array([r["side_L"] for r in recs])
vt = np.array([r["v"] for r in recs])
cf = np.zeros((len(orig_ix), 5), np.float32)
langs = np.array([r["lang"] for r in recs])
for k, i in enumerate(orig_ix):
    tgt_lang = SWAP[recs[i]["lang"]]
    msk = tr & (vt == "orig") & (langs == tgt_lang) & (bins == bins[i])
    cf[k] = C[msk].mean(0) if msk.any() else C[tr & (vt == "orig") & (langs == tgt_lang)].mean(0)
tr_cf = np.array([recs[i]["si"] in trep for i in orig_ix])

Xall = np.concatenate([X, Xs], 0)
Call = np.concatenate([C, cf], 0)
trall = np.concatenate([tr, tr_cf], 0)
mu = Xall[trall].mean(0); sg = Xall[trall].std(0) + 1e-6
Xt = torch.tensor((Xall - mu) / sg, dtype=torch.float32)
Ct = torch.tensor(Call, dtype=torch.float32)

net = nn.Sequential(nn.Linear(2048, 256), nn.GELU(approximate='tanh'), nn.Linear(256, 256), nn.GELU(approximate='tanh'), nn.Linear(256, 5))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
tri = np.where(trall)[0]
for ep in range(60):
    perm = np.random.permutation(tri)
    for i in range(0, len(perm), 512):
        b = perm[i:i + 512]; opt.zero_grad()
        ((net(Xt[b]) - Ct[b]) ** 2).mean().backward(); opt.step()
net.eval()
with torch.no_grad():
    P = net(Xt[:len(recs)]).numpy()
    Pswap = net(Xt[len(recs):]).numpy()

r2 = 1 - ((C[te] - P[te]) ** 2).sum() / ((C[te] - C[te].mean(0)) ** 2).sum()
print("combined MLP: held R2 (true rows) %.3f" % r2, flush=True)

def decode(c):
    return (U @ np.asarray(c)).reshape(gc.H, gc.AD)[:, :4].sum(0) * astd[:4]

for v, cond in (("reverse", lambda r: r["t"] < 24), ("hover", lambda r: True),
                ("orig", lambda r: r["t"] > r["n"] - 40)):
    rows = [i for i, r in enumerate(recs) if (not tr[i]) and r["v"] == v and cond(r)]
    d = np.stack([decode(P[j]) - decode(C[j]) for j in rows[:60]])
    print("  endpoint cmd err %-8s %.2f m" % (v, np.linalg.norm(d[:, :3], axis=1).mean()), flush=True)

# L/R grounding: held orig rows, prompt-swap shift vs behavioral L/R axis
held_k = [k for k, i in enumerate(orig_ix) if not tr[i]]
gate_mask = np.array([r["lang"] in gate_langs for r in recs])
bL = C[tr & (vt == "orig") & gate_mask & sideL].mean(0)
bR = C[tr & (vt == "orig") & gate_mask & ~sideL].mean(0)
b = bL - bR; bhat = b / np.linalg.norm(b)
d = np.stack([(P[orig_ix[k]] - Pswap[k]) * (1 if sideL[orig_ix[k]] else -1) for k in held_k]).mean(0)
print("L/R grounding: |target axis| %.2f  cos(pred, target) %+.3f  |pred|/|target| %.2f"
      % (np.linalg.norm(b), float(d @ bhat / (np.linalg.norm(d) + 1e-9)),
         np.linalg.norm(d) / np.linalg.norm(b)), flush=True)

# fwd/back grounding (identical-frame orig/reverse pairs, held)
bykey = {(recs[i]["si"], recs[i]["t"]): i for i in orig_ix}
pairs = [(bykey[(r["si"], r["n"] - r["t"])], i) for i, r in enumerate(recs)
         if r["v"] == "reverse" and not tr[i] and (r["si"], r["n"] - r["t"]) in bykey]
bfb = (C[[j for j, _ in pairs]] - C[[i for _, i in pairs]]).mean(0)
dfb = (P[[j for j, _ in pairs]] - P[[i for _, i in pairs]]).mean(0)
print("FWD/BACK grounding: cos %+.3f  mag %.2f" % (
    float(dfb @ bfb / (np.linalg.norm(dfb) * np.linalg.norm(bfb))),
    np.linalg.norm(dfb) / np.linalg.norm(bfb)), flush=True)

L = [m for m in net if isinstance(m, nn.Linear)]
out = os.path.join(gc.RD, "vlmc_mlp_rend.npz")
np.savez(out, mu=mu.astype(np.float32), sg=sg.astype(np.float32),
         W1=L[0].weight.detach().numpy().T, b1=L[0].bias.detach().numpy(),
         W2=L[1].weight.detach().numpy().T, b2=L[1].bias.detach().numpy(),
         W3=L[2].weight.detach().numpy().T, b3=L[2].bias.detach().numpy(),
         clo=gc.CLO, chi=gc.CHI)
print("saved", out, flush=True)
print("MLP_MAP_DONE", flush=True)
