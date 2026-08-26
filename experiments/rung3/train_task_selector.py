"""VLM task SELECTOR: 4-way classifier on fused (rendered-domain) prefix features.

Replaces the task one-hot in the winning no-clock prior (north-star non-negotiable:
one-hot is a scaffold). The command factorization stays split: state carries
geometry (closed-loop-proven), the VLM carries only the discrete task choice —
a classification, which fused features ground (within-scene axes cos 0.92-1.0)
and which must clear the paraphrase bar that killed frozen text encoders.

Training rows: [X_rend, label = its episode's task] + [X_rendlr (same frames,
within-family swapped prompt), label = the SWAPPED prompt's task] — the paired
rows dissociate language from scene, forcing the classifier to read the prompt.
Grouped split by source episode (frozen rng(0) 160/40).

Gate a printed here: held accuracy on true and swapped rows separately.
Exports rung3/task_selector.npz (mu, sg, W1,b1,W2,b2 — GELU-tanh MLP, numpy
forward at serving via gate_ctx_common-style loader).
"""
import glob
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
TASKS = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]  # matches prior tasks order
SWAP = {gc.PROMPT_L: gc.PROMPT_R, gc.PROMPT_R: gc.PROMPT_L,
        gc.PROMPT_CFL: gc.PROMPT_CFR, gc.PROMPT_CFR: gc.PROMPT_CFL}
torch.manual_seed(0); np.random.seed(0)

src = gc.load_eps(with_images=False)
rng = np.random.default_rng(0)
idx = rng.permutation(len(src)); trep = set(idx[:160].tolist())

groups = []
for si, e in enumerate(src):
    groups.append((si, "orig", e)); groups.append((si, "reverse", ta.reverse(e)))
    for f in (ta.crop_to_gate, ta.crop_from_gate):
        a = f(e)
        if a is not None:
            groups.append((si, a["lang"], a))  # variant name unused below; keep ep
    groups.append((si, "hover", ta.hover(e, len(e["action"]) // 2)))
recs = []
for si, _v, e in groups:
    n = min(len(e["action"]), len(e["state"]) - 1)
    for t in range(0, n, STRIDE):
        recs.append(dict(si=si, lang=e["lang"], tr=si in trep,
                         is_orig=e["lang"] in TASKS and _v == "orig"))
X = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{RUN}/Xrendshard_*.npy"))], 0)
Xs = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{RUN}/Xrendlrshard_*.npy"))], 0)
assert len(X) == len(recs)
orig_ix = [i for i, r in enumerate(recs) if r["is_orig"]]
assert len(Xs) == len(orig_ix), (len(Xs), len(orig_ix))

# rows: true-prompt ORIG rows (labeled by episode task) + swapped rows (labeled by swapped task)
Xt_rows = X[orig_ix]
y_true = np.array([TASKS.index(recs[i]["lang"]) for i in orig_ix])
y_swap = np.array([TASKS.index(SWAP[recs[i]["lang"]]) for i in orig_ix])
tr_mask = np.array([recs[i]["tr"] for i in orig_ix])

Xall = np.concatenate([Xt_rows, Xs], 0)
yall = np.concatenate([y_true, y_swap], 0)
trall = np.concatenate([tr_mask, tr_mask], 0)
if os.environ.get("PARA", "0") == "1":
    # paraphrase-augmented rows (TRAIN episodes only by construction; the gate-b
    # eval paraphrases are a disjoint hand-authored set)
    Xp = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{RUN}/Xparashard_*.npy"))], 0)
    pm = np.load(f"{RUN}/parameta.npz")
    assert len(Xp) == len(pm["task"]), (len(Xp), len(pm["task"]))
    Xall = np.concatenate([Xall, Xp], 0)
    yall = np.concatenate([yall, pm["task"]], 0)
    trall = np.concatenate([trall, np.ones(len(Xp), bool)], 0)
    print("paraphrase rows folded in:", len(Xp), flush=True)
mu = Xall[trall].mean(0); sg = Xall[trall].std(0) + 1e-6
Xn = torch.tensor((Xall - mu) / sg, dtype=torch.float32)
yt = torch.tensor(yall, dtype=torch.long)

net = nn.Sequential(nn.Linear(2048, 128), nn.GELU(approximate="tanh"), nn.Linear(128, 4))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
tri = np.where(trall)[0]
for ep in range(40):
    perm = np.random.permutation(tri)
    for i in range(0, len(perm), 512):
        b = perm[i:i + 512]; opt.zero_grad()
        nn.functional.cross_entropy(net(Xn[b]), yt[b]).backward(); opt.step()
net.eval()
with torch.no_grad():
    pred = net(Xn).argmax(1).numpy()
n_or = len(orig_ix)
for name, sl in (("true-prompt", slice(0, n_or)), ("swapped-prompt", slice(n_or, None))):
    m = ~trall[sl]
    acc = (pred[sl][m] == yall[sl][m]).mean()
    print("GATE-a held accuracy %-14s %.3f (n=%d)" % (name, acc, m.sum()), flush=True)
per = {}
for k, t in enumerate(TASKS):
    m = (~trall) & (yall == k)
    per[t.split()[4] if "center" not in t else "center-" + t.split()[6]] = float((pred[m] == k).mean())
print("per-task held acc:", {k: round(v, 3) for k, v in per.items()}, flush=True)

L = [m for m in net if isinstance(m, nn.Linear)]
np.savez(os.path.join(gc.RD, "task_selector.npz"), mu=mu.astype(np.float32), sg=sg.astype(np.float32),
         W1=L[0].weight.detach().numpy().T, b1=L[0].bias.detach().numpy(),
         W2=L[1].weight.detach().numpy().T, b2=L[1].bias.detach().numpy(),
         tasks=np.array(TASKS))
print("saved task_selector.npz"); print("SELECTOR_DONE", flush=True)
