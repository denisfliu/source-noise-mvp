"""VLM gate-locator: head on cached fused (rendered-domain) features -> the 3D mocap
anchor of the gate NAMED IN THE PROMPT. Replaces the waypoint oracle's hand-supplied
gate position with VLM perception+language. Trains on the existing clean rendered
cache (no new extraction); labels from the published scene YAML anchors per task.
Eval: held-episode meters error per task. CPU.
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
torch.manual_seed(0); np.random.seed(0)
# anchor of the gate each task's prompt names (scene YAMLs; z = aperture mid-height)
ANCHOR = {gc.PROMPT_L:  np.array([0.861, 0.694, 1.075]),
          gc.PROMPT_R:  np.array([0.544, -1.147, 1.0]),
          gc.PROMPT_CFL: np.array([2.756, -0.3275, 1.0]),
          gc.PROMPT_CFR: np.array([2.756, -0.3275, 1.0])}

src = gc.load_eps(with_images=False)
rng = np.random.default_rng(0)
idx = rng.permutation(len(src)); trep = set(idx[:160].tolist())
groups = []
for si, e in enumerate(src):
    groups.append((si, e)); groups.append((si, ta.reverse(e)))
    for f in (ta.crop_to_gate, ta.crop_from_gate):
        a = f(e)
        if a is not None:
            groups.append((si, a))
    groups.append((si, ta.hover(e, len(e["action"]) // 2)))
recs = []
for si, e in groups:
    n = min(len(e["action"]), len(e["state"]) - 1)
    for t in range(0, n, STRIDE):
        # label only rows whose prompt names a gate (aug prompts like "hold position" skip)
        recs.append(dict(si=si, lang=e["lang"], tr=si in trep,
                         y=ANCHOR.get(e["lang"])))
X = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{RUN}/Xrendshard_*.npy"))], 0)
assert len(X) == len(recs)
keep = [i for i, r in enumerate(recs) if r["y"] is not None]
Xk = X[keep]; Y = np.stack([recs[i]["y"] for i in keep]).astype(np.float32)
tr = np.array([recs[i]["tr"] for i in keep]); te = ~tr
langs = np.array([recs[i]["lang"] for i in keep])
mu, sg = Xk[tr].mean(0), Xk[tr].std(0) + 1e-6
net = nn.Sequential(nn.Linear(2048, 128), nn.GELU(approximate="tanh"), nn.Linear(128, 3))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
Xn = torch.tensor((Xk - mu) / sg, dtype=torch.float32); Yt = torch.tensor(Y)
tri = np.where(tr)[0]
for ep in range(40):
    perm = np.random.permutation(tri)
    for i in range(0, len(perm), 512):
        b = perm[i:i + 512]; opt.zero_grad()
        ((net(Xn[b]) - Yt[b]) ** 2).mean().backward(); opt.step()
net.eval()
with torch.no_grad():
    P = net(Xn).numpy()
for task in (gc.PROMPT_L, gc.PROMPT_R, gc.PROMPT_CFL, gc.PROMPT_CFR):
    m = te & (langs == task)
    err = np.linalg.norm(P[m] - Y[m], axis=1)
    print("LOCATOR held err %-70s %.3f±%.3f m" % (task[:66], err.mean(), err.std()), flush=True)
L = [m for m in net if isinstance(m, nn.Linear)]
np.savez(os.path.join(gc.RD, "gate_locator.npz"), mu=mu.astype(np.float32), sg=sg.astype(np.float32),
         W1=L[0].weight.detach().numpy().T, b1=L[0].bias.detach().numpy(),
         W2=L[1].weight.detach().numpy().T, b2=L[1].bias.detach().numpy())
print("saved gate_locator.npz"); print("LOCATOR_DONE", flush=True)
