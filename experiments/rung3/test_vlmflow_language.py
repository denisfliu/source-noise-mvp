"""Language-differentiation probe for the VLA-native two-stage head (Denis,
2026-08-06): SAME rendered frame, DIFFERENT prompt -> does the sampled c change,
and change CORRECTLY? Uses the cache's identical-frame pairs (forward row at
source-frame f vs reverse row at the same f; hover rows repeat one frame).
Held (frozen-split) episodes only. CPU.

Readouts per pair type, decoded to net-displacement meters:
  fwd vs back: cos(Delta c_sampled, Delta y_true) + magnitude ratio (the language
               axis must land on the behavioral axis, not just move)
  fwd vs hold: |cmd(hold)| should collapse toward 0 while cmd(fwd) stays.
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
RD = gc.RD
STRIDE = 12
U = np.load(os.path.join(RD, "pin_U_gate_rrr_k5.npy"))
ns, amean, astd = gc.load_norm()
H = gc.H

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
rows = []; ri = 0
for si, e in groups:
    n = min(len(e["action"]), len(e["state"]) - 1)
    ac = e["action"].astype(np.float32)
    for t in range(0, n, STRIDE):
        rows.append(dict(si=si, lang=e["lang"], f=int(e["fidx"][t]) if "fidx" in e else t, ri=ri,
                         y=(gc.segY(ac[t:], amean, astd) @ U).astype(np.float32),
                         held=si not in trep))
        ri += 1
X = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{RUN}/Xrendshard_*.npy"))], 0)
assert len(X) == len(rows)

d = torch.load(os.path.join(RD, "vlmflow_head_rend.pt"), map_location="cpu", weights_only=False)


class VNet(nn.Module):
    def __init__(self, xdim=2048, cdim=5, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(xdim + cdim + 1, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))

    def forward(self, ct, t, x):
        return self.net(torch.cat([ct, t, x], 1))


net = VNet(d["in_dim"]); net.load_state_dict(d["state_dict"]); net.eval()
torch.manual_seed(0)


@torch.no_grad()
def csample(xrows, k=8, steps=10):
    xn = torch.tensor((xrows - d["xmu"]) / d["xsd"], dtype=torch.float32)
    n = len(xn); xr = xn.repeat_interleave(k, 0)
    c = torch.randn(n * k, 5)
    for s in range(steps):
        t = torch.full((n * k, 1), s / steps)
        c = c + net(c, t, xr) / steps
    return (c.reshape(n, k, 5).mean(1) * torch.tensor(d["ysd"]) + torch.tensor(d["ymu"])).numpy()


def disp(c):
    return (c @ U.T).reshape(-1, H, gc.AD)[:, :, :3].sum(1) * astd[:3]


by = {}
for r in rows:
    if r["held"]:
        by.setdefault((r["si"], r["f"], r["lang"]), r)
fwd_langs = set(gc.ALL_PROMPTS if hasattr(gc, "ALL_PROMPTS") else
                [gc.PROMPT_L, gc.PROMPT_R, gc.PROMPT_CFL, gc.PROMPT_CFR])
pairs_rev, pairs_hold = [], []
for (si, f, lang), r in by.items():
    if lang not in fwd_langs:
        continue
    rb = by.get((si, f, ta.PROMPT_BACK))
    if rb is not None:
        pairs_rev.append((r, rb))
    rh = by.get((si, f, ta.PROMPT_HOLD))
    if rh is not None:
        pairs_hold.append((r, rh))
print(f"held identical-frame pairs: fwd/back {len(pairs_rev)}  fwd/hold {len(pairs_hold)}", flush=True)

for name, pairs in (("fwd vs back", pairs_rev),):
    A = csample(X[[p[0]["ri"] for p in pairs]]); B = csample(X[[p[1]["ri"] for p in pairs]])
    Ya = np.stack([p[0]["y"] for p in pairs]); Yb = np.stack([p[1]["y"] for p in pairs])
    dS, dY = disp(B) - disp(A), disp(Yb) - disp(Ya)
    cs = (dS * dY).sum(1) / (np.linalg.norm(dS, axis=1) * np.linalg.norm(dY, axis=1) + 1e-9)
    mag = np.linalg.norm(dS, axis=1) / (np.linalg.norm(dY, axis=1) + 1e-9)
    print("%s: cos(Δcmd, Δtrue) %.3f±%.3f   |Δcmd|/|Δtrue| %.2f±%.2f   n=%d" % (
        name, cs.mean(), cs.std(), mag.mean(), mag.std(), len(pairs)), flush=True)
if pairs_hold:
    A = csample(X[[p[0]["ri"] for p in pairs_hold]]); Hc = csample(X[[p[1]["ri"] for p in pairs_hold]])
    print("fwd vs hold: |cmd| fwd %.2f m -> hold %.2f m (true hold ~0)   n=%d" % (
        np.linalg.norm(disp(A), axis=1).mean(), np.linalg.norm(disp(Hc), axis=1).mean(),
        len(pairs_hold)), flush=True)
print("LANGTEST_DONE", flush=True)
