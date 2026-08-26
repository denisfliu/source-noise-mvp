"""Stage-1 flow-matching head on VLA features: p(c | phi), NO one-hot, no state.

The two-stage-flow architecture (Denis, 2026-08-06): the VLA as it is — pi0's
fused (post-fusion) prefix feature phi carries vision+language; a small
rectified-flow head samples the coarse command c from p(c|phi); c pins the
source noise; the action expert denoises the rest. This trains the head on the
existing rendered-domain cache (Xrendshard_*, same rows/labels/split as
vlmc_ridge_rend / vlmc_mlp_rend), so held R^2 is directly comparable to the
deterministic maps that were offline-excellent but closed-loop-dead. CPU.
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
torch.manual_seed(0)
U = np.load(os.path.join(RD, "pin_U_gate_rrr_k5.npy"))
ns, amean, astd = gc.load_norm()
H = gc.H

# rebuild the cache's row alignment (identical to gate_locator.py / extract_aug_features)
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
Y, TR = [], []
for si, e in groups:
    n = min(len(e["action"]), len(e["state"]) - 1)
    ac = e["action"].astype(np.float32)
    for t in range(0, n, STRIDE):
        Y.append((gc.segY(ac[t:], amean, astd) @ U).astype(np.float32))
        TR.append(si in trep)
X = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{RUN}/Xrendshard_*.npy"))], 0)
Y = np.stack(Y); TR = np.array(TR)
assert len(X) == len(Y), (len(X), len(Y))
print(f"rows {len(X)} (train {TR.sum()})", flush=True)

xmu, xsd = X[TR].mean(0), X[TR].std(0) + 1e-6
ymu, ysd = Y[TR].mean(0), Y[TR].std(0) + 1e-6
Xn = torch.tensor((X - xmu) / xsd, dtype=torch.float32)
Yn = torch.tensor((Y - ymu) / ysd)


class VNet(nn.Module):
    def __init__(self, xdim=2048, cdim=5, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(xdim + cdim + 1, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))

    def forward(self, ct, t, x):
        return self.net(torch.cat([ct, t, x], 1))


net = VNet(X.shape[1])
opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)
tri = np.where(TR)[0]
for ep_i in range(80):
    perm_i = np.random.permutation(tri)
    tot = 0.0
    for j in range(0, len(perm_i), 512):
        b = perm_i[j:j + 512]
        c1 = Yn[b]; c0 = torch.randn_like(c1)
        t = torch.rand(len(b), 1)
        ct = (1 - t) * c0 + t * c1
        loss = ((net(ct, t, Xn[b]) - (c1 - c0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
    if ep_i % 20 == 0 or ep_i == 79:
        print(f"epoch {ep_i} cfm-mse {tot/len(perm_i):.4f}", flush=True)
net.eval()


@torch.no_grad()
def sample(x, k=1, steps=10):
    n = len(x)
    xr = x.repeat_interleave(k, 0)
    c = torch.randn(n * k, 5)
    for s in range(steps):
        t = torch.full((n * k, 1), s / steps)
        c = c + net(c, t, xr) / steps
    return (c.reshape(n, k, 5) * torch.tensor(ysd) + torch.tensor(ymu)).numpy()


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


he = ~TR
S1 = sample(Xn[he], k=1)[:, 0]
S8 = sample(Xn[he], k=8).mean(1)
print("held R^2: vlm-flow 1-sample %.3f  8-mean %.3f" % (r2(Y[he], S1), r2(Y[he], S8)), flush=True)
for name in ("vlmc_ridge_rend.npz", "vlmc_mlp_rend.npz"):
    m = gc.load_ridge(os.path.join(RD, name))
    P = gc.apply_ridge(m, X[he], clamp=True)
    print("held R^2: %-20s %.3f (same held rows)" % (name[5:-4], r2(Y[he], P)), flush=True)
torch.save({"state_dict": net.state_dict(), "xmu": xmu.astype(np.float32),
            "xsd": xsd.astype(np.float32), "ymu": ymu.astype(np.float32),
            "ysd": ysd.astype(np.float32), "in_dim": X.shape[1], "H": H, "AD": gc.AD,
            "K": 5, "arch": "cfm-3x512-vlmfeat"}, os.path.join(RD, "vlmflow_head_rend.pt"))
print("saved vlmflow_head_rend.pt"); print("VLMFLOW_DONE", flush=True)
