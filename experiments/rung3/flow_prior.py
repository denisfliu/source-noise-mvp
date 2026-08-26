"""Flow-matching prior over the pin coordinate: p(c | state, task-onehot).

Same inputs, rows, and frozen split as the deployed no-clock MLP prior
(noprog_prior_rrr4) so numbers are directly comparable — only the head changes:
rectified-flow v_theta(c_t, t, x), Euler sampling at inference. Motivation
(Denis, 2026-08-06): MSE regression averages modes (endgame parking, late/wide
turns, confidently-wrong off-manifold commands); a generative head can hold
multimodal command distributions and commit to one sample.

Offline report: held R^2 of single samples and of K-sample means vs the MLP
prior on identical held rows; per-task; and sample SPREAD at on-manifold vs
off-manifold (compound-seam) states — the seam is where the MLP extrapolates
confidently wrong (cos -0.86); a calibrated generative prior should show high
variance there instead. CPU-only.
"""
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
torch.manual_seed(0)
rng = np.random.default_rng(0)

policy = gc.make_policy()
_D = np.zeros((224, 224, 3), np.uint8)


def mstate(raw, lang):
    return np.asarray(policy._input_transform(
        {"observation/image": _D, "observation/wrist_image": _D,
         "observation/state": raw, "prompt": lang})["state"]).reshape(-1)


eps = gc.load_eps(with_images=False)
perm = rng.permutation(len(eps))
tr_set = set(perm[:160].tolist())
X, Y, TR, TASK = [], [], [], []
for i, e in enumerate(eps):
    if e["lang"] not in TASKS4:
        continue
    oh = np.zeros(4, np.float32); oh[TASKS4.index(e["lang"])] = 1.0
    st = e["state"].astype(np.float32); ac = e["action"].astype(np.float32)
    for t in range(0, len(st) - H, 8):
        X.append(np.concatenate([mstate(st[t], e["lang"]), oh]).astype(np.float32))
        Y.append((gc.segY(ac[t:], amean, astd) @ U).astype(np.float32))
        TR.append(i in tr_set); TASK.append(e["lang"])
X = np.stack(X); Y = np.stack(Y); TR = np.array(TR); TASK = np.array(TASK)
print(f"rows {len(X)} (train {TR.sum()})", flush=True)

xmu, xsd = X[TR].mean(0), X[TR].std(0) + 1e-6
ymu, ysd = Y[TR].mean(0), Y[TR].std(0) + 1e-6
Xn = torch.tensor((X - xmu) / xsd); Yn = torch.tensor((Y - ymu) / ysd)


class VNet(nn.Module):
    def __init__(self, xdim, cdim=5, w=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(xdim + cdim + 1, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))

    def forward(self, ct, t, x):
        return self.net(torch.cat([ct, t, x], 1))


net = VNet(X.shape[1])
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
tri = np.where(TR)[0]
for ep_i in range(120):
    perm_i = np.random.permutation(tri)
    tot = 0.0
    for j in range(0, len(perm_i), 512):
        b = perm_i[j:j + 512]
        c1 = Yn[b]; c0 = torch.randn_like(c1)
        t = torch.rand(len(b), 1)
        ct = (1 - t) * c0 + t * c1
        v = net(ct, t, Xn[b])
        loss = ((v - (c1 - c0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
    if ep_i % 20 == 0 or ep_i == 119:
        print(f"epoch {ep_i} cfm-mse {tot/len(perm_i):.4f}", flush=True)
net.eval()


@torch.no_grad()
def sample(x, k=1, steps=10):
    """k samples of c for each row of x (normalized in, physical out)."""
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
# deployed MLP prior baseline on the same held rows
d = torch.load(os.path.join(RD, "noprog_prior_rrr4.pt"), map_location="cpu", weights_only=False)
layers, din = [], d["in_dim"]
for h_ in d["hidden"]:
    layers += [nn.Linear(din, h_), nn.SiLU()]; din = h_
layers += [nn.Linear(din, 5)]
mlp = nn.Sequential(*layers); mlp.load_state_dict(d["state_dict"]); mlp.eval()
with torch.no_grad():
    Pm = mlp(torch.tensor((X[he] - d["mu"]) / d["sd"], dtype=torch.float32)).numpy()
print("\nheld R^2 (all tasks):  flow-1sample %.3f   flow-8mean %.3f   MLP prior %.3f" % (
    r2(Y[he], S1), r2(Y[he], S8), r2(Y[he], Pm)), flush=True)
for task in TASKS4:
    m = TASK[he] == task
    print("  %-24s flow-8mean %.3f  MLP %.3f" % (task.split(" gate")[0][-14:] + "/" + task[-30:-26],
          r2(Y[he][m], S8[m]), r2(Y[he][m], Pm[m])), flush=True)

# spread probe: on-manifold demo state vs off-manifold seam state (CFL onehot)
def spread_at(pos, yaw):
    raw = np.array([*pos, yaw, 0, 0, 0], np.float32)
    oh = np.zeros(4, np.float32); oh[0] = 1.0
    x = np.concatenate([mstate(raw, gc.PROMPT_CFL), oh]).astype(np.float32)
    xn = torch.tensor(((x - xmu) / xsd)[None], dtype=torch.float32)
    S = sample(xn, k=32)[0]
    disp = (S @ U.T).reshape(32, H, gc.AD)[:, :, :3].sum(1) * astd[:3]
    return disp.mean(0), disp.std(0)


for name, pos in (("on-manifold (CFL demo start)", [0.0, 0.9, 1.5]),
                  ("off-manifold (compound seam)", [1.522, -0.614, 0.997])):
    mu_d, sd_d = spread_at(pos, 0.0)
    print("spread %-30s cmd-mean %s  cmd-std %s" % (name, np.round(mu_d, 2), np.round(sd_d, 2)), flush=True)
torch.save({"state_dict": net.state_dict(), "xmu": xmu, "xsd": xsd, "ymu": ymu, "ysd": ysd,
            "tasks": TASKS4, "in_dim": X.shape[1], "H": H, "AD": gc.AD, "K": 5,
            "arch": "cfm-3x256"}, os.path.join(RD, "flow_prior_rrr4.pt"))
print("saved flow_prior_rrr4.pt"); print("FLOW_PRIOR_DONE", flush=True)
