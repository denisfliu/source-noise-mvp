"""POSE-BOTTLENECK decomposition (Denis, 2026-08-06: 'only the MLP points the right
way — we need deep understanding of why').

Hypothesis: c is ~99% state-driven (probe_pin_state finding), so the MLP prior fits
an easy smooth function of POSITION; VLM heads must first recover pose from pixels,
and their command error is dominated by that pose error.

Test: (A) pose probe phi -> (x,y,z) on union rows, held error = the VLM's effective
position-sensor precision. (B) CASCADE = mlp_prior(pose_hat(phi), onehot): swap only
the position source. (C) compare command fields on the basin frames:
mlp(true state) vs cascade vs direct VLM heads.
  cascade ~= mlp(true)  -> pose is fine; VLM heads' c-MAPPING is the fault
  cascade ~= direct-vlm -> POSE PRECISION is the bottleneck (the deep reason)
GPU (feature recompute for probe frames), openpi env.
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
from fat_tube_gen import sample_rows

RUN = os.path.expanduser("~/ctxrun")
RD = gc.RD
U = np.load(os.path.join(RD, "pin_U_gate_rrr_k5.npy"))
ns, amean, astd = gc.load_norm()
H = gc.H
TASKS4 = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]
torch.manual_seed(0)

# ---- union rows: pooled phi (cached) + position labels + task + held mask
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
POS, TASK, HE = [], [], []
for si, e in groups:
    n = min(len(e["action"]), len(e["state"]) - 1)
    for t in range(0, n, 12):
        POS.append(e["state"][t, :4].astype(np.float32))
        TASK.append(e["lang"]); HE.append(si not in trep)
X1 = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{RUN}/Xrendshard_*.npy"))], 0)
src2, rows2 = sample_rows()
tf = np.load(f"{RUN}/fat_tube_frames.npz"); ST2 = tf["st"]
X2 = np.load(f"{RUN}/fat_tube_phi.npy")
for i, (task, ei, t, dv) in enumerate(rows2):
    POS.append(ST2[i, :4].astype(np.float32)); TASK.append(task); HE.append(False)
X = np.concatenate([X1, X2], 0).astype(np.float32)
POS = np.stack(POS); TASK = np.array(TASK); HE = np.array(HE)
assert len(X) == len(POS)

# ---- (A) pose probe: phi -> (x,y,z,yaw)
tr = ~HE
xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-6
pn = nn.Sequential(nn.Linear(2048, 256), nn.GELU(approximate="tanh"), nn.Linear(256, 4))
opt = torch.optim.AdamW(pn.parameters(), lr=1e-3, weight_decay=1e-4)
Xn = torch.tensor((X - xmu) / xsd); Pt = torch.tensor(POS)
tri = np.where(tr)[0]
for ep in range(40):
    perm = np.random.permutation(tri)
    for j in range(0, len(perm), 512):
        b = perm[j:j + 512]
        loss = ((pn(Xn[b]) - Pt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
pn.eval()
with torch.no_grad():
    Php = pn(Xn).numpy()
err = np.linalg.norm(Php[HE, :3] - POS[HE, :3], axis=1)
print("POSE PROBE held |xyz err|: mean %.3f m  p50 %.3f  p90 %.3f   yaw err %.2f rad" % (
    err.mean(), np.median(err), np.percentile(err, 90),
    np.abs(Php[HE, 3] - POS[HE, 3]).mean()), flush=True)

# ---- load heads
policy = gc.make_policy()
_D = np.zeros((224, 224, 3), np.uint8)


def mstate(raw, lang):
    return np.asarray(policy._input_transform(
        {"observation/image": _D, "observation/wrist_image": _D,
         "observation/state": raw, "prompt": lang})["state"]).reshape(-1)


d = torch.load(os.path.join(RD, "noprog_prior_rrr4.pt"), map_location="cpu", weights_only=False)
lay, di = [], d["in_dim"]
for h_ in d["hidden"]:
    lay += [nn.Linear(di, h_), nn.SiLU()]; di = h_
lay += [nn.Linear(di, 5)]
mlp = nn.Sequential(*lay); mlp.load_state_dict(d["state_dict"]); mlp.eval()


def prior_c(pos4, lang):
    raw = np.array([*pos4, 0, 0, 0], np.float32)
    x = np.concatenate([mstate(raw, lang),
                        np.eye(4, dtype=np.float32)[TASKS4.index(lang)]])
    with torch.no_grad():
        return mlp(torch.tensor(((x - d["mu"]) / d["sd"])[None], dtype=torch.float32))[0].numpy()


class VNet(nn.Module):
    def __init__(self, xdim, cdim=5, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(xdim + cdim + 1, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))

    def forward(self, ct, t, x):
        return self.net(torch.cat([ct, t, x], 1))


vf = torch.load(os.path.join(RD, "vlmflow_head_union.pt"), map_location="cpu", weights_only=False)
vfn = VNet(vf["in_dim"]); vfn.load_state_dict(vf["state_dict"]); vfn.eval()


def cfm(x, k=8):
    torch.manual_seed(0)
    xn = torch.tensor((x - vf["xmu"]) / vf["xsd"], dtype=torch.float32)
    n = len(xn); xr = xn.repeat_interleave(k, 0)
    c = torch.randn(n * k, 5)
    with torch.no_grad():
        for s in range(10):
            t = torch.full((n * k, 1), s / 10)
            c = c + vfn(c, t, xr) / 10
    return (c.reshape(n, k, 5).mean(1) * torch.tensor(vf["ysd"]) + torch.tensor(vf["ymu"])).numpy()


def decode(C):
    return (np.atleast_2d(C) @ U.T).reshape(-1, H, gc.AD)[:, :, :3].sum(1) * astd[:3]


# ---- (C) basin frames: recompute phi, then three fields
gf = np.load(f"{RUN}/gain2_frames.npz"); GST = gf["st"]; meta = gf["meta"]
obs3 = [{"observation/image": gf["fwd"][i], "observation/wrist_image": gf["wr"][i],
         "observation/state": GST[i], "prompt": gc.PROMPT_L} for i in range(len(GST))]
phis = []
for i in range(0, len(obs3), gc.BS):
    phis.append(gc.ctx_pool(policy, obs3[i:i + gc.BS]))
phiG = np.concatenate(phis, 0)
with torch.no_grad():
    poseG = pn(torch.tensor((phiG - xmu) / xsd)).numpy()
pose_err_G = np.linalg.norm(poseG[:, :3] - GST[:, :3], axis=1)
print("pose err on basin frames: mean %.3f m (on-route pts %.3f, 1.0m-offset pts %.3f)" % (
    pose_err_G.mean(), pose_err_G[meta[:, 1] == 0].mean(),
    pose_err_G[meta[:, 1] >= 9].mean()), flush=True)
fields = {
    "mlp_true": np.stack([decode(prior_c(GST[i, :4], gc.PROMPT_L))[0] for i in range(len(GST))]),
    "cascade": np.stack([decode(prior_c(poseG[i], gc.PROMPT_L))[0] for i in range(len(GST))]),
    "vlm_union": decode(cfm(phiG)),
}
plist = []
for Dl in (0.25, 0.5, 1.0):
    for a in (1, 2):
        for sg in (+1, -1):
            plist.append((Dl, a, sg))
print("\nfield comparison on basin frames:")
print("%-10s %28s   %s" % ("head", "gain y/z @.25 @.5 @1.0", "|Δ vs mlp_true| (m)"))
for nm, C in fields.items():
    G = {}
    for b in range(5):
        i0 = np.where((meta[:, 0] == b) & (meta[:, 1] == 0))[0][0]
        for pi, (Dl, ax, sg) in enumerate(plist, start=1):
            ii = np.where((meta[:, 0] == b) & (meta[:, 1] == pi))[0][0]
            G.setdefault((Dl, ax), []).append(-(C[ii, ax] - C[i0, ax]) / (sg * Dl))
    dv = np.linalg.norm(C - fields["mlp_true"], axis=1).mean()
    print("%-10s  %.2f/%.2f  %.2f/%.2f  %.2f/%.2f      %.3f" % (nm,
        np.mean(G[(0.25, 1)]), np.mean(G[(0.25, 2)]), np.mean(G[(0.5, 1)]),
        np.mean(G[(0.5, 2)]), np.mean(G[(1.0, 1)]), np.mean(G[(1.0, 2)]), dv), flush=True)
print("POSE_BOTTLENECK_DONE", flush=True)
