"""Command-field comparison along fixed reference trajectories (Denis, 2026-08-06):
same states/frames, five heads — where do VLM-feature commands go wrong?

Stage 1 (this file, MODE=render, tv env, GPU): render fwd+wrist frames at chunk-start
states of (a) a KNOWN-GOOD left route (hard-aug prior 5/5: traj_ha_left_t1) and (b) a
FAILED vlmflow left rollout (traj_vf_left_t1). Saves cmp_frames.npz.

Stage 2 (MODE=eval, openpi env, GPU): at each point compute commands from
  state-based:  MLP prior (noprog_prior_rrr4), flow prior (8-mean)
  feature-based: vlmc_ridge_rend, vlmc_mlp_rend, vlmflow head (8-mean)
decode to net-displacement meters; reference = actual displacement of the good
trajectory over the next H steps. Report per-head error and mean bias (x,y,z), on
the good route and along the failed route's early states.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODE = os.environ.get("MODE", "eval")
RUN = os.path.expanduser("~/ctxrun")
APC, NPT = 8, 30

REFS = [("good_ha", f"{RUN}/traj_ha_left_t1.npy"), ("fail_vf", f"{RUN}/traj_vf_left_t1.npy")]

if MODE == "render":
    import torch
    from PIL import Image
    from gsplat import rasterization
    CK = ("/home/ubuntu/code/falsify-pi/data/gate_scenes_export/left_scene/mocap_outputs/"
          "sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt")
    Tw2g = np.array([[0.12614431661544656, 2.138646801849853e-06, -0.00025306576654559085, -0.15671883492487332],
                     [-2.138646801849853e-06, -0.1261265572041315, -0.0021319289354524646, -0.08013551648879384],
                     [-0.00025306576654559085, 0.0021319289354524646, -0.12612630156484925, -0.18772133850562778],
                     [0, 0, 0, 1.0]])
    sd = torch.load(CK, map_location="cuda", weights_only=False)["pipeline"]
    def gg(n):
        for p in ("_model.gauss_params.", "_model."):
            if p + n in sd:
                return sd[p + n].to("cuda")
    means, quats = gg("means"), gg("quats")
    scales, opac = torch.exp(gg("scales")), torch.sigmoid(gg("opacities")).squeeze(-1)
    colors = torch.cat([gg("features_dc")[:, None, :], gg("features_rest")], 1)
    bg = torch.tensor([0.149, 0.1647, 0.2157], device="cuda")
    Kf = torch.tensor([[502.2632, 0., 506.3971], [0., 500.6736, 385.41], [0., 0., 1.]], device="cuda")[None].float()
    Kd = torch.tensor([[478.2450, 0., 511.9041], [0., 476.7944, 383.5003], [0., 0., 1.]], device="cuda")[None].float()
    Tbc_f = np.array([[0, 0, -1, 0.10], [1, 0, 0, -0.03], [0, -1, 0, -0.01], [0, 0, 0, 1.]])
    Tbc_d = np.array([[0, 1, 0, 0.0], [1, 0, 0, 0.0], [0, 0, -1, 0.05], [0, 0, 0, 1.]])
    _ov = np.asarray(Image.open("/home/ubuntu/code/falsify-pi/configs/embodiments/assets/"
                                "carl_wrist_overlay_pinhole_rgb.png").convert("RGBA").resize((256, 256), Image.BILINEAR), np.uint8)
    def Rz(p):
        c, s = np.cos(p), np.sin(p); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    def rend(pos, yaw, Tbc, Kk):
        pn = np.array([pos[0], -pos[1], -pos[2]]); T = np.eye(4); T[:3, :3] = Rz(yaw); T[:3, 3] = pn
        c2w = Tw2g @ (T @ Tbc); R = c2w[:3, :3] * np.array([1, -1, -1]); Ri = R.T
        V = np.eye(4); V[:3, :3] = Ri; V[:3, 3] = -Ri @ c2w[:3, 3]
        Vt = torch.tensor(V, device="cuda", dtype=torch.float32)[None]
        with torch.no_grad():
            r, a, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
                                    viewmats=Vt, Ks=Kk, width=1024, height=768, packed=False, near_plane=0.001,
                                    far_plane=1e10, render_mode="RGB", sh_degree=3, rasterize_mode="classic")
        return ((r[..., :3] + (1 - a) * bg).clamp(0, 1).squeeze(0) * 255).byte().cpu().numpy()
    def to256(a): return np.asarray(Image.fromarray(a).resize((256, 256), Image.BILINEAR), np.uint8)
    def ov(x):
        rgb = _ov[..., :3].astype(np.float32); al = _ov[..., 3:4].astype(np.float32) / 255.
        return (al * rgb + (1 - al) * x.astype(np.float32)).clip(0, 255).astype(np.uint8)
    def r224(a): return np.asarray(Image.fromarray(a).resize((224, 224), Image.BICUBIC), np.uint8)
    out = {}
    for tag, path in REFS:
        P = np.load(path)
        steps = list(range(0, min(len(P) - 1, NPT * APC), APC))
        F, W, S = [], [], []
        for t in steps:
            pos, yaw = P[t, :3], (P[t, 3] if P.shape[1] > 3 else 0.0)
            F.append(r224(to256(rend(pos, -yaw, Tbc_f, Kf))))
            W.append(r224(ov(to256(rend(pos, -yaw, Tbc_d, Kd)))))
            S.append(np.array([*pos, yaw, 0, 0, 0], np.float32))
        out[f"{tag}_fwd"] = np.stack(F); out[f"{tag}_wr"] = np.stack(W)
        out[f"{tag}_st"] = np.stack(S); out[f"{tag}_steps"] = np.array(steps)
    np.savez(f"{RUN}/cmp_frames.npz", **out)
    print("CMP_RENDER_DONE", flush=True)
    sys.exit(0)

import torch
import torch.nn as nn
import gate_ctx_common as gc

RD = gc.RD
U = np.load(os.path.join(RD, "pin_U_gate_rrr_k5.npy"))
ns, amean, astd = gc.load_norm()
H = gc.H
TASKS4 = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]
policy = gc.make_policy()
rf = np.load(f"{RUN}/cmp_frames.npz")


def decode(C):
    return (np.atleast_2d(C) @ U.T).reshape(-1, H, gc.AD)[:, :, :4].sum(1)[:, :3] * astd[:3]


d = torch.load(os.path.join(RD, "noprog_prior_rrr4.pt"), map_location="cpu", weights_only=False)
layers, din = [], d["in_dim"]
for h_ in d["hidden"]:
    layers += [nn.Linear(din, h_), nn.SiLU()]; din = h_
layers += [nn.Linear(din, 5)]
mlp = nn.Sequential(*layers); mlp.load_state_dict(d["state_dict"]); mlp.eval()


class VNet(nn.Module):
    def __init__(self, xdim, cdim=5, w=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(xdim + cdim + 1, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))

    def forward(self, ct, t, x):
        return self.net(torch.cat([ct, t, x], 1))


fp = torch.load(os.path.join(RD, "flow_prior_rrr4.pt"), map_location="cpu", weights_only=False)
fpn = VNet(fp["in_dim"]); fpn.load_state_dict(fp["state_dict"]); fpn.eval()
vf = torch.load(os.path.join(RD, "vlmflow_head_rend.pt"), map_location="cpu", weights_only=False)
vfn = VNet(vf["in_dim"], w=512); vfn.load_state_dict(vf["state_dict"]); vfn.eval()
torch.manual_seed(0)


def cfm_sample(net, dd, x, k=8, steps=10):
    xn = torch.tensor((x - dd["xmu"]) / dd["xsd"], dtype=torch.float32)
    n = len(xn); xr = xn.repeat_interleave(k, 0)
    c = torch.randn(n * k, 5)
    with torch.no_grad():
        for s in range(steps):
            t = torch.full((n * k, 1), s / steps)
            c = c + net(c, t, xr) / steps
    return (c.reshape(n, k, 5).mean(1) * torch.tensor(dd["ysd"]) + torch.tensor(dd["ymu"])).numpy()


maps = {n: gc.load_ridge(os.path.join(RD, f"vlmc_{n}_rend.npz")) for n in ("ridge", "mlp")}
oh = np.zeros(4, np.float32); oh[TASKS4.index(gc.PROMPT_L)] = 1.0
for tag, path in REFS:
    st = rf[f"{tag}_st"]; steps = rf[f"{tag}_steps"]
    obs = [{"observation/image": rf[f"{tag}_fwd"][i], "observation/wrist_image": rf[f"{tag}_wr"][i],
            "observation/state": st[i], "prompt": gc.PROMPT_L} for i in range(len(st))]
    phis = []
    for i in range(0, len(obs), gc.BS):
        phis.append(gc.ctx_pool(policy, obs[i:i + gc.BS]))
    phi = np.concatenate(phis, 0)
    ms = np.stack([np.asarray(policy._input_transform(dict(o))["state"]).reshape(-1) for o in obs])
    xs = np.concatenate([ms, np.tile(oh, (len(ms), 1))], 1).astype(np.float32)
    P = np.load(path)
    cmds = {
        "mlp_prior": decode(mlp(torch.tensor((xs - d["mu"]) / d["sd"], dtype=torch.float32)).detach().numpy()),
        "flow_prior": decode(cfm_sample(fpn, fp, xs)),
        "vlmc_ridge": decode(gc.apply_ridge(maps["ridge"], phi, clamp=True)),
        "vlmc_mlp": decode(gc.apply_ridge(maps["mlp"], phi, clamp=True)),
        "vlmflow": decode(cfm_sample(vfn, vf, phi)),
    }
    # reference displacement: where the reference traj actually went over the next H steps
    ref = np.stack([P[min(t + H, len(P) - 1), :3] - P[t, :3] for t in steps])
    print(f"\n=== {tag} ({os.path.basename(path)}), {len(steps)} chunk-starts, prompt=LEFT")
    print("%-12s %18s %26s" % ("head", "mean |err| (m)", "mean bias x/y/z (m)"))
    for name, C in cmds.items():
        err = np.linalg.norm(C - ref, axis=1)
        bias = (C - ref).mean(0)
        print("%-12s %12.3f       %8.2f %6.2f %6.2f" % (name, err.mean(), *bias))
    # early-route detail: first 6 points, z-component commands vs ref
    print("first-6 z-cmd:", {n: np.round(c[:6, 2], 2).tolist() for n, c in
                             list(cmds.items())[0:1] + list(cmds.items())[4:5]},
          " ref-z", np.round(ref[:6, 2], 2).tolist(), flush=True)
print("CMP_EVAL_DONE", flush=True)
