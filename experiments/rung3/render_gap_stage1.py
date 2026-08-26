"""Render-gap probe, stage 1 (RUNS IN THE /tmp/tv GSPLAT ENV, GPU).

For sampled demo frames, re-render the forward + downward-wrist observations at
the demo pose through the EXACT serving chain (rend 1024x768 -> 256 bilinear ->
[wrist strut overlay] -> 224 bicubic; render_yaw = -state[3] per the rollout
client convention). Saves ~/ctxrun/rendergap_frames.npz with rendered 224 pairs
+ (ep_file, t) indices for stage 2 (openpi env) to compare features/c against
the stored frames. Constants copied from gate_video_overlay.py (verified chain).
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

DEV = "cuda"
RD = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(RD, "data_gate_synth")
OUT = os.path.expanduser("~/ctxrun/rendergap_frames.npz")
N_EP_PER_SIDE, N_T = 4, 8
D = np.diag([1., -1, -1, 1])

SCENES = {
    "left": dict(
        ck="/home/ubuntu/code/falsify-pi/data/gate_scenes_export/left_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt",
        tw2g=np.array([[0.12614431661544656, 2.138646801849853e-06, -0.00025306576654559085, -0.15671883492487332],
                       [-2.138646801849853e-06, -0.1261265572041315, -0.0021319289354524646, -0.08013551648879384],
                       [-0.00025306576654559085, 0.0021319289354524646, -0.12612630156484925, -0.18772133850562778],
                       [0, 0, 0, 1.]]),
        eps=range(100, 150)),
    "right": dict(
        ck="/home/ubuntu/code/falsify-pi/data/gate_scenes_export/right_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_144353/nerfstudio_models/step-000029999.ckpt",
        tw2g=np.array([[0.136708, -0.001053, 0.006031, -0.111938], [0.00108, 0.13684, -0.000588, 0.030456],
                       [-0.006027, 0.000635, 0.136711, -0.201447], [0, 0, 0, 1.]]) @ D,
        eps=range(150, 200)),
}
Kf = torch.tensor([[502.2632, 0., 506.3971], [0., 500.6736, 385.41], [0., 0., 1.]], device=DEV)[None].float()
Kd = torch.tensor([[478.2450, 0., 511.9041], [0., 476.7944, 383.5003], [0., 0., 1.]], device=DEV)[None].float()
Tbc_f = np.array([[0, 0, -1, 0.10], [1, 0, 0, -0.03], [0, -1, 0, -0.01], [0, 0, 0, 1.]])
Tbc_d = np.array([[0, 1, 0, 0.0], [1, 0, 0, 0.0], [0, 0, -1, 0.05], [0, 0, 0, 1.]])
_ov = np.asarray(Image.open("/home/ubuntu/code/falsify-pi/configs/embodiments/assets/carl_wrist_overlay_pinhole_rgb.png")
                 .convert("RGBA").resize((256, 256), Image.BILINEAR), np.uint8)


def Rz(p):
    c, s = np.cos(p), np.sin(p)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def make_renderer(ck, tw2g):
    sd = torch.load(ck, map_location=DEV, weights_only=False)["pipeline"]
    def gg(n):
        for p in ("_model.gauss_params.", "_model."):
            if p + n in sd:
                return sd[p + n].to(DEV)
    means, quats = gg("means"), gg("quats")
    scales, opac = torch.exp(gg("scales")), torch.sigmoid(gg("opacities")).squeeze(-1)
    colors = torch.cat([gg("features_dc")[:, None, :], gg("features_rest")], 1)
    bg = torch.tensor([0.149, 0.1647, 0.2157], device=DEV)

    def vm(pos, yaw, Tbc):
        pn = np.array([pos[0], -pos[1], -pos[2]]); T = np.eye(4); T[:3, :3] = Rz(yaw); T[:3, 3] = pn
        c2w = tw2g @ (T @ Tbc); R = c2w[:3, :3] * np.array([1, -1, -1]); Ri = R.T
        V = np.eye(4); V[:3, :3] = Ri; V[:3, 3] = -Ri @ c2w[:3, 3]
        return V

    @torch.no_grad()
    def rend(pos, yaw, Tbc, Kk):
        V = torch.tensor(vm(pos, yaw, Tbc), device=DEV, dtype=torch.float32)[None]
        r, a, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
                                viewmats=V, Ks=Kk, width=1024, height=768, packed=False, near_plane=0.001,
                                far_plane=1e10, render_mode="RGB", sh_degree=3, rasterize_mode="classic")
        return ((r[..., :3] + (1 - a) * bg).clamp(0, 1).squeeze(0) * 255).byte().cpu().numpy()
    return rend


def to256(a):
    return np.asarray(Image.fromarray(a).resize((256, 256), Image.BILINEAR), np.uint8)


def ov(x):
    rgb = _ov[..., :3].astype(np.float32); al = _ov[..., 3:4].astype(np.float32) / 255.
    return (al * rgb + (1 - al) * x.astype(np.float32)).clip(0, 255).astype(np.uint8)


def r224(a):
    return np.asarray(Image.fromarray(a).resize((224, 224), Image.BICUBIC), np.uint8)


recs = {"ep_file": [], "t": [], "side": [], "fwd224": [], "wrist224": []}
for side, sc in SCENES.items():
    rend = make_renderer(sc["ck"], sc["tw2g"])
    files = [f"{DD}/ep_{i:04d}.npz" for i in list(sc["eps"])[::len(list(sc["eps"])) // N_EP_PER_SIDE][:N_EP_PER_SIDE]]
    for f in files:
        st = np.load(f, allow_pickle=True)["state"].astype(np.float32)
        for t in np.linspace(0, len(st) - 2, N_T).astype(int):
            pos, yaw = st[t, :3], -float(st[t, 3])   # render_yaw = -state[3]
            recs["fwd224"].append(r224(to256(rend(pos, yaw, Tbc_f, Kf))))
            recs["wrist224"].append(r224(ov(to256(rend(pos, yaw, Tbc_d, Kd)))))
            recs["ep_file"].append(os.path.basename(f)); recs["t"].append(int(t)); recs["side"].append(side)
        print("rendered", os.path.basename(f), flush=True)
np.savez(OUT, fwd224=np.stack(recs["fwd224"]), wrist224=np.stack(recs["wrist224"]),
         ep_file=np.array(recs["ep_file"]), t=np.array(recs["t"]), side=np.array(recs["side"]))
print("STAGE1_DONE", OUT, len(recs["t"]), flush=True)
