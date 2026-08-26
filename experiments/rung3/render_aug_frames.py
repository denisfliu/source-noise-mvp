"""Render every unique source frame referenced by the augmented rec set through the
EXACT serving chain (RUNS IN THE /tmp/tv GSPLAT ENV, GPU) -> RUN/rendered_frames.npz.

Feeds extract_aug_features.py OBS=rendered (the domain-matching fix). The rec
construction here MUST mirror extract_aug_features.py (same generators, same
STRIDE default 12, same group order) — the extractor's (si,fidx) lookup fails
loudly on any drift. Render/composite chain copied from render_gap_stage1.py
(verified against gate_video_overlay.py): 1024x768 -> 256 bilinear -> [wrist
strut overlay] -> 224 bicubic; render_yaw = -state[3]. Scene = the episode's
side splat (lang), matching how rollouts pick SCENE.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import gate_traj_algebra as ta

RUN = os.path.expanduser("~/ctxrun")
OUT = os.environ.get("OUT", f"{RUN}/rendered_frames.npz")
STRIDE = int(os.environ.get("STRIDE", "12"))
DEV = "cuda"
D = np.diag([1., -1, -1, 1])
SCENES = {
    "left": dict(
        ck="/home/ubuntu/code/falsify-pi/data/gate_scenes_export/left_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt",
        tw2g=np.array([[0.12614431661544656, 2.138646801849853e-06, -0.00025306576654559085, -0.15671883492487332],
                       [-2.138646801849853e-06, -0.1261265572041315, -0.0021319289354524646, -0.08013551648879384],
                       [-0.00025306576654559085, 0.0021319289354524646, -0.12612630156484925, -0.18772133850562778],
                       [0, 0, 0, 1.]])),
    "right": dict(
        ck="/home/ubuntu/code/falsify-pi/data/gate_scenes_export/right_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_144353/nerfstudio_models/step-000029999.ckpt",
        tw2g=np.array([[0.136708, -0.001053, 0.006031, -0.111938], [0.00108, 0.13684, -0.000588, 0.030456],
                       [-0.006027, 0.000635, 0.136711, -0.201447], [0, 0, 0, 1.]]) @ D),
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


def make_renderer(sc):
    sd = torch.load(sc["ck"], map_location=DEV, weights_only=False)["pipeline"]
    def gg(n):
        for p in ("_model.gauss_params.", "_model."):
            if p + n in sd:
                return sd[p + n].to(DEV)
    means, quats = gg("means"), gg("quats")
    scales, opac = torch.exp(gg("scales")), torch.sigmoid(gg("opacities")).squeeze(-1)
    colors = torch.cat([gg("features_dc")[:, None, :], gg("features_rest")], 1)
    bg = torch.tensor([0.149, 0.1647, 0.2157], device=DEV)
    tw2g = sc["tw2g"]

    @torch.no_grad()
    def rend(pos, yaw, Tbc, Kk):
        pn = np.array([pos[0], -pos[1], -pos[2]]); T = np.eye(4); T[:3, :3] = Rz(yaw); T[:3, 3] = pn
        c2w = tw2g @ (T @ Tbc); R = c2w[:3, :3] * np.array([1, -1, -1]); Ri = R.T
        V = np.eye(4); V[:3, :3] = Ri; V[:3, 3] = -Ri @ c2w[:3, 3]
        Vt = torch.tensor(V, device=DEV, dtype=torch.float32)[None]
        r, a, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
                                viewmats=Vt, Ks=Kk, width=1024, height=768, packed=False, near_plane=0.001,
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


src = gc.load_eps(with_images=False)
groups = []
for si, e in enumerate(src):
    groups.append((si, e))
    groups.append((si, ta.reverse(e)))
    for f in (ta.crop_to_gate, ta.crop_from_gate):
        a = f(e)
        if a is not None:
            groups.append((si, a))
    groups.append((si, ta.hover(e, len(e["action"]) // 2)))
need = set()
for si, e in groups:
    n = min(len(e["action"]), len(e["state"]) - 1)
    for t in range(0, n, STRIDE):
        need.add((si, int(e["fidx"][t]) if "fidx" in e else t))
print("unique frames to render:", len(need), flush=True)

SI, FI, FWD, WRI = [], [], [], []
for side in ("left", "right"):
    rend = make_renderer(SCENES[side])
    # Scene assignment: ONLY right-gate episodes live in the right splat; center
    # episodes (CFL/CFR) are pixel-verified to be left-scene renders (2026-08-05
    # review — the binary-label scene test was the is_left bug reborn here).
    todo = [(si, fi) for (si, fi) in sorted(need)
            if (src[si]["lang"] == gc.PROMPT_R) == (side == "right")]
    for k, (si, fi) in enumerate(todo):
        st = src[si]["state"][fi]
        pos, yaw = st[:3], -float(st[3])
        FWD.append(r224(to256(rend(pos, yaw, Tbc_f, Kf))))
        WRI.append(r224(ov(to256(rend(pos, yaw, Tbc_d, Kd)))))
        SI.append(si); FI.append(fi)
        if k % 500 == 0:
            print("  %s %d/%d" % (side, k, len(todo)), flush=True)
    del rend
    torch.cuda.empty_cache()
np.savez(OUT, si=np.array(SI, np.int32), fidx=np.array(FI, np.int32),
         fwd224=np.stack(FWD), wrist224=np.stack(WRI))
print("RENDER_AUG_FRAMES_DONE", len(SI), OUT, flush=True)
