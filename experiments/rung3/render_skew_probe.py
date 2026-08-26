"""Is the observation the SERVER sees the same one the training cache was built from?

The grounded prior's language-embedding input is far outside its training distribution at the very
first inference, with the drone at the origin where all 100 demos also start — so the skew cannot be
explained by novel viewpoints. The embedding depends on the prompt (verified identical) and the
images, so this re-renders each demo's own frame through the LIVE serving render path
(gate_video_overlay: rasterize 1024x768 -> 256 BILINEAR -> 224 BICUBIC) and compares it against the
stored demo frame the cache used (256 stored -> 224 BICUBIC).

  python render_skew_probe.py --eps 100 101 102 --frames 0 20
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

RD = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
CK = ("/home/ubuntu/code/falsify-pi/data/gate_scenes_export/left_scene/mocap_outputs/"
      "sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt")
Tw2g = np.array([
    [0.12614431661544656, 2.138646801849853e-06, -0.00025306576654559085, -0.15671883492487332],
    [-2.138646801849853e-06, -0.1261265572041315, -0.0021319289354524646, -0.08013551648879384],
    [-0.00025306576654559085, 0.0021319289354524646, -0.12612630156484925, -0.18772133850562778],
    [0, 0, 0, 1.]])
Kf = torch.tensor([[502.2632, 0., 506.3971], [0., 500.6736, 385.41], [0., 0., 1.]], device=DEV)[None].float()
Tbc_f = np.array([[0, 0, -1, 0.10], [1, 0, 0, -0.03], [0, -1, 0, -0.01], [0, 0, 0, 1.]])

_sd = torch.load(CK, map_location=DEV, weights_only=False)["pipeline"]


def _gg(n):
    for p in ("_model.gauss_params.", "_model."):
        if p + n in _sd:
            return _sd[p + n].to(DEV)
    raise KeyError(n)


means, quats = _gg("means"), _gg("quats")
scales, opac = torch.exp(_gg("scales")), torch.sigmoid(_gg("opacities")).squeeze(-1)
colors = torch.cat([_gg("features_dc")[:, None, :], _gg("features_rest")], 1)
bg = torch.tensor([0.149, 0.1647, 0.2157], device=DEV)


def Rz(p):
    c, s = np.cos(p), np.sin(p)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def vm(pos, yaw, Tbc):
    pn = np.array([pos[0], -pos[1], -pos[2]])
    T = np.eye(4)
    T[:3, :3] = Rz(yaw)
    T[:3, 3] = pn
    c2w = Tw2g @ (T @ Tbc)
    R = c2w[:3, :3] * np.array([1, -1, -1])
    Ri = R.T
    V = np.eye(4)
    V[:3, :3] = Ri
    V[:3, 3] = -Ri @ c2w[:3, 3]
    return V


@torch.no_grad()
def rend(pos, yaw, W=1024, H=768):
    V = torch.tensor(vm(pos, yaw, Tbc_f), device=DEV, dtype=torch.float32)[None]
    r, a, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
                            viewmats=V, Ks=Kf, width=W, height=H, packed=False, near_plane=0.001,
                            far_plane=1e10, render_mode="RGB", sh_degree=3, rasterize_mode="classic")
    return ((r[..., :3] + (1 - a) * bg).clamp(0, 1).squeeze(0) * 255).byte().cpu().numpy()


def to(a, n, how):
    return np.asarray(Image.fromarray(a).resize((n, n), how), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", default=[100, 101, 150, 151])
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 20, 60])
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    print(f"{'ep/frame':12s} {'yaw sign':>9s} {'mean|Δ| 224':>12s} {'p95|Δ|':>8s} "
          f"{'corr':>6s}   (0-255 scale)")
    for i in a.eps:
        d = np.load(f"{RD}/data_gate_synth/ep_{i:04d}.npz", allow_pickle=True)
        st = d["state"].astype(np.float32)
        for t in a.frames:
            if t >= len(st):
                continue
            stored = to(d["image"][t], 224, Image.BICUBIC)
            # the client sends state[3] = -yaw, so the demo's stored state[3] IS -yaw
            best = None
            for sign, tag in ((-1.0, "-state[3]"), (1.0, "+state[3]")):
                live = to(to(rend(st[t, :3], sign * float(st[t, 3])), 256, Image.BILINEAR), 224, Image.BICUBIC)
                dif = np.abs(live.astype(np.float32) - stored.astype(np.float32))
                cor = np.corrcoef(live.ravel().astype(np.float32), stored.ravel().astype(np.float32))[0, 1]
                if best is None or dif.mean() < best[1]:
                    best = (tag, dif.mean(), np.percentile(dif, 95), cor, live)
            print(f"ep{i:04d}/{t:<5d} {best[0]:>9s} {best[1]:12.2f} {best[2]:8.1f} {best[3]:6.3f}")
            if a.dump:
                Image.fromarray(np.concatenate([stored, best[4]], 1)).save(
                    f"{a.dump}/cmp_ep{i:04d}_t{t}.png")


if __name__ == "__main__":
    main()
