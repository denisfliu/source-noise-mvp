"""Real demo frame vs the Gaussian-splat render from the same pose (2026-09-26), to show how far simulated
observations sit from the training images. Real frames are the policy's own 256x256 forward and downward images from
gate_nav3 (raw fisheye, RGB); renders go through the rollout's own chain (pinhole at each camera's calibrated K,
1024x768 -> 256x256, strut overlay on the downward view), pose = the episode's state (mocap position, render yaw =
-state yaw, as gate_rollout_batch does).

  ~/code/openpi/.venv/bin/python real_vs_splat.py --extract   # real frames + poses -> real_vs_splat_frames.npz (needs pyarrow)
  /home/dfliu/code/tv/bin/python real_vs_splat.py             # renders, writes real_vs_splat.png (needs gsplat)
"""
import io, os, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SP = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda"
DS = os.path.expanduser("~/.cache/huggingface/lerobot/local/gate_nav3/data/chunk-000")
FALSIFY = "/home/dfliu/code/falsify/data/gate_scenes_export"
SCENES = {   # checkpoint and world-to-splat transform, copied from gate_rollout_batch.py
    "left": (f"{FALSIFY}/left_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt",
             np.array([[0.12614431661544656, 2.138646801849853e-06, -0.00025306576654559085, -0.15671883492487332],
                       [-2.138646801849853e-06, -0.1261265572041315, -0.0021319289354524646, -0.08013551648879384],
                       [-0.00025306576654559085, 0.0021319289354524646, -0.12612630156484925, -0.18772133850562778], [0, 0, 0, 1.]])),
    "right": (f"{FALSIFY}/right_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_144353/nerfstudio_models/step-000029999.ckpt",
              np.array([[0.136708, -0.001053, 0.006031, -0.111938], [0.00108, 0.13684, -0.000588, 0.030456],
                        [-0.006027, 0.000635, 0.136711, -0.201447], [0, 0, 0, 1.]]) @ np.diag([1., -1, -1, 1]))}
# (scene, real episode, fraction of the episode): start, approach, at the gate
SHOTS = [("left", 0, 0.05), ("left", 0, 0.30), ("left", 0, 0.45), ("right", 50, 0.05), ("right", 50, 0.30), ("right", 50, 0.45)]
FRAMES = f"{SP}/real_vs_splat_frames.npz"


def extract():
    import pyarrow.parquet as pq
    imgs, down, poses = [], [], []
    for scene, ep, frac in SHOTS:
        t = pq.read_table(f"{DS}/episode_{ep:06d}.parquet").to_pandas(); i = int(frac * len(t))
        imgs.append(np.asarray(Image.open(io.BytesIO(t["image"][i]["bytes"])).convert("RGB").resize((256, 256))))
        down.append(np.asarray(Image.open(io.BytesIO(t["wrist_image"][i]["bytes"])).convert("RGB").resize((256, 256))))
        poses.append(np.r_[np.asarray(t["state"][i], np.float64)[:4], i])
    np.savez(FRAMES, imgs=np.stack(imgs), down=np.stack(down), poses=np.stack(poses))


if "--extract" in sys.argv:
    extract(); raise SystemExit(f"wrote {FRAMES}")

import torch  # noqa: E402
from gsplat import rasterization  # noqa: E402

Kf = torch.tensor([[502.2632, 0., 506.3971], [0., 500.6736, 385.41], [0., 0., 1.]], device=DEV)[None].float()
Kd = torch.tensor([[478.2450, 0., 511.9041], [0., 476.7944, 383.5003], [0., 0., 1.]], device=DEV)[None].float()
Tbc_f = np.array([[0, 0, -1, 0.10], [1, 0, 0, -0.03], [0, -1, 0, -0.01], [0, 0, 0, 1.]])
Tbc_d = np.array([[0, 1, 0, 0.0], [1, 0, 0, 0.0], [0, 0, -1, 0.05], [0, 0, 0, 1.]])
STRUT = np.asarray(Image.open("/home/dfliu/code/falsify/configs/embodiments/assets/carl_wrist_overlay_pinhole_rgb.png")
                   .convert("RGBA").resize((256, 256), Image.BILINEAR), np.float32)
BG = torch.tensor([0.149, 0.1647, 0.2157], device=DEV)


def load(ck):
    sd = torch.load(ck, map_location=DEV, weights_only=False)["pipeline"]
    g = lambda n: next(sd[p + n].to(DEV) for p in ("_model.gauss_params.", "_model.") if p + n in sd)
    return dict(means=g("means"), quats=g("quats"), scales=torch.exp(g("scales")), opacities=torch.sigmoid(g("opacities")).squeeze(-1),
                colors=torch.cat([g("features_dc")[:, None, :], g("features_rest")], 1))


def viewmat(pos, yaw, Tw2g, Tbc):
    c, s = np.cos(yaw), np.sin(yaw); T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [pos[0], -pos[1], -pos[2]]
    c2w = Tw2g @ (T @ Tbc); R = c2w[:3, :3] * np.array([1, -1, -1]); V = np.eye(4); V[:3, :3] = R.T; V[:3, 3] = -R.T @ c2w[:3, 3]
    return V


@torch.no_grad()
def render(G, pos, yaw, Tw2g, down=False):
    V = torch.tensor(viewmat(pos, yaw, Tw2g, Tbc_d if down else Tbc_f), device=DEV, dtype=torch.float32)[None]
    r, a, _ = rasterization(**G, viewmats=V, Ks=Kd if down else Kf, width=1024, height=768, packed=False, near_plane=0.001, far_plane=1e10,
                            render_mode="RGB", sh_degree=3, rasterize_mode="classic")
    img = ((r[..., :3] + (1 - a) * BG).clamp(0, 1).squeeze(0) * 255).byte().cpu().numpy()
    img = np.asarray(Image.fromarray(img).resize((256, 256), Image.BILINEAR), np.float32)
    if down:   # the strut the rollout paints over the downward view
        al = STRUT[..., 3:] / 255.; img = al * STRUT[..., :3] + (1 - al) * img
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def main():
    Z = np.load(FRAMES); cache, rows = {}, []
    for (scene, ep, _), img, dn, s in zip(SHOTS, Z["imgs"], Z["down"], Z["poses"]):
        if scene not in cache:
            cache.clear(); torch.cuda.empty_cache(); cache[scene] = load(SCENES[scene][0])
        G, T = cache[scene], SCENES[scene][1]
        rows.append(([Image.fromarray(img), render(G, s[:3], -s[3], T), Image.fromarray(dn), render(G, s[:3], -s[3], T, down=True)],
                     f"{scene} gate, ep {ep}, frame {int(s[4])}"))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})
    labels = ["Forward\nreal", "Forward\nsplat", "Downward\nreal", "Downward\nsplat"]
    fig = plt.figure(figsize=(7.2, 5.15))
    # four image rows (a gap between the two cameras), six columns (a gap between the two scenes)
    gs = fig.add_gridspec(5, 7, height_ratios=[1, 1, 0.07, 1, 1], width_ratios=[1, 1, 1, 0.07, 1, 1, 1],
                          left=0.09, right=0.995, top=0.93, bottom=0.005, wspace=0.03, hspace=0.03)
    R, C = [0, 1, 3, 4], [0, 1, 2, 4, 5, 6]; top = {}
    for c, (ims, _) in zip(C, rows):
        for r, lab, im in zip(R, labels, ims):
            ax = fig.add_subplot(gs[r, c]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                top[c] = ax
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9, rotation=0, ha="right", va="center", labelpad=4)
    for c0, name in ((0, "Left Gate"), (4, "Right Gate")):
        l, r_ = top[c0].get_position(), top[c0 + 2].get_position()
        fig.text((l.x0 + r_.x1) / 2, 0.955, name, ha="center", va="bottom", fontsize=10.5)
        fig.add_artist(plt.Line2D([l.x0 + 0.005, r_.x1 - 0.005], [0.948, 0.948], color="black", lw=0.6))
    fig.savefig(f"{SP}/real_vs_splat.png", dpi=300); fig.savefig(f"{SP}/real_vs_splat.pdf", dpi=300)
    print("\n".join(r[1] for r in rows)); print(f"{SP}/real_vs_splat.png")


if __name__ == "__main__":
    main()
