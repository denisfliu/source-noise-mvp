"""FAT-TUBE coverage fix (basin probe, 2026-08-06): the feature heads' stability
basin ends at the demo/cache tube (~0.3 m). Generate rendered observations in a
+-1.0 m tube around LEFT/RIGHT demo routes with RETURN-TO-ROUTE labels, retrain
the feature heads, and re-gate on the basin probe.

Labels are demo-derived: for a state offset delta off demo (e, t), the label chunk
is the demo's own continuation with the return displacement -delta folded into the
first RETURN_STEPS actions — "fly back onto the demo and continue". No sim ground
truth anywhere.

MODE=render  (tv env, GPU): sample tube points, render fwd+wrist, save fat_tube_frames.npz
MODE=extract (openpi env, GPU): fused phi per row -> fat_tube_phi.npy
MODE=build   (openpi env, CPU): labels + train pure-feature and hybrid CFM heads,
             report held R^2, save *_fat.pt heads
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODE = os.environ.get("MODE", "build")
RUN = os.path.expanduser("~/ctxrun")
STRIDE, RETURN_STEPS = 24, 10
RADII = (0.25, 0.5, 1.0)
N_EP_PER_TASK = 30

import gate_ctx_common as gc  # noqa: E402

RD = gc.RD
rng_global = np.random.default_rng(0)


def sample_rows():
    """Deterministic row list shared by all stages: (task, ep_index, t, delta)."""
    src = gc.load_eps(with_images=False)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(src)); tr = set(idx[:160].tolist())
    rows = []
    for task in (gc.PROMPT_L, gc.PROMPT_R):
        eps_t = [i for i in sorted(tr) if src[i]["lang"] == task][:N_EP_PER_TASK]
        for ei in eps_t:
            e = src[ei]
            n = min(len(e["action"]), len(e["state"]) - 1)
            for t in range(0, n - gc.H, STRIDE):
                rows.append((task, ei, t, np.zeros(3)))
                for _ in range(4):
                    r = RADII[rng.integers(len(RADII))]
                    v = rng.normal(size=3); v[0] *= 0.5          # thinner along-route
                    v = v / (np.linalg.norm(v) + 1e-9) * r
                    rows.append((task, ei, t, v))
    return src, rows


if MODE == "render":
    import torch
    from PIL import Image
    from gsplat import rasterization
    src, rows = sample_rows()
    SCENES = {}
    LEFT_CK = ("/home/ubuntu/code/falsify-pi/data/gate_scenes_export/left_scene/mocap_outputs/"
               "sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt")
    LEFT_M = np.array([[0.12614431661544656, 2.138646801849853e-06, -0.00025306576654559085, -0.15671883492487332],
                       [-2.138646801849853e-06, -0.1261265572041315, -0.0021319289354524646, -0.08013551648879384],
                       [-0.00025306576654559085, 0.0021319289354524646, -0.12612630156484925, -0.18772133850562778],
                       [0, 0, 0, 1.0]])
    RIGHT_CK = ("/home/ubuntu/code/falsify-pi/data/gate_scenes_export/right_scene/mocap_outputs/"
                "sagesplat_mocap/sagesplat/2026-05-11_144353/nerfstudio_models/step-000029999.ckpt")
    RIGHT_M = (np.array([[0.136708, -0.001053, 0.006031, -0.111938],
                         [0.00108, 0.13684, -0.000588, 0.030456],
                         [-0.006027, 0.000635, 0.136711, -0.201447],
                         [0, 0, 0, 1.0]]) @ np.diag([1.0, -1, -1, 1]))
    Kf = torch.tensor([[502.2632, 0., 506.3971], [0., 500.6736, 385.41], [0., 0., 1.]], device="cuda")[None].float()
    Kd = torch.tensor([[478.2450, 0., 511.9041], [0., 476.7944, 383.5003], [0., 0., 1.]], device="cuda")[None].float()
    Tbc_f = np.array([[0, 0, -1, 0.10], [1, 0, 0, -0.03], [0, -1, 0, -0.01], [0, 0, 0, 1.]])
    Tbc_d = np.array([[0, 1, 0, 0.0], [1, 0, 0, 0.0], [0, 0, -1, 0.05], [0, 0, 0, 1.]])
    _ov = np.asarray(Image.open("/home/ubuntu/code/falsify-pi/configs/embodiments/assets/"
                                "carl_wrist_overlay_pinhole_rgb.png").convert("RGBA").resize((256, 256), Image.BILINEAR), np.uint8)

    def load_scene(ck):
        sd = torch.load(ck, map_location="cuda", weights_only=False)["pipeline"]
        def gg(n):
            for p in ("_model.gauss_params.", "_model."):
                if p + n in sd:
                    return sd[p + n].to("cuda")
        return dict(means=gg("means"), quats=gg("quats"), scales=torch.exp(gg("scales")),
                    opac=torch.sigmoid(gg("opacities")).squeeze(-1),
                    colors=torch.cat([gg("features_dc")[:, None, :], gg("features_rest")], 1))

    def Rz(p):
        c, s = np.cos(p), np.sin(p); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

    bg = torch.tensor([0.149, 0.1647, 0.2157], device="cuda")

    def rend(S, M, pos, yaw, Tbc, Kk):
        pn = np.array([pos[0], -pos[1], -pos[2]]); T = np.eye(4); T[:3, :3] = Rz(yaw); T[:3, 3] = pn
        c2w = M @ (T @ Tbc); R = c2w[:3, :3] * np.array([1, -1, -1]); Ri = R.T
        V = np.eye(4); V[:3, :3] = Ri; V[:3, 3] = -Ri @ c2w[:3, 3]
        Vt = torch.tensor(V, device="cuda", dtype=torch.float32)[None]
        with torch.no_grad():
            r, a, _ = rasterization(means=S["means"], quats=S["quats"], scales=S["scales"],
                                    opacities=S["opac"], colors=S["colors"], viewmats=Vt, Ks=Kk,
                                    width=1024, height=768, packed=False, near_plane=0.001,
                                    far_plane=1e10, render_mode="RGB", sh_degree=3, rasterize_mode="classic")
        return ((r[..., :3] + (1 - a) * bg).clamp(0, 1).squeeze(0) * 255).byte().cpu().numpy()

    def to256(a): return np.asarray(Image.fromarray(a).resize((256, 256), Image.BILINEAR), np.uint8)
    def ov(x):
        rgb = _ov[..., :3].astype(np.float32); al = _ov[..., 3:4].astype(np.float32) / 255.
        return (al * rgb + (1 - al) * x.astype(np.float32)).clip(0, 255).astype(np.uint8)
    def r224(a): return np.asarray(Image.fromarray(a).resize((224, 224), Image.BICUBIC), np.uint8)

    SCENES[gc.PROMPT_L] = (load_scene(LEFT_CK), LEFT_M)
    SCENES[gc.PROMPT_R] = (load_scene(RIGHT_CK), RIGHT_M)
    F, W, ST = [], [], []
    for k, (task, ei, t, dv) in enumerate(rows):
        e = src[ei]
        pos = e["state"][t, :3].astype(np.float64) + dv
        yaw = float(e["state"][t, 3])
        S, M = SCENES[task]
        F.append(r224(to256(rend(S, M, pos, -yaw, Tbc_f, Kf))))
        W.append(r224(ov(to256(rend(S, M, pos, -yaw, Tbc_d, Kd)))))
        ST.append(np.array([*pos, yaw, 0, 0, 0], np.float32))
        if k % 400 == 0:
            print(f"render {k}/{len(rows)}", flush=True)
    np.savez(f"{RUN}/fat_tube_frames.npz", fwd=np.stack(F), wr=np.stack(W), st=np.stack(ST))
    print("FAT_RENDER_DONE", len(rows), flush=True)
    sys.exit(0)

if MODE == "extract":
    src, rows = sample_rows()
    policy = gc.make_policy()
    rf = np.load(f"{RUN}/fat_tube_frames.npz")
    # materialize ONCE — NpzFile subscripting in a loop re-decompresses the whole
    # archive per access (2026-08-04 livelock rule; violated here on first run)
    FWD, WR, ST = rf["fwd"], rf["wr"], rf["st"]
    obs = [{"observation/image": FWD[i], "observation/wrist_image": WR[i],
            "observation/state": ST[i], "prompt": rows[i][0]} for i in range(len(rows))]
    phis = []
    for i in range(0, len(obs), gc.BS):
        phis.append(gc.ctx_pool(policy, obs[i:i + gc.BS]))
        if i % (gc.BS * 20) == 0:
            print(f"extract {i}/{len(obs)}", flush=True)
    np.save(f"{RUN}/fat_tube_phi.npy", np.concatenate(phis, 0))
    print("FAT_EXTRACT_DONE", flush=True)
    sys.exit(0)

# MODE=build (CPU): labels + heads
import torch
import torch.nn as nn

ns, amean, astd = gc.load_norm()
U = np.load(os.path.join(RD, "pin_U_gate_rrr_k5.npy"))
H = gc.H
src, rows = sample_rows()
phi = np.load(f"{RUN}/fat_tube_phi.npy")
rf = np.load(f"{RUN}/fat_tube_frames.npz")
policy = gc.make_policy()
assert len(phi) == len(rows)
Y = []
for task, ei, t, dv in rows:
    e = src[ei]
    chunk = e["action"][t:t + H].astype(np.float32).copy()
    if chunk.shape[0] < H:
        chunk = np.concatenate([chunk, np.zeros((H - len(chunk), chunk.shape[1]), np.float32)])
    chunk[:RETURN_STEPS, :3] -= dv.astype(np.float32) / RETURN_STEPS   # fly back onto the demo
    Y.append((gc.segY(chunk, amean, astd) @ U).astype(np.float32))
Y = np.stack(Y)
FWD, WR, ST = rf["fwd"], rf["wr"], rf["st"]
ms = np.stack([np.asarray(policy._input_transform(
    {"observation/image": FWD[i], "observation/wrist_image": WR[i],
     "observation/state": ST[i], "prompt": rows[i][0]})["state"]).reshape(-1)
    for i in range(len(rows))])
he = np.array([i % 10 == 9 for i in range(len(rows))])   # 10% held for sanity
torch.manual_seed(0)


class VNet(nn.Module):
    def __init__(self, xdim, cdim=5, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(xdim + cdim + 1, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))

    def forward(self, ct, t, x):
        return self.net(torch.cat([ct, t, x], 1))


def train_head(X, name):
    xmu, xsd = X[~he].mean(0), X[~he].std(0) + 1e-6
    ymu, ysd = Y[~he].mean(0), Y[~he].std(0) + 1e-6
    Xn = torch.tensor((X - xmu) / xsd, dtype=torch.float32)
    Yn = torch.tensor((Y - ymu) / ysd)
    net = VNet(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)
    tri = np.where(~he)[0]
    rng = np.random.default_rng(0)
    for ep_i in range(100):
        perm = rng.permutation(tri)
        for j in range(0, len(perm), 512):
            b = perm[j:j + 512]
            c1 = Yn[b]; c0 = torch.randn_like(c1)
            t = torch.rand(len(b), 1)
            loss = ((net((1 - t) * c0 + t * c1, t, Xn[b]) - (c1 - c0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        n = int(he.sum()); xr = Xn[he].repeat_interleave(8, 0)
        c = torch.randn(n * 8, 5)
        for s in range(10):
            t = torch.full((n * 8, 1), s / 10)
            c = c + net(c, t, xr) / 10
        P = (c.reshape(n, 8, 5).mean(1) * torch.tensor(ysd) + torch.tensor(ymu)).numpy()
    r2 = 1 - ((Y[he] - P) ** 2).sum() / ((Y[he] - Y[he].mean(0)) ** 2).sum()
    print(f"{name}: held R^2 (8-mean) {r2:.3f}  rows {len(X)}", flush=True)
    torch.save({"state_dict": net.state_dict(), "xmu": xmu.astype(np.float32),
                "xsd": xsd.astype(np.float32), "ymu": ymu.astype(np.float32),
                "ysd": ysd.astype(np.float32), "in_dim": X.shape[1], "H": H,
                "AD": gc.AD, "K": 5, "arch": f"cfm-3x512-{name}"},
               os.path.join(RD, f"{name}.pt"))


train_head(phi.astype(np.float32), "vlmflow_head_fat")
train_head(np.concatenate([ms, phi], 1).astype(np.float32), "hybrid_head_fat")
print("FAT_BUILD_DONE", flush=True)
