"""ATTENTION-READOUT stage-1 head (Denis, 2026-08-06: pooled readout is a critical
flaw — the head must attend over image patches / text tokens, else the VLM's
structure never reaches c).

Token layout per row (34 x 2048, fp16): 16 tokens per camera (4x4 pooling of the
16x16 patch grid — keeps coarse "where") x 2 cameras + masked text mean + global
mean. Head = learned queries cross-attending the 34 tokens -> 1024-d readout ->
CFM v-net conditioning. Trained on the UNION row set (accuracy + basin lesson).

MODE=extract (openpi env, GPU): tokens for rendered_frames rows, fat_tube rows,
    and the gain2 basin frames -> ~2 GB fp16 npys.
MODE=train (tv env torch GPU): train head, report held R^2 + BASIN PROBE gains.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODE = os.environ.get("MODE", "train")
RUN = os.path.expanduser("~/ctxrun")
NTOK = 34

def _tube_rows(gc, ta):
    """Inlined from fat_tube_gen.sample_rows — that module runs stage code at import
    (same MODE env), which killed this pipeline mid-extract on 2026-08-06."""
    src = gc.load_eps(with_images=False)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(src)); tr = set(idx[:160].tolist())
    RADII = (0.25, 0.5, 1.0)
    rows = []
    for task in (gc.PROMPT_L, gc.PROMPT_R):
        eps_t = [i for i in sorted(tr) if src[i]["lang"] == task][:30]
        for ei in eps_t:
            e = src[ei]
            n = min(len(e["action"]), len(e["state"]) - 1)
            for t in range(0, n - gc.H, 24):
                rows.append((task, ei, t, np.zeros(3)))
                for _ in range(4):
                    r = RADII[rng.integers(len(RADII))]
                    v = rng.normal(size=3); v[0] *= 0.5
                    v = v / (np.linalg.norm(v) + 1e-9) * r
                    rows.append((task, ei, t, v))
    return src, rows

if MODE == "extract":
    import jax
    import jax.numpy as jnp
    import gate_ctx_common as gc
    import gate_traj_algebra as ta
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    policy = gc.make_policy()

    def ctx_tokens(raws):
        """(B, 34, 2048): per-camera 4x4-pooled patch tokens + text mean + global mean."""
        tds = [policy._input_transform(dict(r)) for r in raws]
        b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
        o = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
        n_img = len(o.images)
        tok, mask, ar = policy._model.embed_prefix(o)
        attn = make_attn_mask(mask, ar); pos = jnp.cumsum(mask, axis=1) - 1
        (pout, _), _ = policy._model.PaliGemma.llm([tok, None], mask=attn, positions=pos)
        pf = pout.astype(jnp.float32)                       # (B, T, D) — stays on device
        mk = mask.astype(jnp.float32)
        B, D = pf.shape[0], pf.shape[-1]
        img = pf[:, :2 * 256].reshape(B, 2, 16, 16, D)
        img = img.reshape(B, 2, 4, 4, 4, 4, D).mean(axis=(3, 5)).reshape(B, 32, D)
        imask = jnp.stack([o.image_masks[k].astype(jnp.float32)
                           for k in list(o.images)[:2]], 1)  # (B, 2)
        img = img * jnp.repeat(imask, 16, axis=1)[..., None]
        txt = pf[:, n_img * 256:]; tm = mk[:, n_img * 256:]
        tmean = (txt * tm[..., None]).sum(1) / jnp.clip(tm.sum(1, keepdims=True), 1e-6)
        gmean = (pf * mk[..., None]).sum(1) / jnp.clip(mk.sum(1, keepdims=True), 1e-6)
        out = jnp.concatenate([img, tmean[:, None], gmean[:, None]], 1)
        return np.asarray(out).astype(np.float16)

    def run_set(obs, out_name):
        chunks = []
        for i in range(0, len(obs), gc.BS):
            chunks.append(ctx_tokens(obs[i:i + gc.BS]))
            if i % (gc.BS * 20) == 0:
                print(f"{out_name} {i}/{len(obs)}", flush=True)
        np.save(f"{RUN}/{out_name}.npy", np.concatenate(chunks, 0))

    # set 1: on-route rendered rows (identical alignment to the thin cache)
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
    rf = np.load(f"{RUN}/rendered_frames.npz")
    rowmap = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rf["si"], rf["fidx"]))}
    FWD, WR = rf["fwd224"], rf["wrist224"]
    obs1 = []
    for si, e in groups:
        n = min(len(e["action"]), len(e["state"]) - 1)
        for t in range(0, n, 12):
            fi = int(e["fidx"][t]) if "fidx" in e else t
            i = rowmap[(si, fi)]
            obs1.append({"observation/image": FWD[i], "observation/wrist_image": WR[i],
                         "observation/state": e["state"][t], "prompt": e["lang"]})
    if not os.path.exists(f"{RUN}/tok_thin.npy"):
        run_set(obs1, "tok_thin")
    # set 2: fat tube rows
    src2, rows2 = _tube_rows(gc, ta)
    tf = np.load(f"{RUN}/fat_tube_frames.npz")
    F2, W2, S2 = tf["fwd"], tf["wr"], tf["st"]
    obs2 = [{"observation/image": F2[i], "observation/wrist_image": W2[i],
             "observation/state": S2[i], "prompt": rows2[i][0]} for i in range(len(rows2))]
    run_set(obs2, "tok_fat")
    # set 3: basin probe frames
    gf = np.load(f"{RUN}/gain2_frames.npz")
    import gate_ctx_common as _g
    obs3 = [{"observation/image": gf["fwd"][i], "observation/wrist_image": gf["wr"][i],
             "observation/state": gf["st"][i], "prompt": gc.PROMPT_L} for i in range(len(gf["st"]))]
    run_set(obs3, "tok_gain")
    print("TOK_EXTRACT_DONE", flush=True)
    sys.exit(0)

# MODE=train (tv env: torch + CUDA)
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

lbl = json.load(open(f"{RUN}/attn_labels.json"))
Y = np.array(lbl["Y"], np.float32); HE = np.array(lbl["held"], bool)
T1 = np.load(f"{RUN}/tok_thin.npy"); T2 = np.load(f"{RUN}/tok_fat.npy")
X = np.concatenate([T1, T2], 0).astype(np.float32)
assert len(X) == len(Y), (len(X), len(Y))
dev = "cuda"
torch.manual_seed(0)
xmu = X[~HE].mean((0, 1)); xsd = X[~HE].std((0, 1)) + 1e-6
ymu, ysd = Y[~HE].mean(0), Y[~HE].std(0) + 1e-6
Xt = torch.tensor((X - xmu) / xsd, dtype=torch.float32)
Yt = torch.tensor((Y - ymu) / ysd)


class AttnCFM(nn.Module):
    def __init__(self, dtok=2048, dh=256, nq=4, w=512, cdim=5):
        super().__init__()
        self.q = nn.Parameter(0.02 * torch.randn(nq, dh))
        self.k = nn.Linear(dtok, dh); self.v = nn.Linear(dtok, dh)
        self.vnet = nn.Sequential(nn.Linear(nq * dh + cdim + 1, w), nn.SiLU(),
                                  nn.Linear(w, w), nn.SiLU(),
                                  nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))

    def readout(self, tokens):                       # (B, 34, 2048) -> (B, 1024)
        K = self.k(tokens); V = self.v(tokens)
        sc = torch.einsum("qd,btd->bqt", self.q, K) / 16.0
        at = torch.softmax(sc, -1)
        return torch.einsum("bqt,btd->bqd", at, V).flatten(1)

    def forward(self, ct, t, ro):
        return self.vnet(torch.cat([ro, ct, t], 1))


net = AttnCFM().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)
tri = np.where(~HE)[0]
rng = np.random.default_rng(0)
for ep in range(60):
    perm = rng.permutation(tri); tot = 0.0
    for j in range(0, len(perm), 256):
        b = perm[j:j + 256]
        tk = Xt[b].to(dev); c1 = Yt[b].to(dev)
        c0 = torch.randn_like(c1); t = torch.rand(len(b), 1, device=dev)
        ro = net.readout(tk)
        loss = ((net((1 - t) * c0 + t * c1, t, ro) - (c1 - c0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
    if ep % 15 == 0 or ep == 59:
        print(f"epoch {ep} cfm-mse {tot/len(perm):.4f}", flush=True)
net.eval()


@torch.no_grad()
def sample(tk, k=8, steps=10):
    ro = net.readout(tk.to(dev)).repeat_interleave(k, 0)
    n = len(tk); c = torch.randn(n * k, 5, device=dev)
    for s in range(steps):
        t = torch.full((n * k, 1), s / steps, device=dev)
        c = c + net(c, t, ro) / steps
    return (c.reshape(n, k, 5).mean(1).cpu().numpy() * ysd + ymu)


P = sample(Xt[HE])
r2 = 1 - ((Y[HE] - P) ** 2).sum() / ((Y[HE] - Y[HE].mean(0)) ** 2).sum()
print(f"attn head held(on-route) R^2 (8-mean): {r2:.3f}", flush=True)
# basin probe (no openpi in the tv env: constants come from the labels json)
RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
U = np.load(os.path.join(RD, "pin_U_gate_rrr_k5.npy"))
astd = np.array(lbl["astd"], np.float32); H, AD = lbl["H"], lbl["AD"]
G = np.load(f"{RUN}/tok_gain.npy").astype(np.float32)
gf = np.load(f"{RUN}/gain2_frames.npz"); meta = gf["meta"]
Cg = sample(torch.tensor((G - xmu) / xsd, dtype=torch.float32))
D = (Cg @ U.T).reshape(-1, H, AD)[:, :, :3].sum(1) * astd[:3]
plist = []
for Dl in (0.25, 0.5, 1.0):
    for a in (1, 2):
        for sg in (+1, -1):
            plist.append((Dl, a, sg))
Gn = {}
for b in range(5):
    i0 = np.where((meta[:, 0] == b) & (meta[:, 1] == 0))[0][0]
    for pi, (Dl, ax, sg) in enumerate(plist, start=1):
        ii = np.where((meta[:, 0] == b) & (meta[:, 1] == pi))[0][0]
        Gn.setdefault((Dl, ax), []).append(-(D[ii, ax] - D[i0, ax]) / (sg * Dl))
print("attn head BASIN: y/z@.25 %.2f/%.2f  @.5 %.2f/%.2f  @1.0 %.2f/%.2f" % (
    np.mean(Gn[(0.25, 1)]), np.mean(Gn[(0.25, 2)]), np.mean(Gn[(0.5, 1)]),
    np.mean(Gn[(0.5, 2)]), np.mean(Gn[(1.0, 1)]), np.mean(Gn[(1.0, 2)])), flush=True)
torch.save({"state_dict": net.state_dict(), "xmu": xmu, "xsd": xsd, "ymu": ymu, "ysd": ysd,
            "ntok": NTOK, "H": H, "AD": AD, "K": 5, "arch": "attn-cfm-4q256"},
           os.path.join(RD, "attn_head_union.pt"))
print("saved attn_head_union.pt"); print("ATTN_TRAIN_DONE", flush=True)
