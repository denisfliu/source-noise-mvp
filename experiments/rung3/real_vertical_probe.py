"""Open-loop chunks on REAL frames: pin (xswap, own head + sigma map) vs pi0 scratch3 vs the real
continuation (2026-09-11; Denis: "pi0.5 and ours kept going down in real; do ours and scratch look the same?").
Anchors = the 76 real anchors of the real-anchor suite (synthpin_xswap.npz meta). Reports net dz / dxy over the
first 8 steps (what the hardware executes per replan) and over the full 50-step chunk, per arm, vs the real
pilot's next 8 / 50 steps, plus the altitude profile of the real and synth demonstrations."""
import os, sys, json, gc
import numpy as np
RD = "/home/dfliu/code/source-noise-mvp/experiments/rung3"; sys.path.insert(0, RD)
ARM = sys.argv[1]  # scratch | pin | report
if ARM == "pin":
    for k, v in dict(SNMVP_HEAD="1", SNMVP_ZERO_PAD_ACTIONS="1", SNMVP_PIN_U=f"{RD}/pin_U_mh16.npy", SNMVP_HEAD_DETACH="0",
                     SNMVP_HEAD_LAM="0.3", SNMVP_HEAD_GMM="1", SNMVP_PIN_NOISE="1.5", SNMVP_PIN_NOISE_RAND="1", SNMVP_PIN_NOISE_COND="1").items():
        os.environ.setdefault(k, v)
from PIL import Image
if ARM != "report":
    from openpi.training import config as _cfg
    from openpi.policies import policy_config as _pc
    from openpi.shared import normalize as _nz
    import joint_head
EMU = os.environ.get("EMU", "")   # "bgr": emulate the pre-fix client (channel order sent verbatim = swapped)
def r224(im):
    im = np.asarray(im)
    if EMU == "bgr": im = im[:, :, ::-1]
    return np.asarray(Image.fromarray(np.ascontiguousarray(im)).resize((224, 224), Image.BICUBIC), np.uint8)
PROMPT = {"left": "go through the gate on the left and hover over the stuffed animal", "right": "go through the gate on the right and hover over the stuffed animal"}
z = np.load("/home/dfliu/ctxrun/synthpin_xswap.npz", allow_pickle=True); m = z["meta"]
m = json.loads(str(m)) if m.ndim == 0 else list(m)
if isinstance(m, str): m = json.loads(m)
anchors = [(r["side"], int(r["e"]), int(r["t"]), float(r["frac"])) for r in m]
eps = {}
def ep(e):
    if e not in eps: eps[e] = np.load(f"{RD}/data_gate_real/ep_{e:04d}.npz", allow_pickle=True)
    return eps[e]
if ARM != "report":
    cfg = _cfg.get_config("pi0_gate"); ns = dict(_nz.load("/home/dfliu/hf_bundle/gate-drone-pi0/assets/gate_nav"))
U = np.load(f"{RD}/pin_U_mh16.npy").astype(np.float32); H, AD = 50, 32
sigmap = json.load(open(os.environ.get("SIGMAP", f"{RD}/sigma_map_xswap.json"))); xs, ys, cap = np.asarray(sigmap["sig_star"], np.float32), np.asarray(sigmap["sig_serve"], np.float32), float(sigmap["cap"])
rng = np.random.default_rng(0)
out = {"real": [], "scratch": [], "pin": [], "frac": [], "side": []}
for side, e, t, frac in anchors:
    d = ep(e); st = d["state"].astype(np.float32)
    n8, n50 = min(8, len(st) - 1 - t), min(50, len(st) - 1 - t)
    out["real"].append((st[t + n8, :3] - st[t, :3], st[t + n50, :3] - st[t, :3], n8, n50)); out["frac"].append(frac); out["side"].append(side)
def obs_at(e, t, side):
    d = ep(e)
    return {"observation/image": r224(d["image"][t]), "observation/wrist_image": r224(d["wrist"][t]),
            "observation/state": d["state"].astype(np.float32)[t], "prompt": PROMPT[side]}
OUTF = "/home/dfliu/ctxrun/realism/real_vertical_%s" + (("_" + EMU) if EMU else "") + (("_" + os.environ["PIN_TAG"]) if os.environ.get("PIN_TAG") else "") + (("_sig" + os.environ["FORCE_SIGMA"]) if os.environ.get("FORCE_SIGMA") else "") + ".npz"
if ARM == "scratch":
    policy = _pc.create_trained_policy(cfg, os.environ.get("SCR_CK", "/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/4999"), norm_stats=ns)
    rows, chunks = [], []
    for side, e, t, frac in anchors:
        a = np.asarray(policy.infer(obs_at(e, t, side))["actions"], np.float32)[:H, :3]
        rows.append(np.stack([a[:8].sum(0), a.sum(0)])); chunks.append(a)
    np.savez(OUTF % os.environ.get("SCR_TAG", "scratch"), rows=np.asarray(rows), chunks=np.asarray(chunks)); print("saved scratch"); sys.exit(0)
sig_serves = []
if ARM == "pin":
  policy = _pc.create_trained_policy(cfg, os.environ.get("PIN_CK", "/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999"), norm_stats=ns)
  rows, chunks = [], []
  for side, e, t, frac in anchors:
    obs = obs_at(e, t, side)
    c, w, mu, sig, cache = joint_head.head_c(policy, [obs], return_gmm=True, return_cache=True)
    j = int(w[0].argmax()); cc = mu[0, j]; sstar = float(np.linalg.norm(sig[0, j])); ss = float(np.clip(np.interp(sstar, xs, ys), 0, cap))
    if os.environ.get("FORCE_SIGMA"): ss = float(os.environ["FORCE_SIGMA"])
    sig_serves.append(ss)
    g = rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
    noise = (g - (g @ U) @ U.T + (cc @ U.T)).reshape(H, AD).astype(np.float32)
    a = np.asarray(policy.infer(obs, noise=noise, snmvp_sigma=ss, cache=cache)["actions"], np.float32)[:H, :3]
    rows.append(np.stack([a[:8].sum(0), a.sum(0)])); chunks.append(a)
  np.savez(OUTF % "pin", rows=np.asarray(rows), chunks=np.asarray(chunks), sig_serve=np.asarray(sig_serves)); print("saved pin"); sys.exit(0)
sc = np.load(OUTF % "scratch")["rows"]; pn = np.load(OUTF % "pin")["rows"]
out["scratch"] = [(r[0], r[1]) for r in sc]; out["pin"] = [(r[0], r[1]) for r in pn]
frac = np.asarray(out["frac"]); side = np.asarray(out["side"])
def summarize(name, arr8, arr50, mask=None):
    mask = np.ones(len(arr8), bool) if mask is None else mask
    a8, a50 = np.asarray(arr8)[mask], np.asarray(arr50)[mask]
    return (f"{name:28s} n={mask.sum():2d} | first 8 steps: dz med {np.median(a8[:,2]):+.3f} m (p10 {np.percentile(a8[:,2],10):+.3f}, p90 {np.percentile(a8[:,2],90):+.3f}), "
            f"desc frac {np.mean(a8[:,2] < -0.01):.2f}, |dxy| med {np.median(np.linalg.norm(a8[:,:2],axis=1)):.3f} | 50 steps: dz med {np.median(a50[:,2]):+.3f}, |dxy| med {np.median(np.linalg.norm(a50[:,:2],axis=1)):.2f}")
real8 = [r[0] for r in out["real"]]; real50 = [r[1] for r in out["real"]]
sc8 = [r[0] for r in out["scratch"]]; sc50 = [r[1] for r in out["scratch"]]
pn8 = [r[0] for r in out["pin"]]; pn50 = [r[1] for r in out["pin"]]
for label, mask in [("ALL anchors", None), ("early (frac<0.35)", frac < 0.35), ("mid (0.35-0.7)", (frac >= 0.35) & (frac < 0.7)), ("late (frac>=0.7)", frac >= 0.7)]:
    print("==", label)
    print(summarize("real pilot continuation", real8, real50, mask)); print(summarize("pi0 scratch3", sc8, sc50, mask)); print(summarize("pin xswap (own head)", pn8, pn50, mask))
print("== per-anchor agreement with the real dz sign (first 8 steps):")
r8 = np.asarray(real8); print(f"   scratch agrees {np.mean(np.sign(np.asarray(sc8)[:,2]) == np.sign(r8[:,2])):.2f}, pin agrees {np.mean(np.sign(np.asarray(pn8)[:,2]) == np.sign(r8[:,2])):.2f}; corr(dz scratch, dz pin) = {np.corrcoef(np.asarray(sc8)[:,2], np.asarray(pn8)[:,2])[0,1]:.2f}")
print("== anchor altitude and demo altitude profiles")
zs = [ep(e)["state"][t, 2] for _, e, t, _ in anchors]; print(f"   anchor z: med {np.median(zs):.2f} [{np.min(zs):.2f}, {np.max(zs):.2f}]")
for dom, path, ids in [("real", f"{RD}/data_gate_real", range(0, 100, 5)), ("synth", f"{RD}/data_gate_synth3", range(0, 200, 10))]:
    zz = [np.load(f"{path}/ep_{e:04d}.npz", allow_pickle=True)["state"][:, 2] for e in ids]
    print(f"   {dom} demos: z start med {np.median([q[0] for q in zz]):.2f}, min-over-episode med {np.median([q.min() for q in zz]):.2f}, end med {np.median([q[-1] for q in zz]):.2f}")
