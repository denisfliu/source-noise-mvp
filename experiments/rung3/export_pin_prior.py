"""Export the deployable pin prior: fit MLP([model-state, gate-onehot]) -> c on real gate data and save a
standalone artifact (prior_gate_mlp.pt) that gate_inference.py's pin mode loads. model-state = the state
after the pi0 input transform (matches the offline eval's mlp arm, which reached ~0.65 on real); c = U^T of
the normalized action chunk (gate U). Keyword left/right onehot (text embeddings wash out the minimal pair)."""
import json
import os
import numpy as np
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32
CKPT = os.path.expanduser("~/code/openpi/checkpoints/pi0_gate/gate_both_pin/4999")
NORM = os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav")
U = np.load(os.path.join(RD, "pin_U_gate_k5.npy")).astype(np.float32)
HIDDEN = [256, 256]


def resize224(im):
    return np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC)).astype(np.uint8)


def main():
    import openpi.training.config as _cfg
    import openpi.policies.policy_config as _pc
    import openpi.shared.normalize as _nz
    import torch, torch.nn as nn
    ns = _nz.load(NORM)
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
    policy = _pc.create_trained_policy(_cfg.get_config("pi0_gate"), CKPT, norm_stats=ns)

    def seg_to_c(seg):
        m, r = seg.shape
        seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
        ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
        return ch.reshape(-1) @ U

    meta = json.load(open(os.path.join(RD, "data_gate_real", "meta.json")))
    tasks = sorted({meta[k]["lang"] for k in meta})                    # onehot order = sorted lang strings
    tid = {t: i for i, t in enumerate(tasks)}
    X, C = [], []
    for k in sorted(meta):
        d = np.load(os.path.join(RD, "data_gate_real", k + ".npz"))
        T = len(d["action"])
        for t in range(0, T, 3):
            obs = {"observation/image": resize224(d["image"][t]), "observation/wrist_image": resize224(d["wrist"][t]),
                   "observation/state": d["state"][t].astype(np.float32), "prompt": meta[k]["lang"]}
            ms = np.asarray(policy._input_transform(dict(obs))["state"]).reshape(-1)
            oh = np.zeros(len(tasks), np.float32); oh[tid[meta[k]["lang"]]] = 1
            X.append(np.concatenate([ms, oh])); C.append(seg_to_c(d["action"][t:]))
    X, C = np.asarray(X, np.float32), np.asarray(C, np.float32)
    print(f"tasks={len(tasks)} frames={len(X)} in_dim={X.shape[1]} K={C.shape[1]}", flush=True)

    layers, din = [], X.shape[1]
    for h in HIDDEN:
        layers += [nn.Linear(din, h), nn.SiLU()]; din = h
    layers += [nn.Linear(din, C.shape[1])]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(X), torch.tensor(C)
    for step in range(4000):
        b = torch.randint(0, len(xt), (256,)); loss = ((net(xt[b]) - yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(xt).numpy()
    r2 = 1 - ((C - pred) ** 2).sum() / (((C - C.mean(0)) ** 2).sum() + 1e-9)
    print(f"prior train R^2 = {r2:.3f}", flush=True)
    out = os.path.join(RD, "prior_gate_mlp.pt")
    torch.save({"state_dict": net.state_dict(), "tasks": tasks, "H": H, "AD": AD,
                "in_dim": X.shape[1], "hidden": HIDDEN, "K": int(C.shape[1])}, out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
