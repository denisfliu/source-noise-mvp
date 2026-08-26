"""Fit the closed-loop priors on data_libero_multi, using the SAME state transform the server uses
(policy._input_transform), so state features match at inference. Saves prior_state.npz (W for
state->c) and prior_statelang.npz (W for [state,onehot]->c, plus the task ordering). Env:
SNMVP_CKPT (flow ckpt), SNMVP_U (pin U .npy), SNMVP_NORM (norm dir)."""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
sys.path.insert(0, RD)
import pca_pin as PP  # noqa: E402
import openpi.training.config as C  # noqa: E402
import openpi.policies.policy_config as PC  # noqa: E402
import openpi.shared.normalize as NZ  # noqa: E402
H, AD = 50, 32


def r224(im):
    if im.shape[:2] != (224, 224):
        im = np.asarray(Image.fromarray(im).resize((224, 224), Image.BICUBIC))
    return im.astype(np.uint8)


def make_obs(ep, t, pr):
    im = r224(ep["image"][t]); wr = r224(ep["wrist"][t]) if "wrist" in ep else im.copy()
    return {"observation/image": im, "observation/wrist_image": wr,
            "observation/state": ep["state"][t], "prompt": pr}


def main():
    U = np.load(os.environ["SNMVP_U"]).astype(np.float32)
    ns = NZ.load(os.environ.get("SNMVP_NORM", os.path.join(RD, "norm_shared_libero")))
    amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
    pol = PC.create_trained_policy(C.get_config("pi0_libero_shared"), os.environ["SNMVP_CKPT"], norm_stats=ns)

    raw = os.path.join(RD, "data_libero_multi")
    meta = json.load(open(os.path.join(raw, "meta.json")))
    keys = sorted(meta)
    tasks = sorted({meta[k]["task"] for k in keys})
    tid = {t: i for i, t in enumerate(tasks)}

    def seg_to_c(seg):
        m, r = seg.shape
        seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
        ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
        return ch.reshape(-1) @ U

    St, Ct, Oh = [], [], []
    for k in keys:
        d = np.load(os.path.join(raw, k + ".npz"))
        ep = {"image": d["image"], "wrist": d["wrist"], "state": d["state"].astype(np.float32),
              "action": d["action"].astype(np.float32)}
        T = len(ep["action"]); onh = np.eye(len(tasks))[tid[meta[k]["task"]]]
        for t in range(0, T, max(1, T // 5)):
            St.append(np.asarray(pol._input_transform(make_obs(ep, t, meta[k]["lang"]))["state"]).reshape(-1))
            Ct.append(seg_to_c(ep["action"][t:])); Oh.append(onh)
    St, Ct, Oh = np.asarray(St), np.asarray(Ct), np.asarray(Oh)

    def fit(X):
        Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
        W, *_ = np.linalg.lstsq(Xb, Ct, rcond=None)
        return W
    np.savez(os.path.join(RD, "prior_state.npz"), W=fit(St))
    np.savez(os.path.join(RD, "prior_statelang.npz"), W=fit(np.concatenate([St, Oh], 1)), tasks=np.array(tasks))
    print(f"PRIOR_FIT_DONE n={len(St)} tasks={tasks}")


if __name__ == "__main__":
    main()
