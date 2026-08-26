"""Self-contained inference for the gate drone pi0 policy.

Loads a pi0_gate checkpoint and runs the EXACT training-time preprocessing, then receding-horizon
inference. Two modes:
  mode="scratch"  standard pi0 (recommended for first real flights): infer(obs) -> action chunk.
  mode="pin"      research pin: MLP prior -> c injected into the source noise; needs an exported prior.

Deployment notes baked in (see __init__ args):
  * COLOR: the model was trained on RGB (the converter swapped the legacy BGR fisheye to RGB). If your
    serving feeds BGR, keep bgr2rgb=True so this module swaps to match training. If you already feed RGB,
    set bgr2rgb=False. Flying with the wrong channel order silently degrades perception.
  * CAMERAS: training used image + wrist_image. wrist="separate" expects both; wrist="dup" duplicates the
    main image into the wrist slot (what the offline eval did when a wrist feed was missing).
  * RATE: training data is 10 fps, 7-D EE-delta actions. Execute the chunk at ~10 Hz and replan (see
    RecedingHorizon) rather than running all 50 steps open-loop.
"""
import numpy as np
from PIL import Image


class GatePolicy:
    def __init__(self, ckpt, norm_path, config="pi0_gate", mode="scratch",
                 pin_U=None, prior=None, bgr2rgb=True, wrist="separate", img_size=224):
        import openpi.training.config as _cfg
        import openpi.policies.policy_config as _pc
        import openpi.shared.normalize as _nz
        self._ns = _nz.load(norm_path)
        self.policy = _pc.create_trained_policy(_cfg.get_config(config), ckpt, norm_stats=self._ns)
        self.mode, self.bgr2rgb, self.wrist, self.sz = mode, bgr2rgb, wrist, img_size
        if mode == "pin":
            import torch, torch.nn as nn
            self.U = np.load(pin_U).astype(np.float32)                       # (H*AD, K)
            d = torch.load(prior, map_location="cpu", weights_only=False)
            self.tasks = d["tasks"]; self.H, self.AD = d["H"], d["AD"]
            layers, din = [], d["in_dim"]
            for h in d["hidden"]:
                layers += [nn.Linear(din, h), nn.SiLU()]; din = h
            layers += [nn.Linear(din, d["K"])]
            self.prior = nn.Sequential(*layers); self.prior.load_state_dict(d["state_dict"]); self.prior.eval()
            self._torch = torch

    def _img(self, im):
        im = np.asarray(im)
        if self.bgr2rgb:
            im = np.ascontiguousarray(im[..., ::-1])
        if im.shape[:2] != (self.sz, self.sz):
            im = np.asarray(Image.fromarray(im).resize((self.sz, self.sz), Image.BICUBIC))
        return im.astype(np.uint8)

    def _obs(self, image, wrist, state, prompt):
        img = self._img(image)
        wr = self._img(wrist) if (self.wrist == "separate" and wrist is not None) else img.copy()
        return {"observation/image": img, "observation/wrist_image": wr,
                "observation/state": np.asarray(state, np.float32), "prompt": str(prompt)}

    def _onehot(self, prompt):
        p = str(prompt).lower()
        v = np.zeros(len(self.tasks), np.float32)
        # keyword match (robust for the left/right minimal pair; text embeddings wash it out)
        for i, t in enumerate(self.tasks):
            key = "left" if "left" in str(t).lower() else ("right" if "right" in str(t).lower() else None)
            if key and key in p:
                v[i] = 1.0
        if v.sum() == 0:
            v[0] = 1.0  # fallback
        return v

    def infer(self, image, state, prompt, wrist=None, rng=None):
        """Returns the 7-D action chunk (H, 7). image/wrist are HxWx3 uint8 in YOUR color order."""
        obs = self._obs(image, wrist, state, prompt)
        if self.mode == "pin":
            ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
            oh = self._onehot(prompt)
            with self._torch.no_grad():
                x = self._torch.tensor(np.concatenate([ms, oh])[None].astype(np.float32))
                c = self.prior(x)[0].numpy()
            g = (rng or np.random.default_rng()).standard_normal((self.H, self.AD)).astype(np.float32)
            gf = g.reshape(-1)
            noise = (gf - (gf @ self.U) @ self.U.T + (c @ self.U.T)).reshape(self.H, self.AD).astype(np.float32)
            out = self.policy.infer(obs, noise=noise)
        else:
            out = self.policy.infer(obs)
        return np.asarray(out["actions"])[:, :7]


class RecedingHorizon:
    """Replan every `replan` steps, execute the chunk open-loop between. Returns one action per step()."""
    def __init__(self, gp, replan=8):
        self.gp, self.replan, self._chunk, self._i = gp, replan, None, 0

    def step(self, image, state, prompt, wrist=None):
        if self._chunk is None or self._i >= self.replan:
            self._chunk = self.gp.infer(image, state, prompt, wrist=wrist); self._i = 0
        a = self._chunk[self._i]; self._i += 1
        return a  # 7-D action for this control tick


if __name__ == "__main__":  # smoke test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--mode", default="scratch"); ap.add_argument("--pin_U", default=None); ap.add_argument("--prior", default=None)
    a = ap.parse_args()
    gp = GatePolicy(a.ckpt, a.norm, mode=a.mode, pin_U=a.pin_U, prior=a.prior)
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    st = np.zeros(7, np.float32)
    act = gp.infer(img, st, "go through the gate on the left and hover over the stuffed animal", wrist=img)
    print("SMOKE_OK action chunk", act.shape, "dtype", act.dtype, "range", float(act.min()), float(act.max()))
