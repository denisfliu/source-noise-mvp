"""SDEdit baseline for sketch commands (2026-09-02; Meng et al., ICLR 2022 "SDEdit: guided
image synthesis and editing with stochastic differential equations", applied to the flow
policy). The IDENTICAL sketch pipeline as the pin servers (same state machine, same prompt
swap, same resampled track) but the sketch is used the SDEdit way: the whole normalized
sketch chunk a_sketch is the guide, the sampler starts at an intermediate time t0 from
    x_{t0} = t0 * z + (1 - t0) * a_sketch,   z ~ N(0, I),
and integrates the UNMODIFIED policy's flow from t0 down to 0. No pin, no training: the
policy never sees a command subspace. t0 is the realism/faithfulness dial (small t0 = follow
the sketch nearly verbatim, kinematically naive; large t0 = the policy's own behaviour).

  SNMVP_SDEDIT_T0=0.5 python serve_gate_sdedit.py --ckpt <any pi0 ckpt> --norm <assets> \
      --sketch <json> --port <p>
Outside the sketch's active window the policy serves normally.
"""
import argparse
import os
import socket
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import gate_ctx_common as gc
from advice_prompt import AdvicePrompt
from sketch_prompt import SketchPrompt

import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
import openpi.training.config as _cfg
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

H, AD = 50, 32


class SDEditHeadPolicy:
    """SDEdit guided by OUR head's command (2026-09-03 ablation): each replan the observation goes
    to a head-only service (the xswap checkpoint's head), its c is decoded to the minimum-norm
    chunk U c, and the UNPINNED flow starts its integration at t0 from t0 z + (1-t0) U c. The same
    command information the pin carries, delivered training-free."""

    def __init__(self, policy, head_url, pin_u, t0):
        import urllib.request
        self.policy, self.t0 = policy, float(t0)
        self.U = np.load(pin_u).astype(np.float32)
        self.url = head_url.rstrip("/") + "/c"
        self._rng = np.random.default_rng(int(os.environ.get("SNMVP_NOISE_SEED", "0")))
        urllib.request.urlopen(head_url.rstrip("/") + "/health", timeout=10).read()
        print(f"[sdedit-head] guide = U c from {head_url}, t0={self.t0}", flush=True)

    def infer(self, obs):
        import base64, io, json, urllib.request
        from PIL import Image
        obs = dict(obs)
        trial = obs.pop("snmvp_trial", "default")
        def png(a):
            b = io.BytesIO(); Image.fromarray(np.asarray(a, np.uint8)).save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()
        req = {"trial": trial, "prompt": obs.get("prompt", ""), "state": np.asarray(obs["observation/state"], np.float32).tolist(),
               "image": png(obs["observation/image"]), "wrist": png(obs["observation/wrist_image"])}
        r = urllib.request.Request(self.url, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
        c = np.asarray(json.loads(urllib.request.urlopen(r, timeout=120).read())["c"], np.float32)
        guide = (self.U @ c).reshape(H, AD).astype(np.float32)
        z = self._rng.standard_normal((H, AD)).astype(np.float32)
        x_t0 = (self.t0 * z + (1.0 - self.t0) * guide).astype(np.float32)
        return self.policy.infer(obs, noise=x_t0, snmvp_t_start=self.t0)


class SDEditSketchPolicy:
    def __init__(self, policy, act_norm, sketch_path, t0, advice=False):
        self.policy = policy
        self.t0 = float(t0)
        if not 0.0 < self.t0 <= 1.0:
            raise ValueError(f"t0 must be in (0, 1], got {self.t0}")
        self._rng = np.random.default_rng(int(os.environ.get("SNMVP_NOISE_SEED", "0")))
        amean = np.asarray(act_norm.mean[:7], np.float32)
        astd = np.asarray(act_norm.std[:7], np.float32)
        # U is only used by SketchPrompt.step (the pin path); window() does not need it
        U0 = np.zeros((H * AD, 1), np.float32)
        self.sketch = (AdvicePrompt(sketch_path, amean, astd, U0) if advice
                       else SketchPrompt(sketch_path, amean, astd, U0))
        print(f"[sdedit] t0={self.t0}: {int(round(self.t0 * 10))} of 10 Euler steps from the sketch guide", flush=True)

    def infer(self, obs):
        obs = dict(obs)
        trial = obs.pop("snmvp_trial", "default")
        pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
        guide, _sig, prompt, _phase = self.sketch.window(trial, pos)
        if prompt is not None:
            obs["prompt"] = prompt
        if guide is None:
            return self.policy.infer(obs)          # armed / done: the policy's own noise, t = 1
        z = self._rng.standard_normal((H, AD)).astype(np.float32)
        x_t0 = (self.t0 * z + (1.0 - self.t0) * guide).astype(np.float32)
        return self.policy.infer(obs, noise=x_t0, snmvp_t_start=self.t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--sketch", default="", help="sketch json, or an advice json with --advice")
    ap.add_argument("--head-url", default="", help="head-only service: guide = U c (ablation B)")
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--advice", action="store_true", help="treat --sketch as an AdvicePrompt json (pursuit chunk guide)")
    ap.add_argument("--t0", type=float, default=float(os.environ.get("SNMVP_SDEDIT_T0", "0.5")))
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--port", type=int, default=8840)
    a = ap.parse_args()
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    raw = _nz.load(a.norm)
    ns = gc.pad_norm_stats(raw, cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    if a.head_url:
        pol = SDEditHeadPolicy(policy, a.head_url, a.pin_u, a.t0)
    else:
        if not a.sketch:
            raise SystemExit("need --sketch or --head-url")
        pol = SDEditSketchPolicy(policy, raw["actions"], a.sketch, a.t0, advice=a.advice)
    print(f"[sdedit] ready on ws://127.0.0.1:{a.port}", flush=True)
    WebsocketPolicyServer(policy=pol, host="127.0.0.1", port=a.port).serve_forever()


if __name__ == "__main__":
    main()
