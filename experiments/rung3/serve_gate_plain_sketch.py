"""Scratch-sketch mechanism control (2026-08-30): the IDENTICAL sketch pipeline — same
state machine, same prompt swap, same pinned-noise construction — served through a
checkpoint that was NEVER trained to read the noise subspace. If sketches work through the
source-noise channel, this must fail; anything it achieves is creditable to the prompt
swap alone.

  python serve_gate_plain_sketch.py --ckpt <scratch> --norm <assets> --pin-u <U> \
      --sketch <json> --port <p>
"""
import argparse
import os
import socket
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import gate_ctx_common as gc
from sketch_prompt import SketchPrompt

import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
import openpi.training.config as _cfg
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

H, AD = 50, 32


class PlainSketchPolicy:
    def __init__(self, policy, pin_u, act_norm, sketch_path):
        self.policy = policy
        self.U = np.load(pin_u).astype(np.float32)
        self._rng = np.random.default_rng(int(os.environ.get("SNMVP_NOISE_SEED", "0")))
        amean = np.asarray(act_norm.mean[:7], np.float32)
        astd = np.asarray(act_norm.std[:7], np.float32)
        self.sketch = SketchPrompt(sketch_path, amean, astd, self.U)

    def infer(self, obs):
        obs = dict(obs)
        trial = obs.pop("snmvp_trial", "default")
        pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
        c, _sig, prompt, phase = self.sketch.step(trial, pos)
        if prompt is not None:
            obs["prompt"] = prompt
        if c is None:
            return self.policy.infer(obs)          # plain: model draws its own noise
        g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(H, AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_mh16.npy")
    ap.add_argument("--sketch", required=True)
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--port", type=int, default=8830)
    a = ap.parse_args()
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    raw = _nz.load(a.norm)
    ns = gc.pad_norm_stats(raw, cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    pin = PlainSketchPolicy(policy, a.pin_u, raw["actions"], a.sketch)
    print(f"[plain_sketch] ready on ws://127.0.0.1:{a.port}", flush=True)
    WebsocketPolicyServer(policy=pin, host="127.0.0.1", port=a.port).serve_forever()


if __name__ == "__main__":
    main()
