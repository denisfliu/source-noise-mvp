"""Serve the ZERO-PIN control flow: noise = (I - UU^T) g, coarse coordinates 0 — matching
its training source. No command channel; the flow flies on vision alone. Control for the
2026-08-08 question: does source consistency alone (without the answer in it) explain the
pin's low-data advantage?"""
import argparse
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_gate_pin_democ import H, AD, pad_stats
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class ZeroPinPolicy:
    def __init__(self, policy, U):
        self.policy = policy; self.U = U
        self._rng = np.random.default_rng()

    def infer(self, obs):
        g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T).reshape(H, AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8826)
    a = ap.parse_args()
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind((a.host, a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    ns = pad_stats(_nz.load(a.norm), cfg.model.action_dim)
    U = np.load(a.pin_u).astype(np.float32)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    print(f"[serve_gate_zeropin] ready on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(ZeroPinPolicy(policy, U), host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
