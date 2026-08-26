"""Additive-edit CONTROL for the pin mechanism (Denis, 2026-08-07): serve a PLAIN flow
(no pin training), then overwrite the coarse component algebraically in normalized chunk
space: a' = a + U (c - U^T a), with c from the demo-oracle bank. If this matches the
pin-trained flow on the same battery, pin training buys nothing beyond the projection;
if it produces incoherent chunks, that is direct evidence the pin-trained complement is
conditioned on c. The edit and the pin carry identical coarse content — only whether the
flow generated the residual consistently with it differs.
"""
import argparse
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_gate_pin_democ import H, DemoCommandBank, pad_stats
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class AdditivePolicy:
    def __init__(self, policy, bank, amean, astd):
        self.policy = policy; self.bank = bank; self.U = bank.U
        self.amean = amean[:7]; self.astd = astd[:7]

    def infer(self, obs):
        c = self.bank.command(obs.get("prompt", ""), obs["observation/state"])
        result = dict(self.policy.infer(obs))
        a = np.asarray(result["actions"], np.float32)  # (H, 7) physical units
        n = min(len(a), H)
        ch = np.zeros((H, 32), np.float32)
        ch[:n, :7] = (a[:n, :7] - self.amean) / (self.astd + 1e-6)
        if n < H:
            ch[n:, :7] = ch[n - 1, :7]
        y = ch.reshape(-1)
        y = y + (c - y @ self.U) @ self.U.T
        edited = y.reshape(H, 32)[:n, :7] * (self.astd + 1e-6) + self.amean
        out = np.array(a, np.float32)
        out[:n, :7] = edited
        result["actions"] = out
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--domains", default="synth", choices=["synth", "real", "both"])
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8824)
    a = ap.parse_args()
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind((a.host, a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    ns = pad_stats(_nz.load(a.norm), cfg.model.action_dim)
    amean = np.asarray(ns["actions"].mean); astd = np.asarray(ns["actions"].std)
    bank = DemoCommandBank(a.pin_u, amean, astd, domains=a.domains)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    print(f"[serve_gate_additive] ready on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(AdditivePolicy(policy, bank, amean, astd), host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
