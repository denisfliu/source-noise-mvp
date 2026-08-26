"""Pin server with the ENUMERATION-FREE language prior: c = MLP([model_state, e64]),
e64 = PCA projection of the post-fusion language-token embedding of the live prompt.
No task list, no classifier, no string matching. The embedding is averaged over the
first LATCH_N calls of an episode then frozen (task semantics are episode-level;
per-frame re-embedding adds needless variance).
"""
import argparse
import os
import socket
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import pin_basis
from serve_gate_pin_democ import H, AD, pad_stats
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

RD = os.path.dirname(os.path.abspath(__file__))


class LangPinPolicy:
    def __init__(self, policy, pin_u_path, prior_path, ckpt_path=None):
        self.policy = policy
        self.U = np.load(pin_u_path).astype(np.float32)
        d = torch.load(prior_path, map_location="cpu", weights_only=False)
        pin_basis.verify(d, pin_u_path)
        if ckpt_path:
            pin_basis.verify_features(d, ckpt_path)
        self.nstate = d["nstate"]; self.K = d["K"]
        self.mu = d["mu"]; self.sd = d["sd"]; self.Em = d["Em"]; self.P = d["P"]
        layers, din = [], d["in_dim"]
        for hdim in d["hidden"]:
            layers += [nn.Linear(din, hdim), nn.SiLU()]; din = hdim
        layers += [nn.Linear(din, self.K)]
        self.prior = nn.Sequential(*layers); self.prior.load_state_dict(d["state_dict"]); self.prior.eval()
        # LATCH_N=0 -> recompute the language embedding every step (no latch). Latching
        # assumes the instruction's meaning is constant for the episode, which is false for
        # multi-stage instructions ("left gate twice, then right"): the active sub-goal is a
        # language x image reading that must keep running (Denis, 2026-08-09). It was also
        # inherited from the CLASSIFIER's argmax thrash, a discontinuity this regression head
        # does not have. Kept as a switch so the two can be compared.
        self.LATCH_N = int(os.environ.get("LATCH_N", "12"))
        # CLOG=<path>: record (position, e64, c) per inference so the served command can be
        # compared against the demo-consistent command at the same state. The prior is fit only
        # on demo states; once a long command carries the drone off that manifold, both its
        # inputs are extrapolations, and nothing in the offline metrics reports that.
        self.CLOG = os.environ.get("CLOG", "")
        self._log = []
        self._eacc = None; self._n = 0
        self._rng = np.random.default_rng()
        print(f"[langprior] in_dim={d['in_dim']} nstate={self.nstate}", flush=True)

    def _e64(self, obs):
        e = gc.lang_pool(self.policy, [dict(obs)])[0]
        return (e - self.Em) @ self.P

    def _record(self, obs, e64, c):
        if not self.CLOG:
            return
        pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
        self._log.append(np.concatenate([pos, c, e64]).astype(np.float32))
        np.save(self.CLOG, np.stack(self._log))

    def infer(self, obs):
        if self.LATCH_N <= 0:  # live: the instruction is re-read against the current view
            e64 = self._e64(obs)
            ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
            x = np.concatenate([ms, e64]).astype(np.float32)
            with torch.no_grad():
                c = self.prior(torch.tensor(((x - self.mu) / self.sd)[None]))[0].numpy()
            self._record(obs, e64, c)
            g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
            noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(H, AD).astype(np.float32)
            return self.policy.infer(obs, noise=noise)
        pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
        if np.linalg.norm(pos - np.array([0.0, 0.0, 1.5])) < 0.15 and self._n >= self.LATCH_N:
            self._eacc = None; self._n = 0
        if self._n < self.LATCH_N:
            e = self._e64(obs)
            self._eacc = e if self._eacc is None else self._eacc + e
            self._n += 1
            if self._n == self.LATCH_N:
                print(f"[langprior] embedding latched (norm {np.linalg.norm(self._eacc / self._n):.2f})", flush=True)
        e64 = self._eacc / max(self._n, 1)
        ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        x = np.concatenate([ms, e64]).astype(np.float32)
        with torch.no_grad():
            c = self.prior(torch.tensor(((x - self.mu) / self.sd)[None]))[0].numpy()
        self._record(obs, e64, c)
        g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(H, AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True); ap.add_argument("--prior", default=f"{RD}/langprior_rrr.pt")
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8828)
    a = ap.parse_args()
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind((a.host, a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    ns = pad_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    pin = LangPinPolicy(policy, a.pin_u, a.prior, ckpt_path=a.ckpt)
    print(f"[serve_gate_pin_langprior] ready on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(pin, host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
