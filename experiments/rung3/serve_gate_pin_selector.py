"""Grounded-command pin server (task #6 closed-loop): the winning clockless prior, with
the task one-hot supplied by the VLM TASK SELECTOR (4-way MLP on the policy's own pooled
prefix features — vision + text) instead of exact prompt-string matching. First live-loop
removal of the one-hot scaffold: language/scene understanding selects the movement; state
carries geometry; the pin carries the movement.
"""
import argparse
import os
import socket
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_gate_pin_democ import H, AD, pad_stats
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

RD = os.path.dirname(os.path.abspath(__file__))


class Selector:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.mu, self.sg = z["mu"], z["sg"]
        self.W1, self.b1, self.W2, self.b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        self.tasks = [str(t) for t in z["tasks"]]

    def probs(self, feat2048):
        x = (feat2048 - self.mu) / self.sg
        h = x @ self.W1 + self.b1
        h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
        logits = h @ self.W2 + self.b2
        e = np.exp(logits - logits.max())
        return e / e.sum()


class SelectorPinPolicy:
    def __init__(self, policy, pin_u_path, prior_path, selector_path):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        self._jax, self._jnp, self._model = jax, jnp, _model
        self.policy = policy
        self.U = np.load(pin_u_path).astype(np.float32)
        self.sel = Selector(selector_path)
        d = torch.load(prior_path, map_location="cpu", weights_only=False)
        self.tasks = d["tasks"]; self.K = d["K"]
        self.mu = d["mu"].astype(np.float32); self.sd = d["sd"].astype(np.float32)
        layers, din = [], d["in_dim"]
        for hdim in d["hidden"]:
            layers += [nn.Linear(din, hdim), nn.SiLU()]; din = hdim
        layers += [nn.Linear(din, self.K)]
        self.prior = nn.Sequential(*layers); self.prior.load_state_dict(d["state_dict"]); self.prior.eval()
        self.slot = [self.tasks.index(t) for t in self.sel.tasks]  # selector order -> prior slots
        self._rng = np.random.default_rng()
        self.n_calls = 0; self.n_by_task = np.zeros(4, int)
        # Task selection is an EPISODE-level decision (v2 finding: per-frame re-voting
        # thrashes commands mid-flight). Accumulate probs over the first LATCH_N calls of
        # an episode, then lock. Episode boundary = state back at the spawn point.
        self.LATCH_N = 12
        self._acc = np.zeros(4); self._acc_n = 0; self._locked = None
        print(f"[selector] prior in_dim={d['in_dim']} tasks aligned {self.slot} latch_n={self.LATCH_N}", flush=True)

    def _prefix_feat(self, obs):
        # POST-FUSION pooling (gate_ctx_common.ctx_pool) — the selector was trained on
        # contextualized features; pre-fusion pooling washes out language (2026-08-03
        # finding; the pre-fusion variant made the selector output a constant, 2026-08-08).
        import gate_ctx_common as gc
        return gc.ctx_pool(self.policy, [dict(obs)])[0]

    def infer(self, obs):
        pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
        if np.linalg.norm(pos - np.array([0.0, 0.0, 1.5])) < 0.15 and self._acc_n >= self.LATCH_N:
            self._acc[:] = 0.0; self._acc_n = 0; self._locked = None  # new episode
        if self._locked is None:
            p = self.sel.probs(self._prefix_feat(obs))
            self._acc += p; self._acc_n += 1
            sel_ix = int(self._acc.argmax())
            if self._acc_n >= self.LATCH_N:
                self._locked = sel_ix
                print(f"[selector] LATCHED {self.sel.tasks[sel_ix].split('gate')[1][:20]!r} "
                      f"acc {np.round(self._acc / self._acc_n, 3)}", flush=True)
        else:
            sel_ix = self._locked
        oh = np.zeros(4, np.float32)
        oh[self.slot[sel_ix]] = 1.0
        self.n_calls += 1; self.n_by_task[self.slot[sel_ix]] += 1
        ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        x = np.concatenate([ms, oh]).astype(np.float32)
        with torch.no_grad():
            c = self.prior(torch.tensor(((x - self.mu) / self.sd)[None]))[0].numpy()
        g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(H, AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True); ap.add_argument("--prior", required=True)
    ap.add_argument("--selector", default=f"{RD}/task_selector.npz")
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8827)
    a = ap.parse_args()
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind((a.host, a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    ns = pad_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    pin = SelectorPinPolicy(policy, a.pin_u, a.prior, a.selector)
    print(f"[serve_gate_pin_selector] ready on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(pin, host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
