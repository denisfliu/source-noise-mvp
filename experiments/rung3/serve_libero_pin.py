"""Pin server for the LIBERO ladder arms.

Command sources (--source):
  prior  c = MLP([model_state, task_onehot40])  — libero_prior.py, the ladder's source
  oracle c from the nearest demo states of the SAME task (k-NN over demo state -> chunk c),
         demos only — the execution ceiling, so a null ladder result can be attributed to
         command quality vs the flow (the drone component battery showed this is essential)
  plain  no pin at all (for scratch arms; identical to serve_gate_plain)

The client sends the task instruction as `prompt`; the task index is resolved by exact match
against the LIBERO task list (a scaffold, as on the drone).
"""
import argparse
import glob
import json
import os
import socket
import sys

import numpy as np
import pin_basis
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
import openpi.training.config as _cfg
from openpi import transforms as _T
from openpi.transforms import NormStats
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

RD = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero")
H, AD = 50, 32


def pads(nsd, dim):
    out = {}
    for k, s in nsd.items():
        n = np.asarray(s.mean).shape[-1]
        if n >= dim:
            out[k] = s; continue
        p = dim - n
        ext = lambda a, f: None if a is None else np.concatenate(
            [np.asarray(a, np.float32), np.full(p, f, np.float32)])
        out[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
    return out


class DemoOracle:
    """k-NN over (demo state -> chunk command) pairs of the same task. Demos only."""

    def __init__(self, U, nrm, task_text, k=5, stride=4):
        import pyarrow.parquet as pq
        self.U, self.k = U, k
        self.bank = {}
        by_task = {}
        for f in sorted(glob.glob(f"{SRC}/data/chunk-*/episode_*.parquet")):
            tb = pq.read_table(f, columns=["state", "actions", "task_index"])
            ti = int(tb.column("task_index")[0].as_py())
            by_task.setdefault(ti, []).append(
                (np.asarray(tb.column("state").to_pylist(), np.float32),
                 np.asarray(tb.column("actions").to_pylist(), np.float32)))
        for ti, eps in by_task.items():
            S, C = [], []
            for st, ac in eps:
                for t in range(0, max(len(st) - H, 1), stride):
                    ch = np.zeros((H, AD), np.float32)
                    m = min(H, len(ac) - t)
                    ch[:m, :ac.shape[1]] = ac[t:t + m]
                    if m < H:
                        ch[m:, :ac.shape[1]] = ac[min(t + m, len(ac)) - 1]
                    S.append(st[t]); C.append(nrm({"actions": ch})["actions"].reshape(-1) @ U)
            S = np.array(S, np.float32); C = np.array(C, np.float32)
            mu, sd = S.mean(0), S.std(0) + 1e-6
            self.bank[task_text[ti]] = ((S - mu) / sd, C, mu, sd)
        print(f"[oracle] banks for {len(self.bank)} tasks", flush=True)

    def __call__(self, prompt, state):
        S, C, mu, sd = self.bank[prompt]
        q = (np.asarray(state, np.float32).reshape(-1) - mu) / sd
        d = np.linalg.norm(S - q, axis=1)
        ix = np.argpartition(d, self.k)[:self.k]
        w = 1.0 / (d[ix] + 1e-6)
        return (w[:, None] * C[ix]).sum(0) / w.sum()


class LiberoPinPolicy:
    def __init__(self, policy, U, source, prior_path, nrm, task_text):
        self.policy, self.U, self.source = policy, U, source
        self._rng = np.random.default_rng()
        self.prior = self.oracle = None
        if source == "prior":
            d = torch.load(prior_path, map_location="cpu", weights_only=False)
            pin_basis.verify(d, pin_u_path)
            self.tasks = list(d["tasks"]); self.ntask = int(d["ntask"])
            self.mu, self.sd = d["mu"], d["sd"]
            layers, din = [], d["in_dim"]
            for h in d["hidden"]:
                layers += [nn.Linear(din, h), nn.SiLU()]; din = h
            layers += [nn.Linear(din, d["K"])]
            self.prior = nn.Sequential(*layers)
            self.prior.load_state_dict(d["state_dict"]); self.prior.eval()
            print(f"[prior] in_dim={d['in_dim']} tasks={self.ntask}", flush=True)
        elif source == "oracle":
            self.oracle = DemoOracle(U, nrm, task_text)

    def _c(self, obs):
        prompt = str(obs.get("prompt", "")).strip()
        if self.source == "oracle":
            return self.oracle(prompt, obs["observation/state"])
        if prompt not in self.tasks:
            raise ValueError(f"prompt not in LIBERO task list: {prompt!r}")
        oh = np.zeros(self.ntask, np.float32); oh[self.tasks.index(prompt)] = 1.0
        ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        x = np.concatenate([ms, oh]).astype(np.float32)
        with torch.no_grad():
            return self.prior(torch.tensor(((x - self.mu) / self.sd)[None]))[0].numpy()

    def infer(self, obs):
        if self.source == "plain":
            return self.policy.infer(obs)
        c = self._c(obs)
        g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(H, AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--source", default="prior", choices=["prior", "oracle", "plain"])
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_rrr_k5_shared.npy")
    ap.add_argument("--prior", default=f"{RD}/libero_prior.pt")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8860)
    a = ap.parse_args()
    probe = socket.socket(); probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind((a.host, a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    ns = _nz.load(os.path.expanduser("~/code/openpi/assets/pi0_libero_shared/physical-intelligence/libero"))
    nsp = pads(ns, cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=nsp)
    U = np.load(a.pin_u).astype(np.float32)
    tasks = [json.loads(l) for l in open(f"{SRC}/meta/tasks.jsonl")]
    task_text = {t["task_index"]: t["task"] for t in tasks}
    pol = LiberoPinPolicy(policy, U, a.source, a.prior, _T.Normalize(nsp, use_quantiles=False), task_text)
    print(f"[serve_libero_pin:{a.source}] ready on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(pol, host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
