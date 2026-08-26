"""Pin server with a DEMO-DERIVED oracle command: c = inverse-distance-weighted mean of
the k nearest demo states' chunk coordinates (same task, demos only — no sim ground truth).
This is the "perfect pin prediction" bound for component-wise evaluation (Denis,
2026-08-07): it measures the flow's execution ceiling with commands as good as the demos
can define, isolating execution from prediction.
"""
import argparse
import glob
import json
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.transforms import NormStats
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

RD = os.path.dirname(os.path.abspath(__file__))
H, AD = 50, 32
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = LEFT.replace("left", "right")
CFL = "go through the center gate from the left and hover over the stuffed animal"
CFR = "go through the center gate from the right and hover over the stuffed animal"


def pad_stats(ns, dim):
    out = {}
    for k, s in ns.items():
        n = np.asarray(s.mean).shape[-1]
        if n >= dim:
            out[k] = s; continue
        p = dim - n
        ext = lambda a, f: None if a is None else np.concatenate([np.asarray(a, np.float32), np.full(p, f, np.float32)])
        out[k] = NormStats(mean=ext(s.mean, 0.), std=ext(s.std, 1.), q01=ext(s.q01, 0.), q99=ext(s.q99, 1.))
    return out


class DemoCommandBank:
    """Per-task bank of (demo state, chunk coordinate c) pairs; query by k-NN in state space."""

    def __init__(self, pin_u_path, amean, astd, domains="synth", stride=2, k=5):
        self.U = np.load(pin_u_path).astype(np.float32)
        self.k = k
        self.amean, self.astd = amean, astd
        eps = []  # (lang, state (T,7), action (T,7))
        if domains in ("synth", "both"):
            for i, f in enumerate(sorted(glob.glob(f"{RD}/data_gate_synth/ep_*.npz"))):
                d = np.load(f, allow_pickle=True)
                lang = (CFL, CFR, LEFT, RIGHT)[i // 50]
                eps.append((lang, d["state"].astype(np.float32), d["action"].astype(np.float32)))
        if domains in ("real", "both"):
            meta = json.load(open(f"{RD}/data_gate_real/meta.json"))
            for key in sorted(meta):
                d = np.load(f"{RD}/data_gate_real/{key}.npz", allow_pickle=True)
                eps.append((meta[key]["lang"], d["state"].astype(np.float32), d["action"].astype(np.float32)))
        self.bank = {}  # lang -> (S standardized, C, s_mu, s_sd)
        by = {}
        for lang, st, ac in eps:
            by.setdefault(lang, []).append((st, ac))
        for lang, pairs in by.items():
            S, C = [], []
            for st, ac in pairs:
                for t in range(0, len(st) - 1, stride):
                    S.append(st[t]); C.append(self._c_of(ac[t:t + H]))
            S = np.array(S, np.float32); C = np.array(C, np.float32)
            s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
            self.bank[lang] = ((S - s_mu) / s_sd, C, s_mu, s_sd)
            print(f"[democ] bank {lang[:40]!r}: {len(S)} rows", flush=True)

    def _c_of(self, chunk7):
        L = len(chunk7)
        ch = np.zeros((H, AD), np.float32)
        m = min(L, H)
        ch[:m, :7] = (chunk7[:m] - self.amean[:7]) / (self.astd[:7] + 1e-6)
        if m < H:
            ch[m:, :7] = ch[m - 1, :7]
        return ch.reshape(-1) @ self.U

    def command(self, prompt, raw_state):
        p = str(prompt).strip()
        if p not in self.bank:
            raise ValueError(f"prompt not in demo bank: {p!r}")
        S, C, s_mu, s_sd = self.bank[p]
        q = (np.asarray(raw_state, np.float32).reshape(-1) - s_mu) / s_sd
        d = np.linalg.norm(S - q, axis=1)
        idx = np.argpartition(d, self.k)[:self.k]
        w = 1.0 / (d[idx] + 1e-6)
        return (w[:, None] * C[idx]).sum(0) / w.sum()


class DemoCPinPolicy:
    def __init__(self, policy, bank):
        self.policy = policy; self.bank = bank
        self.U = bank.U
        self._rng = np.random.default_rng()

    def infer(self, obs):
        c = self.bank.command(obs.get("prompt", ""), obs["observation/state"])
        g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(H, AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--domains", default="synth", choices=["synth", "real", "both"])
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8823)
    a = ap.parse_args()
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind((a.host, a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    ns = pad_stats(_nz.load(a.norm), cfg.model.action_dim)
    amean = np.asarray(ns["actions"].mean); astd = np.asarray(ns["actions"].std)
    bank = DemoCommandBank(a.pin_u, amean, astd, domains=a.domains)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    print(f"[serve_gate_pin_democ] ready on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(DemoCPinPolicy(policy, bank), host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
