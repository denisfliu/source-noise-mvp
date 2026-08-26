"""Serve the gate pi0 pin with a TRUE VLM-RRR coordinate: c is read off the pi0 VLM
prefix feature at inference (no state-MLP, no task one-hot, no progress clock).
  phi   = contextualized (post-fusion) prefix feature of the current obs (2048,)
  c     = ridge((phi - mu)/sg) + c0, clamped to the training c range
  noise = (I - U U^T) g + U c
Observation-grounded: c updates every step from what the VLM sees.

The feature and ridge-map definitions are imported from gate_ctx_common so the
serving-time phi is BY CONSTRUCTION identical to the extraction-time phi
(pitfall: match the prior's inputs to the serving client).
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class VlmcPinPolicy:
    def __init__(self, policy, pin_u_path, wc_path, H=50, AD=32, feat="context"):
        self.policy = policy
        self.U = np.load(pin_u_path).astype(np.float32)  # (H*AD, K)
        self.m = gc.load_ridge(wc_path)
        self.H = H; self.AD = AD; self._rng = np.random.default_rng()
        self.pool = gc.ctx_pool if feat == "context" else gc.prefusion_pool
        self.clog = os.environ.get("SNMVP_C_LOG")  # jsonl: per-call c + state (interpretability)
        self._n = 0
        kind = "ridge" if "W" in self.m else "mlp"
        print(f"[vlmc] U={self.U.shape} map={kind} feat={feat} clamp=on clog={self.clog}", flush=True)

    def infer(self, obs):
        phi = self.pool(self.policy, [obs])
        c = gc.apply_ridge(self.m, phi, clamp=True)[0].astype(np.float32)
        if self.clog:
            import json
            with open(self.clog, "a") as f:
                f.write(json.dumps({"i": self._n, "state": np.asarray(obs["observation/state"]).tolist(),
                                    "c": c.tolist()}) + "\n")
            self._n += 1
        g = self._rng.standard_normal((self.H, self.AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(self.H, self.AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True); ap.add_argument("--wc", required=True)
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--feat", default="context", choices=["context", "prefix"])
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8796)
    a = ap.parse_args()
    cfg = _cfg.get_config(a.config)
    ns = gc.pad_norm_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    pin = VlmcPinPolicy(policy, a.pin_u, a.wc, feat=a.feat)
    print(f"[serve_gate_pin_vlmc] ready on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(pin, host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
