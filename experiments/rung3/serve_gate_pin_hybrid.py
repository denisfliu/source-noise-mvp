"""Two-stage flow, HYBRID stage 1: c ~ CFM([model_state, fused phi]) — state gives
the off-manifold restoring field, phi gives language/task. No one-hot."""
import argparse, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

class VNet(nn.Module):
    def __init__(self, xdim, cdim=5, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(xdim + cdim + 1, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))
    def forward(self, ct, t, x):
        return self.net(torch.cat([ct, t, x], 1))

class HybridPin:
    def __init__(self, policy, pin_u, head_path, steps=10, k=8):
        self.policy = policy
        self.U = np.load(pin_u).astype(np.float32)
        d = torch.load(head_path, map_location="cpu", weights_only=False)
        self.net = VNet(d["in_dim"]); self.net.load_state_dict(d["state_dict"]); self.net.eval()
        self.d = d; self.H, self.AD = d["H"], d["AD"]; self.steps = steps; self.k = k
        self._rng = np.random.default_rng()
        self._ema = None; self._prev_pos = None  # jitter analysis 2026-08-06: k-mean + EMA(0.5)
        print(f"[hybrid] CFM([state,phi])->c steps={steps} k-mean={k} ema=0.5", flush=True)

    def infer(self, obs):
        phi = gc.ctx_pool(self.policy, [obs])[0]
        ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        x = np.concatenate([ms, phi]).astype(np.float32)
        xn = torch.tensor(((x - self.d["xmu"]) / self.d["xsd"])[None], dtype=torch.float32)
        with torch.no_grad():
            xr = xn.repeat_interleave(self.k, 0)
            c = torch.randn(self.k, 5)
            for s in range(self.steps):
                t = torch.full((self.k, 1), s / self.steps)
                c = c + self.net(c, t, xr) / self.steps
        c = (c.mean(0).numpy() * self.d["ysd"] + self.d["ymu"]).astype(np.float32)
        pos = np.asarray(obs["observation/state"], np.float32)[:3]
        if self._prev_pos is None or np.linalg.norm(pos - self._prev_pos) > 0.5:
            self._ema = c          # new episode (position jump) -> reset the filter
        else:
            self._ema = 0.5 * c + 0.5 * self._ema
        self._prev_pos = pos; c = self._ema
        g = self._rng.standard_normal((self.H, self.AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(self.H, self.AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True); ap.add_argument("--head", required=True)
    ap.add_argument("--config", default="pi0_gate_aug"); ap.add_argument("--port", type=int, default=8815)
    a = ap.parse_args()
    cfg = _cfg.get_config(a.config)
    ns = gc.pad_norm_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    pin = HybridPin(policy, a.pin_u, a.head)
    print(f"[serve_gate_pin_hybrid] ready on ws://127.0.0.1:{a.port}", flush=True)
    WebsocketPolicyServer(pin, host="127.0.0.1", port=a.port).serve_forever()

if __name__ == "__main__":
    main()
