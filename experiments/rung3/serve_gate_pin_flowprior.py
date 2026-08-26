"""Serve the pin with the FLOW-MATCHING command prior (flow_prior.py): per chunk,
sample ONE c from p(c|state,onehot) via Euler-10 — mode commitment instead of the
MLP's MSE mode-average. Same pin construction as every other server."""
import argparse, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

class VNet(nn.Module):
    def __init__(self, xdim, cdim=5, w=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(xdim + cdim + 1, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, cdim))
    def forward(self, ct, t, x):
        return self.net(torch.cat([ct, t, x], 1))

class FlowPriorPin:
    def __init__(self, policy, pin_u, prior_path, steps=10):
        self.policy = policy
        self.U = np.load(pin_u).astype(np.float32)
        d = torch.load(prior_path, map_location="cpu", weights_only=False)
        self.net = VNet(d["in_dim"]); self.net.load_state_dict(d["state_dict"]); self.net.eval()
        self.xmu, self.xsd, self.ymu, self.ysd = d["xmu"], d["xsd"], d["ymu"], d["ysd"]
        self.tasks = list(d["tasks"]); self.H, self.AD = d["H"], d["AD"]; self.steps = steps
        self._rng = np.random.default_rng()
        print(f"[flowprior] steps={steps} tasks={len(self.tasks)}", flush=True)

    def infer(self, obs):
        prompt = str(obs.get("prompt", ""))
        if prompt not in self.tasks:
            raise ValueError(f"prompt not in prior's task list: {prompt!r}")
        ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        oh = np.zeros(len(self.tasks), np.float32); oh[self.tasks.index(prompt)] = 1.0
        x = np.concatenate([ms, oh]).astype(np.float32)
        xn = torch.tensor(((x - self.xmu) / self.xsd)[None], dtype=torch.float32)
        with torch.no_grad():
            c = torch.randn(1, 5)
            for s in range(self.steps):
                t = torch.full((1, 1), s / self.steps)
                c = c + self.net(c, t, xn) / self.steps
        c = (c[0].numpy() * self.ysd + self.ymu).astype(np.float32)
        g = self._rng.standard_normal((self.H, self.AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(self.H, self.AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True); ap.add_argument("--prior", required=True)
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--port", type=int, default=8812)
    a = ap.parse_args()
    cfg = _cfg.get_config(a.config)
    ns = gc.pad_norm_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    pin = FlowPriorPin(policy, a.pin_u, a.prior)
    print(f"[serve_gate_pin_flowprior] ready on ws://127.0.0.1:{a.port}", flush=True)
    WebsocketPolicyServer(pin, host="127.0.0.1", port=a.port).serve_forever()

if __name__ == "__main__":
    main()
