"""Splice server: the no-clock 4-task prior for known task prompts; for "hold position"
the command is the COMPUTED hover c (zero-action chunk through segY @ U — geometry,
not learning). Serves the AUG flow (hover vocabulary)."""
import argparse, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

class SplicePolicy:
    def __init__(self, policy, pin_u, prior_path):
        self.policy = policy
        self.U = np.load(pin_u).astype(np.float32)
        d = torch.load(prior_path, map_location="cpu", weights_only=False)
        layers, din = [], d["in_dim"]
        for h in d["hidden"]:
            layers += [nn.Linear(din, h), nn.SiLU()]; din = h
        layers += [nn.Linear(din, 5)]
        self.prior = nn.Sequential(*layers); self.prior.load_state_dict(d["state_dict"]); self.prior.eval()
        self.mu, self.sd, self.tasks = d["mu"], d["sd"], list(d["tasks"])
        ns, amean, astd = gc.load_norm()
        self.c_hold = (gc.segY(np.zeros((gc.H, 7), np.float32), amean, astd) @ self.U).astype(np.float32)
        self.H, self.AD = gc.H, gc.AD
        self._rng = np.random.default_rng()
        print(f"[splice] tasks={len(self.tasks)} c_hold={np.round(self.c_hold,2)}", flush=True)

    def infer(self, obs):
        prompt = str(obs.get("prompt", ""))
        if prompt == "hold position":
            c = self.c_hold
        else:
            ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
            v = np.zeros(len(self.tasks), np.float32); v[self.tasks.index(prompt)] = 1.0
            x = np.concatenate([ms, v]).astype(np.float32)
            with torch.no_grad():
                c = self.prior(torch.tensor(((x - self.mu) / self.sd)[None]))[0].numpy().astype(np.float32)
        g = self._rng.standard_normal((self.H, self.AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(self.H, self.AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True); ap.add_argument("--prior", required=True)
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--port", type=int, default=8810)
    a = ap.parse_args()
    cfg = _cfg.get_config(a.config)
    ns = gc.pad_norm_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    print(f"[serve_gate_pin_splice] ready on ws://127.0.0.1:{a.port}", flush=True)
    WebsocketPolicyServer(SplicePolicy(policy, a.pin_u, a.prior), host="127.0.0.1", port=a.port).serve_forever()

if __name__ == "__main__":
    main()
