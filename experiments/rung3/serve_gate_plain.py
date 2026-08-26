"""Plain pi0 server: policy.infer(obs) with the model's own N(0,I) source noise.
For scratch-arm evaluation — serving a pin-free flow through the pin server feeds
it noise whose 5 pinned coordinates are ~4-6 sigma off its training distribution
(Denis, 2026-08-07)."""
import argparse, os, socket, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--port", type=int, default=8822)
    a = ap.parse_args()
    # Fail before the (slow) model load if the port is taken — a stale server on
    # this port would otherwise silently serve the wrong checkpoint (2026-08-07).
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", a.port)); probe.close()
    cfg = _cfg.get_config(a.config)
    ns = gc.pad_norm_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    print(f"[serve_gate_plain] ready on ws://127.0.0.1:{a.port}", flush=True)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=a.port).serve_forever()

if __name__ == "__main__":
    main()
