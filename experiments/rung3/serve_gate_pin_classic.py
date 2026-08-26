"""Serve the ORIGINAL working pin stack (the one that threaded the gate 100%/10 seeds
on 2026-08-03): hf_bundle gate_inference.GatePolicy(mode='pin') — gate_both_pin flow,
PCA pin_U_gate_k5, state+task-onehot MLP prior — wrapped for the websocket clients
(gate_lead_diag.py, gate_video_overlay.py obs schema). Reproduction of record, no edits.
"""
import argparse
import os
import sys

import numpy as np

B = os.path.expanduser("~/hf_bundle/gate-drone-pi0")
sys.path.insert(0, B)
from gate_inference import GatePolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class Adapter:
    def __init__(self, gp):
        self.gp = gp

    def infer(self, obs):
        img = np.asarray(obs["observation/image"])
        wrist = np.asarray(obs.get("observation/wrist_image", img))
        state = np.asarray(obs["observation/state"], np.float32)
        act = self.gp.infer(img, state, obs.get("prompt", ""), wrist=wrist)
        return {"actions": np.asarray(act, np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--mode", default="pin", choices=["pin", "scratch"])
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    ckpt = a.ckpt or os.path.join(B, "checkpoints", "gate_both_pin" if a.mode == "pin" else "gate_both_scratch")
    kw = dict(pin_U=os.path.join(B, "assets", "pin_U_gate_k5.npy"),
              prior=os.path.join(B, "assets", "prior_gate_mlp.pt")) if a.mode == "pin" else {}
    gp = GatePolicy(ckpt=ckpt, norm_path=os.path.join(B, "assets", "gate_nav"),
                    mode=a.mode, bgr2rgb=False, wrist="separate", **kw)
    print(f"[serve_gate_pin_classic] ready on ws://127.0.0.1:{a.port}", flush=True)
    WebsocketPolicyServer(Adapter(gp), host="127.0.0.1", port=a.port).serve_forever()


if __name__ == "__main__":
    main()
