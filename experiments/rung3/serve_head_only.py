"""Head-only service (2026-09-03): the jointly-trained checkpoint's command head over HTTP, nothing
else. Used to hand OUR command to a different flow (the SDEdit-with-head ablation). Same component
selection as the joint server (GMM argmax with per-trial pi-hysteresis latch).

  SNMVP_HEAD=1 SNMVP_PIN_U=<U> ... python serve_head_only.py --ckpt <xswap> --norm <assets> --port 9200
POST /c  json {"trial", "prompt", "state": [7], "image": png_b64, "wrist": png_b64} -> {"c": [16], "j": int}
"""
import argparse
import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
import joint_head  # noqa: E402
joint_head.enable_head(os.environ.get("SNMVP_PIN_U", f"{RD}/pin_U_mh16.npy"))
import gate_ctx_common as gc  # noqa: E402
import openpi.policies.policy_config as _pc  # noqa: E402
import openpi.shared.normalize as _nz  # noqa: E402
import openpi.training.config as _cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--port", type=int, default=9200)
    a = ap.parse_args()
    cfg = _cfg.get_config(a.config)
    raw = _nz.load(a.norm)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=gc.pad_norm_stats(raw, cfg.model.action_dim))
    hyst = float(os.environ.get("SNMVP_GMM_HYST", "0.2"))
    latch = {}

    def dec(b):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"), np.uint8)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            self._send({"ckpt": a.ckpt})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            obs = {"observation/image": dec(req["image"]), "observation/wrist_image": dec(req["wrist"]),
                   "observation/state": np.asarray(req["state"], np.float32), "prompt": req.get("prompt", "")}
            c, w, mu, sig = joint_head.head_c(policy, [obs], return_gmm=True)
            w, mu = w[0], mu[0]
            j = int(w.argmax()); jp = latch.get(req.get("trial"))
            if jp is not None and w[jp] >= w[j] - hyst:
                j = jp
            latch[req.get("trial")] = j
            self._send({"c": np.asarray(mu[j], np.float32).tolist(), "j": j})

    print(f"[head_only] ready on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
