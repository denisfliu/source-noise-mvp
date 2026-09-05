"""VLM movement-reasoning service (2026-09-03; Denis: "better reasoning in the prediction for c").

A small open VLM (default Qwen2.5-VL-3B-Instruct) reads the drone's front camera image and the
task instruction and answers, in the movement vocabulary's own physical units, what the drone
should do over the next horizon: forward / right / up metres and a heading change, plus a short
trace (what it sees, what is done, what is next). The pin server converts those words into the
coarse command coordinates through the basis; the flow supplies everything finer.

    ~/code/vlmenv/bin/python vlm_reason_server.py --port 9190 [--model Qwen/Qwen2.5-VL-3B-Instruct]

POST /reason  json {"image_png_b64": str, "instruction": str, "flown_m": float,
                    "body_offset": [forward_m, right_m, up_m] (displacement from start, body frame),
                    "prev": [ {trace dicts of previous replans, most recent last} ] }
       -> json {"seen","done","next","forward_m","right_m","up_m","turn_deg","raw","ms"}
GET  /health  -> {"model": ..., "device": ...}
"""
import argparse
import base64
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

SYSTEM = (
    "You are the navigator of a small indoor quadrotor. You see its front camera image. "
    "Directions are relative to the camera: 'forward' is straight into the image, 'right' is toward the "
    "image's right edge, 'up' is toward the ceiling. The room contains free-standing rectangular gates "
    "(two posts and a top bar) and a table with a stuffed animal on it. "
    "A gate is passed by flying through the middle of its opening, level, keeping straight until it is behind you. "
    "The drone cruises at about 0.6 m/s, so a 2-second move is at most 1.2 m in total. "
    "When every gate in the task has been passed, fly to the stuffed animal and hover above it (all motion zero)."
)
DESCRIBE = (
    "Describe this camera image in two or three sentences: how many gates (pairs of vertical posts with a top bar) "
    "are visible, where each is in the image (left third / middle / right third, near or far), whether any gate's "
    "opening is directly ahead, whether the table with the stuffed animal is visible and where, and what is "
    "directly ahead of the camera."
)
USER = (
    "Task: {instruction}\n"
    "Flown so far: {flown:.1f} m. Displacement from the start point, in the current camera frame: "
    "forward {f:+.1f} m, right {r:+.1f} m, up {u:+.1f} m.\n"
    "{history}"
    "Using your description, decide the motion for the next 2 seconds. Output a JSON object with exactly these keys: "
    "done (string: which parts of the task are already completed, or none), "
    "next (string: the immediate sub-goal), "
    "forward_m, right_m, up_m (numbers in metres; negative right_m means left, negative forward_m means backward), "
    "turn_deg (number; positive turns right). Answer with the JSON object only."
)


class Reasoner:
    def __init__(self, model_id, min_side=448):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=min_side * min_side,
                                                  max_pixels=min_side * min_side * 2)
        self.model_id = model_id
        self.min_side = min_side
        self.lock = threading.Lock()

    def __call__(self, image, instruction, flown, body_offset, prev):
        from qwen_vl_utils import process_vision_info
        if image.width < self.min_side:
            image = image.resize((self.min_side, self.min_side), Image.BICUBIC)
        hist = ""
        if prev:
            last = prev[-3:]
            hist = "Your previous decisions (oldest first): " + "; ".join(
                f"[{p.get('next', '?')}: fwd {p.get('forward_m', 0):+.1f}, right {p.get('right_m', 0):+.1f}]" for p in last) + ".\n"
        # stage 1: perception only
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": DESCRIBE}]}]
        seen = self._gen(msgs, 120)
        # stage 2: decision, with the description in context
        msgs += [{"role": "assistant", "content": seen},
                 {"role": "user", "content": USER.format(instruction=instruction, flown=flown, f=body_offset[0],
                                                         r=body_offset[1], u=body_offset[2], history=hist)}]
        raw = self._gen(msgs, 160)
        d = parse(raw); d["seen"] = seen.strip()
        return d, raw

    def _gen(self, msgs, max_new):
        from qwen_vl_utils import process_vision_info
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = self.proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
        with self.lock, torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return self.proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    d = {}
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            for k in ("forward_m", "right_m", "up_m", "turn_deg"):
                mm = re.search(rf'"{k}"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
                if mm:
                    d[k] = float(mm.group(1))
            for k in ("seen", "done", "next"):
                mm = re.search(rf'"{k}"\s*:\s*"([^"]*)"', raw)
                if mm:
                    d[k] = mm.group(1)
    for k in ("forward_m", "right_m", "up_m", "turn_deg"):
        try:
            d[k] = float(d.get(k, 0.0))
        except (TypeError, ValueError):
            d[k] = 0.0
    for k in ("seen", "done", "next"):
        d[k] = str(d.get(k, ""))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--port", type=int, default=9190)
    a = ap.parse_args()
    R = Reasoner(a.model)
    print(f"[vlm_reason] {a.model} loaded; {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"model": a.model, "device": "cuda"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            img = Image.open(io.BytesIO(base64.b64decode(req["image_png_b64"]))).convert("RGB")
            t0 = time.time()
            d, raw = R(img, req["instruction"], float(req.get("flown_m", 0.0)),
                       req.get("body_offset", [0, 0, 0]), req.get("prev", []))
            d["raw"] = raw; d["ms"] = int((time.time() - t0) * 1000)
            self._send(200, d)

    print(f"[vlm_reason] ready on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
