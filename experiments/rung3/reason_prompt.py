"""ReasonPrompt (2026-09-03): the VLM's movement words as the coarse command.

Each replan, the drone's front image + the task instruction go to the vlm_reason_server; its
answer (forward / right / up metres and a heading change for the next ~2 s, plus a trace) is
converted from the camera frame to the world frame with the drone's own heading (state[3]),
laid out as a constant-velocity 50-step chunk in normalized action units, projected through U,
and ONLY the coarse coordinates named by `mode` replace the head's. The flow keeps the fine
words. No scene geometry, no targets, no triggers: the reasoner decides from what it sees.

modes (coordinate index = 4*band + axis; bands 0-6, 6-12, 12-25, 25-50; axes x,y,z,yaw):
  coarse_xyz   {8, 9, 10, 12, 13, 14}      (default)
  coarse_xy    {8, 9, 12, 13}
  coarse_all   {8..15}                      (also heading)
  all          all 16 (the reasoner's straight line, verbatim)
Every replan's trace is appended to `log_path` (jsonl) so the reasoning can be read afterwards.
"""
import base64
import io
import json
import os
import time
import urllib.request

import numpy as np
from PIL import Image

H, AD = 50, 32
MAX_STEP_M = 0.030   # per-step cap: ~1.2x demo pace (0.025 m/step)
MODES = {"coarse_xyz": [8, 9, 10, 12, 13, 14], "coarse_xy": [8, 9, 12, 13],
         "coarse_all": list(range(8, 16)), "all": list(range(16))}


class ReasonPrompt:
    def __init__(self, url, amean, astd, U, mode="coarse_xyz", log_path=""):
        self.url = url.rstrip("/") + "/reason"
        self.amean, self.astd, self.U = amean, astd, U
        self.mode = mode
        self.dims = MODES[mode]
        self.log_path = log_path
        self._st = {}
        urllib.request.urlopen(self.url.replace("/reason", "/health"), timeout=10).read()
        print(f"[reason] {url} mode={mode} dims={self.dims}", flush=True)

    @staticmethod
    def _png(img):
        b = io.BytesIO(); Image.fromarray(np.asarray(img, np.uint8)).save(b, format="PNG")
        return base64.b64encode(b.getvalue()).decode()

    def window(self, trial, obs, instruction):
        """-> (normalized (H, AD) chunk, trace dict). Always active."""
        st = self._st.setdefault(trial, {"start": None, "prev": [], "flown": 0.0, "last_pos": None, "k": 0})
        state = np.asarray(obs["observation/state"], np.float32).reshape(-1)
        pos, psi = state[:3], float(state[3])
        if st["start"] is None:
            st["start"] = pos.copy()
        if st["last_pos"] is not None:
            st["flown"] += float(np.linalg.norm(pos - st["last_pos"]))
        st["last_pos"] = pos.copy()
        # displacement from start, expressed in the current camera frame
        d = pos - st["start"]
        fwd = np.array([np.cos(psi), np.sin(psi)]); rgt = np.array([np.sin(psi), -np.cos(psi)])
        body_off = [float(d[:2] @ fwd), float(d[:2] @ rgt), float(d[2])]
        req = {"image_png_b64": self._png(obs["observation/image"]), "instruction": instruction,
               "flown_m": st["flown"], "body_offset": body_off, "prev": st["prev"][-3:]}
        t0 = time.time()
        r = urllib.request.Request(self.url, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
        ans = json.loads(urllib.request.urlopen(r, timeout=120).read())
        f, rt, up, turn = (float(ans.get(k, 0.0)) for k in ("forward_m", "right_m", "up_m", "turn_deg"))
        # camera frame -> world
        dx = f * np.cos(psi) + rt * np.sin(psi)
        dy = f * np.sin(psi) - rt * np.cos(psi)
        move = np.array([dx, dy, up], np.float32)
        per = move / H
        n = float(np.linalg.norm(per))
        if n > MAX_STEP_M:
            per *= MAX_STEP_M / n
        seg = np.zeros((H, 7), np.float32)
        seg[:, :3] = per
        seg[:, 3] = -np.radians(turn) / H          # yaw positive = toward +y (left); "turn right" is negative
        ch = np.zeros((H, AD), np.float32)
        ch[:, :4] = (seg[:, :4] - self.amean[:4]) / (self.astd[:4] + 1e-6)
        trace = {"trial": trial, "k": st["k"], "pos": pos.round(3).tolist(), "heading_deg": round(float(np.degrees(psi)), 1),
                 "seen": ans.get("seen", ""), "done": ans.get("done", ""), "next": ans.get("next", ""),
                 "forward_m": f, "right_m": rt, "up_m": up, "turn_deg": turn,
                 "world_move": move.round(3).tolist(), "vlm_ms": ans.get("ms"), "rt_ms": int((time.time() - t0) * 1000)}
        st["prev"].append({"next": trace["next"], "forward_m": f, "right_m": rt})
        st["k"] += 1
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(trace) + "\n")
        return ch, trace

    def compose(self, c_head, ch):
        c = np.asarray(c_head, np.float32).copy()
        cp = ch.reshape(-1) @ self.U
        c[self.dims] = cp[self.dims]
        return c
