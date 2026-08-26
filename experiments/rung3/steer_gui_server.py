"""LIVE steering GUI (in sim). Fly the pinned policy in the gsplat scene from a browser and
nudge the coarse command while it flies — the demo version of the steering interface.

Runs in the /tmp/tv env (gsplat + torch + openpi_client) and talks to an already-running pin
server (serve_gate_pin_prog4.py), which applies obs["snmvp_nudge"] = [dx, dy, dz] metres.

    /tmp/tv/bin/python steer_gui_server.py --port 8090 --policy-port 8839 --scene left
    ssh -L 8090:127.0.0.1:8090 <box>     then open http://127.0.0.1:8090

Endpoints: /            UI
           /stream.mjpg live view
           /set?x=&y=&z=  set the nudge (metres, applied to every command)
           /reset       restart the episode
           /state       JSON: position, nudge, commanded displacement, step
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

os.environ.setdefault("SCENE", "left")

STATE = {"pos": [0, 0, 1.5], "nudge": [0.0, 0.0, 0.0], "step": 0, "cmd_disp": [0, 0, 0],
         "running": True, "reset": False, "jpeg": None, "trace": []}
LOCK = threading.Lock()


def sim_loop(args):
    """Reuses gate_video_overlay's scene/render setup by importing it with SCENE preset."""
    import importlib
    import io

    from PIL import Image
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    # gate_video_overlay builds the scene at import and then flies NCH chunks; NCH=0 gives
    # us the scene, cameras and render helpers with no rollout and no video written.
    os.environ["SCENE"] = args.scene
    os.environ["SIDE"] = args.scene
    os.environ["NCH"] = "0"
    os.environ["PORT"] = str(args.policy_port)
    os.environ["OUT"] = "/tmp/_steer_gui_unused.mp4"
    os.environ["TRAJ"] = ""
    G = importlib.import_module("gate_video_overlay")

    pol = WebsocketClientPolicy(host="127.0.0.1", port=args.policy_port)
    prompt = args.prompt or f"go through the gate on the {args.scene} and hover over the stuffed animal"

    def episode():
        pos = np.array([0.0, 0.0, 1.5])
        yaw = 0.0
        first = True
        with LOCK:
            STATE["trace"] = [pos.tolist()]
        for step in range(args.max_chunks):
            with LOCK:
                if STATE["reset"]:
                    STATE["reset"] = False
                    return
                nudge = list(STATE["nudge"])
            imf, imw = G.obs_fwd(pos, yaw), G.obs_wrist(pos, yaw)
            o = {"observation/image": imf, "observation/wrist_image": imw,
                 "observation/state": np.array([pos[0], pos[1], pos[2], -yaw, 0, 0, 0], np.float32),
                 "prompt": prompt, "progress": min(1.0, step * 8 / 271.0),
                 "snmvp_nudge": nudge}
            if first:
                o["reset"] = True
                first = False
            act = np.asarray(pol.infer(o)["actions"])[:, :7]
            n = min(len(act), 8)
            cs = np.cumsum(act[:, :3], 0)[:n]
            for i in range(0, n, 2):
                wp = pos + cs[i]
                wy = yaw - float(act[:i + 1, 3].sum())
                frame = G.rend(wp, wy, G.Tbc_f, G.Kv, G.Wv, G.Hv)
                buf = io.BytesIO()
                Image.fromarray(np.asarray(frame, np.uint8)).save(buf, "JPEG", quality=72)
                with LOCK:
                    STATE["jpeg"] = buf.getvalue()
                time.sleep(0.02)
            pos = pos + cs[-1]
            yaw = yaw - float(act[:n, 3].sum())
            with LOCK:
                STATE["pos"] = [round(float(v), 3) for v in pos]
                STATE["step"] = step
                STATE["cmd_disp"] = [round(float(v), 3) for v in np.asarray(cs[-1])]
                STATE["trace"].append(pos.tolist())
            if abs(pos[0]) > 60 or abs(pos[1]) > 60:
                return

    while True:
        episode()
        time.sleep(0.5)


PAGE = """<!doctype html><meta charset=utf-8><title>Live steering</title>
<style>
body{background:#14161b;color:#e9e9e5;font:15px/1.5 "Avenir Next",ui-sans-serif,system-ui;margin:0;padding:20px 26px}
h1{font-size:1.2rem;margin:0 0 3px} p.l{color:#99a0ab;margin:0 0 16px;max-width:70ch}
.wrap{display:flex;gap:22px;flex-wrap:wrap}
img{width:640px;max-width:100%;border-radius:10px;background:#000}
.panel{background:#1d1f25;border:1px solid #32353d;border-radius:11px;padding:16px 18px;min-width:290px}
label{display:block;margin:12px 0 4px;font:600 .8rem ui-monospace,Menlo,monospace;color:#99a0ab}
input[type=range]{width:100%;accent-color:#7db4e6}
.val{font:600 .95rem ui-monospace,Menlo,monospace}
button{background:#7db4e6;color:#12141a;border:0;border-radius:7px;padding:8px 14px;font-weight:700;cursor:pointer;margin-top:14px}
button.ghost{background:#2a2d35;color:#e9e9e5}
.read{margin-top:16px;font:.86rem/1.7 ui-monospace,Menlo,monospace;color:#99a0ab}
.read b{color:#e9e9e5;font-weight:600}
</style>
<h1>Live steering — the pinned command, nudged in flight</h1>
<p class="l">The sliders add metres of net displacement to the coarse command every step. The policy is
unchanged; only what it is asked to do changes. Set everything to zero to fly the baseline.</p>
<div class="wrap">
 <img id="v" src="/stream.mjpg" alt="live view">
 <div class="panel">
  <label>altitude nudge (z) <span class="val" id="lz">0.00 m</span></label>
  <input id="z" type="range" min="-0.6" max="0.6" step="0.05" value="0">
  <label>lateral nudge (x) <span class="val" id="lx">0.00 m</span></label>
  <input id="x" type="range" min="-0.6" max="0.6" step="0.05" value="0">
  <label>forward nudge (y) <span class="val" id="ly">0.00 m</span></label>
  <input id="y" type="range" min="-0.6" max="0.6" step="0.05" value="0">
  <button onclick="zero()">zero all</button>
  <button class="ghost" onclick="fetch('/reset')">restart episode</button>
  <div class="read">
   position <b id="p">—</b><br>chunk displacement <b id="d">—</b><br>step <b id="s">—</b>
  </div>
 </div>
</div>
<script>
const ids=["x","y","z"];
function send(){
  const q=ids.map(i=>`${i}=${document.getElementById(i).value}`).join("&");
  ids.forEach(i=>document.getElementById("l"+i).textContent=(+document.getElementById(i).value).toFixed(2)+" m");
  fetch("/set?"+q);
}
ids.forEach(i=>document.getElementById(i).addEventListener("input",send));
function zero(){ids.forEach(i=>document.getElementById(i).value=0);send();}
setInterval(async()=>{const r=await (await fetch("/state")).json();
  document.getElementById("p").textContent=r.pos.join(", ");
  document.getElementById("d").textContent=r.cmd_disp.join(", ");
  document.getElementById("s").textContent=r.step;},400);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        elif u.path == "/set":
            q = parse_qs(u.query)
            with LOCK:
                for i, k in enumerate("xyz"):
                    if k in q:
                        STATE["nudge"][i] = float(q[k][0])
            self.send_response(204); self.end_headers()
        elif u.path == "/reset":
            with LOCK:
                STATE["reset"] = True
            self.send_response(204); self.end_headers()
        elif u.path == "/state":
            with LOCK:
                body = json.dumps({k: STATE[k] for k in ("pos", "nudge", "step", "cmd_disp")}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        elif u.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        j = STATE["jpeg"]
                    if j:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--policy-port", type=int, default=8839)
    ap.add_argument("--scene", default="left")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-chunks", type=int, default=60)
    args = ap.parse_args()
    threading.Thread(target=sim_loop, args=(args,), daemon=True).start()
    print(f"[steer-gui] http://127.0.0.1:{args.port}  (policy on {args.policy_port})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
